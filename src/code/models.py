import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from omegaconf import OmegaConf

from .config import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##### General init function #####
def init_model(config):
    """Initialize a model from config (trained from scratch); returns the model."""
    # Re-seed right before init so weight initialization is reproducible.
    set_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_type = config.model_kwargs.model_type
    if model_type in READOUT_CLASSES:
        mc = OmegaConf.to_container(config.model_kwargs.model_config, resolve=True)
        model = READOUT_CLASSES[model_type](ReadoutConfig(**mc))
    else:
        model_config = transformers.GPTNeoXConfig(**config.model_kwargs["model_config"])
        model = transformers.AutoModelForCausalLM.from_config(model_config)
        model.output_key = "logits"
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config, model_path, device="cuda"):
    model_type = config.model_kwargs.model_type
    if model_type in READOUT_CLASSES:
        model = READOUT_CLASSES[model_type].from_pretrained(model_path)
        model.output_key = "logits"
        return model.to(device), None
    if model_type in ("transformer", "llm"):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        model.output_key = "logits"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path) if model_type == "llm" else None
        return model.to(device), tokenizer
    raise ValueError(f"Unsupported model_type for loading: {model_type}")


################# Readout-head ladder (representational convergence) ########################
# A ladder of architectures that all learn a shared item embedding table (read out by the
# Evaluator via get_input_embeddings) so we can test that different architectures trained on
# the same generative process converge to the same predicted geometry. Implemented as
# PreTrainedModel subclasses so they slot into the HF Trainer, CustomSaveCallback, and
# load_model unchanged. All consume the pointwise batch: input_ids (B, arity) -> logits (B, C).

class ReadoutConfig(transformers.PretrainedConfig):
    """Config for the readout heads. Sizing fields (vocab_size/num_classes/arity) are patched
    from the dataset in train.py, mirroring how the transformer's vocab_size is patched."""
    model_type = "readout"

    def __init__(self, vocab_size=2, num_classes=2, embed_dim=16, arity=1,
                 hidden_size=64, num_layers=2, init_scale=0.02, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.arity = arity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale


class _ReadoutBase(transformers.PreTrainedModel):
    config_class = ReadoutConfig
    main_input_name = "input_ids"
    _tied_weights_keys = None
    all_tied_weights_keys = {}          # these heads tie no weights (transformers>=5 reads this)

    def __init__(self, config):
        super().__init__(config)
        self.output_key = "logits"
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.embedding.weight, std=config.init_scale)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, module):
        self.embedding = module

    def _finish(self, logits, labels):
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


class KLinearReadout(_ReadoutBase):
    """Unified k-linear head: shared embedding E + rank-(k+1) readout tensor W[c, i1..ik]
    contracted with e(x1)...e(xk). k=1 -> linear W.e(x); k=2 -> bilinear e(a)^T M_c e(b).
    arity is read from the config (patched from dataset.arity)."""

    def __init__(self, config):
        super().__init__(config)
        self.arity = config.arity
        shape = (config.num_classes,) + (config.embed_dim,) * config.arity
        self.readout = nn.Parameter(torch.randn(*shape) * config.init_scale)

    def forward(self, input_ids, labels=None, **kwargs):
        e = self.embedding(input_ids)                       # (B, arity, d)
        letters = "ijklmn"[: self.arity]
        eq = f"c{letters}," + ",".join("b" + l for l in letters) + "->bc"
        logits = torch.einsum(eq, self.readout, *[e[:, j] for j in range(self.arity)])
        return self._finish(logits, labels)

    def item_reps(self, item_ids, start_token=None):
        return self.embedding(item_ids)                     # pre-decode code == the embedding


class MLPReadout(_ReadoutBase):
    """MLP over the concatenated input-item embeddings."""

    def __init__(self, config):
        super().__init__(config)
        layers, in_dim = [], config.embed_dim * config.arity
        for _ in range(config.num_layers):
            layers += [nn.Linear(in_dim, config.hidden_size), nn.ReLU()]
            in_dim = config.hidden_size
        layers.append(nn.Linear(in_dim, config.num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, labels=None, **kwargs):
        e = self.embedding(input_ids).flatten(1)            # (B, arity*d)
        return self._finish(self.mlp(e), labels)

    def item_reps(self, item_ids, start_token=None):
        return self.mlp[:-1](self.embedding(item_ids))      # penultimate activation (pre-decode)


class RNNReadout(_ReadoutBase):
    """GRU causal language model over walk sequences (a genuine recurrent sequence model): predicts
    the next token at each position. The pre-decode representation is the GRU hidden state."""

    def __init__(self, config):
        super().__init__(config)
        self.rnn = nn.GRU(config.embed_dim, config.hidden_size,
                          num_layers=config.num_layers, batch_first=True)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)   # next-token over the vocab

    def forward(self, input_ids, labels=None, **kwargs):
        h, _ = self.rnn(self.embedding(input_ids))          # (B, T, hidden)
        logits = self.head(h)
        out = {"logits": logits}
        if labels is not None:                              # causal next-token loss (shift)
            out["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                          labels[:, 1:].reshape(-1))
        return out

    def item_reps(self, item_ids, start_token):
        """Hidden-state activation after reading each item in the minimal context [start, item]."""
        start = torch.full((len(item_ids), 1), int(start_token), dtype=torch.long, device=item_ids.device)
        seq = torch.cat([start, item_ids.view(-1, 1)], dim=1)
        h, _ = self.rnn(self.embedding(seq))
        return h[:, -1]


READOUT_CLASSES = {"klinear": KLinearReadout, "mlp": MLPReadout, "rnn": RNNReadout}


################# Superposition (over-capacity) ########################

def train_sparse_autoencoder(n_features, d=2, sparsity=0.05, steps=3000,
                             batch_size=1024, lr=1e-2, seed=0):
    """Nonlinear sparse autoencoder on sparse feature activations (demo_graphs §2 finale).

    When more co-occurring features than dimensions must be packed (n_features > d), the
    optimal code *superposes* them: the learned encoder columns spread into a regular polygon
    (e.g. a pentagon for n=5, d=2) rather than collapsing. Returns W of shape (d, n_features).
    This is standalone (unsupervised reconstruction), not part of the Trainer/Evaluator path.
    """
    g = torch.Generator().manual_seed(seed)
    W = nn.Parameter(0.1 * torch.randn(d, n_features, generator=g))
    b = nn.Parameter(torch.zeros(n_features))
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        mask = (torch.rand(batch_size, n_features, generator=g) < sparsity).float()
        x = mask * torch.rand(batch_size, n_features, generator=g)
        xr = torch.relu((x @ W.T) @ W + b)
        loss = ((x - xr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().cpu().numpy()