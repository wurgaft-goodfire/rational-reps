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
    cls = MODEL_CLASSES[config.model_kwargs.model_type]
    mc = OmegaConf.to_container(config.model_kwargs.model_config, resolve=True)
    model = cls(cls.config_class(**mc))
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


class _CausalLMBase:
    """What the causal LMs share: item probing by a forward pass on the minimal context."""

    def input_reps(self, input_ids):
        """Pre-decode code for a batch of model inputs ``(B, T)``: the final hidden state at the
        last input position, which is where the outcome is predicted."""
        return self(input_ids, output_hidden_states=True)["hidden_states"][-1][:, -1, :]


def load_model(config, model_path, device="cuda"):
    model_type = config.model_kwargs.model_type
    if model_type in MODEL_CLASSES:
        model = MODEL_CLASSES[model_type].from_pretrained(model_path)
        model.output_key = "logits"
        return model.to(device), None
    if model_type == "llm":
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        model.output_key = "logits"
        return model.to(device), transformers.AutoTokenizer.from_pretrained(model_path)
    raise ValueError(f"Unsupported model_type for loading: {model_type}")


################# Model ladder (representational convergence) ########################

class ModelConfig(transformers.PretrainedConfig):
    """Config for the from-scratch models. Sizing fields (vocab_size/num_classes/arity) are patched
    from the dataset in train.py, mirroring how the transformer's vocab_size is patched."""
    model_type = "ladder"

    def __init__(self, vocab_size=2, num_classes=2, embed_dim=16, arity=1,
                 hidden_size=64, num_layers=2, init_scale=0.02, l1=0.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.arity = arity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.l1 = l1                                     # SAE sparsity penalty


class _ModelBase(transformers.PreTrainedModel):
    config_class = ModelConfig
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
        """``labels`` is the exact outcome distribution (``distribution_labels``) for the
        readout heads, so cross-entropy takes it as class probabilities."""
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


class LinearModel(_ModelBase):
    """Unified k-linear head: shared embedding E + rank-(k+1) readout tensor W[c, i1..ik]
    contracted with e(x1)...e(xk). k=1 -> linear W.e(x); k=2 -> bilinear e(a)^T M_c e(b).
    arity is read from the config (patched from dataset.arity)."""

    def __init__(self, config):
        super().__init__(config)
        self.arity = config.arity
        shape = (config.num_classes,) + (config.embed_dim,) * config.arity
        self.readout = nn.Parameter(torch.randn(*shape) * config.init_scale)

    def forward(self, input_ids, distribution_labels=None, **kwargs):
        input_ids = input_ids[:, :self.arity]               # the outcome column is the LM's target
        e = self.embedding(input_ids)                       # (B, arity, d)
        letters = "ijklmn"[: self.arity]
        eq = f"c{letters}," + ",".join("b" + l for l in letters) + "->bc"
        logits = torch.einsum(eq, self.readout, *[e[:, j] for j in range(self.arity)])
        return self._finish(logits, distribution_labels)

    def input_reps(self, input_ids):
        """Pre-decode code per input tuple: the embedding. ``None`` with several slots — there the
        readout is a tensor contracting them, so no vector code exists for the decoder to read."""
        if self.config.arity > 1:
            return None
        return self.embedding(input_ids).flatten(1)


class MLPModel(_ModelBase):
    """MLP over the concatenated input-item embeddings."""

    def __init__(self, config):
        super().__init__(config)
        layers, in_dim = [], config.embed_dim * config.arity
        for _ in range(config.num_layers):
            layers += [nn.Linear(in_dim, config.hidden_size), nn.ReLU()]
            in_dim = config.hidden_size
        layers.append(nn.Linear(in_dim, config.num_classes, bias=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, distribution_labels=None, **kwargs):
        e = self.embedding(input_ids[:, :self.config.arity]).flatten(1)   # (B, arity*d)
        return self._finish(self.mlp(e), distribution_labels)

    def input_reps(self, input_ids):
        """Penultimate activation for a batch of input tuples: the slots are embedded, concatenated
        and run through every layer but the readout."""
        return self.mlp[:-1](self.embedding(input_ids).flatten(1))


class RNNModel(_CausalLMBase, _ModelBase):
    """GRU causal language model over walk sequences (a genuine recurrent sequence model): predicts
    the next token at each position. The pre-decode representation is the GRU hidden state."""

    def __init__(self, config):
        super().__init__(config)
        self.rnn = nn.GRU(config.embed_dim, config.hidden_size,
                          num_layers=config.num_layers, batch_first=True)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)   # next-token readout

    def forward(self, input_ids, labels=None, output_hidden_states=False, **kwargs):
        h, _ = self.rnn(self.embedding(input_ids))          # (B, T, hidden)
        logits = self.head(h)
        out = {"logits": logits}
        if output_hidden_states:                            # the recurrent state at every position
            out["hidden_states"] = (h,)
        if labels is not None:                              # causal next-token loss (shift)
            out["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                          labels[:, 1:].reshape(-1))
        return out


class SAEModel(_ModelBase):
    """Sparse autoencoder over the item feature codes z: h = relu(We z + be), logits Wd h
    (bias-free), CE against the target distribution plus ``l1 * mean|h|``. The embedding is
    the frozen feature table (``load_features``), so it trains from the same pointwise batches
    as the rest of the ladder. Bottleneck = hidden_size."""

    def __init__(self, config):
        super().__init__(config)
        self.embedding.weight.requires_grad_(False)         # the feature table is data, not a parameter
        self.encoder = nn.Linear(config.embed_dim, config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.num_classes, bias=False)
        for lin in (self.encoder, self.decoder):
            nn.init.normal_(lin.weight, std=config.init_scale)
        nn.init.zeros_(self.encoder.bias)

    def load_features(self, features):
        """Install the per-item feature codes (n_items, embed_dim) as the frozen embedding. The
        vocabulary also names the outcomes, which the SAE never reads, so those rows stay zero."""
        codes = torch.as_tensor(features, dtype=self.embedding.weight.dtype)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:len(codes)] = codes

    def _code(self, item_ids):
        return torch.relu(self.encoder(self.embedding(item_ids).flatten(1)))

    def forward(self, input_ids, distribution_labels=None, **kwargs):
        h = self._code(input_ids[:, :self.config.arity])
        out = self._finish(self.decoder(h), distribution_labels)
        if distribution_labels is not None:
            out["loss"] = out["loss"] + self.config.l1 * h.abs().mean()
        return out

    def input_reps(self, input_ids):
        return self._code(input_ids)                        # the bottleneck code


class TransformerModel(_CausalLMBase, transformers.GPTNeoXForCausalLM):
    """GPT-NeoX causal LM in the ladder interface: shifted next-token CE, plus item probing."""
    output_key = "logits"


MODEL_CLASSES = {"linear": LinearModel, "mlp": MLPModel, "rnn": RNNModel,
                   "sae": SAEModel, "transformer": TransformerModel}