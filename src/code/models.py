import logging
import torch
import torch.nn.functional as F
import transformers

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

    model_config = transformers.GPTNeoXConfig(**config.model_kwargs["model_config"])
    model = transformers.AutoModelForCausalLM.from_config(model_config)
    model.output_key = "logits"
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    return model


def load_model(config, model_path, device="cuda"):
    if "lm" in config.model_kwargs.model_type:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        model.output_key = "logits"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path) if config.model_kwargs.model_type == "llm" else None
        return model.to(device), tokenizer
    elif "bayes" in config.model_kwargs.model_type:
        model = UnigramPredictor(config)
        return model, None

################# Bayesian Models ########################

class UnigramPredictor:
    """Dirichlet–multinomial unigram baseline: predicts each next token from the
    running category counts plus a flat prior. Exposes the same logits interface
    as the NN models, so the Evaluator treats it identically."""

    def __init__(self, config):
        self.output_key = "logits"
        self.model_type = config.model_kwargs.model_type
        self.prior = torch.ones(config.num_dims)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input_ids, **kwargs):
        """Return log posterior-mean probs, shape (batch, context_length + 1, num_dims)."""
        input_ids = input_ids[:, 1:]  # drop start token
        alphas = self.prior.to(input_ids.device)
        counts = F.one_hot(input_ids, num_classes=alphas.shape[0]).float().cumsum(dim=1)
        posterior = (counts + alphas) / (counts + alphas).sum(-1, keepdim=True)
        prior_pred = (alphas / alphas.sum()).expand(input_ids.shape[0], 1, -1)
        probs = torch.cat([prior_pred, posterior], dim=1)
        return {self.output_key: probs.log()}