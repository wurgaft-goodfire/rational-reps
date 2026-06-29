import logging
import torch
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
    def __init__(self, config):
        self.output_key = "logits"
        self.model_type = config.model_kwargs.model_type
        self.prior = torch.ones(config.num_dims)

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input_ids, **kwargs):
        """
        Unigram predictor model incorporating a flat Dirichlet prior.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, context_length) containing the input ids.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, context_length + 1, num_dims) containing the log posterior probabilities.
        """
        input_ids = input_ids[:, 1:]  # Remove start token
        device = input_ids.device  # Get the device from input_ids tensor

        # Ensure alphas is a torch tensor on the correct device
        alphas = torch.as_tensor(self.prior, device=device)  # Shape: (num_dims,)
        num_dims = alphas.shape[0]
        batch_size, context_length = input_ids.shape

        # One-hot encode input_ids
        one_hot_sequences = torch.nn.functional.one_hot(
            input_ids, num_classes=num_dims
        ).float()
        # Shape: (batch_size, context_length, num_dims)

        # Compute cumulative counts
        cumsum = torch.cumsum(
            one_hot_sequences, dim=1
        )  # Shape: (batch_size, context_length, num_dims)

        # Add alphas to cumulative counts to get posterior counts
        alphas_expanded = alphas.view(1, 1, num_dims)  # Shape: (1, 1, num_dims)
        posterior_counts = (
            cumsum + alphas_expanded
        )  # Broadcasting over batch and seq_length

        # Compute total counts at each time step (sum over categories)
        total_counts = posterior_counts.sum(
            dim=-1, keepdim=True
        )  # Shape: (batch_size, context_length, 1)

        # Compute posterior mean (posterior_counts / total_counts)
        posterior_mean = (
            posterior_counts / total_counts
        )  # Shape: (batch_size, context_length, num_dims)

        # Compute initial probabilities (prior predictive) using alphas
        total_alphas = alphas.sum()  # Scalar
        initial_probs = alphas / total_alphas  # Shape: (num_dims,)
        initial_probs = initial_probs.view(1, 1, num_dims).expand(
            batch_size, -1, -1
        )  # Shape: (batch_size, 1, num_dims)

        # Concatenate initial probabilities with posterior means
        posterior_mean_probs = torch.cat(
            [initial_probs, posterior_mean], dim=1
        )  # Shape: (batch_size, context_length + 1, num_dims)

        log_posterior_mean_probs = torch.log(posterior_mean_probs)

        return {self.output_key: log_posterior_mean_probs}