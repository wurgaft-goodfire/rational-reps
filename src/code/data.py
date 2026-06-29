import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from .config import Config
import os

##### Init dataset #####
def init_dataset(config, overwrite_data: bool = False):
    """
    Initialize a dataset from a given config.
    """
    if config.setting == "balls-and-urns":
        return BallsAndUrnsDataset(config, overwrite_data=overwrite_data)
    else:
        raise ValueError(f"Unsupported setting: {config.setting}. Only 'balls-and-urns' and 'graph-random-walk' settings are supported.")

##### Dataset class #####
class MixtureDataset(Dataset):
    def __init__(
        self, config: "Config", overwrite_data: bool = False
    ):
        self.config = config
        self.num_dims = config.num_dims
        self.num_tasks = config.num_tasks
        self.infinite_task_diversity = config.num_tasks == "inf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.config.random_seed)

        if not self.infinite_task_diversity:
            if not os.path.exists(self.config.data_path) or overwrite_data:
                self.dataset_dict = self.make_dataset(num_tasks=self.num_tasks, save=True)
            else:
                self.dataset_dict = torch.load(self.config.data_path)
            self.tasks = self.dataset_dict["tasks"].to(self.device)
            self.num_tasks = len(self.tasks)
            self._process_task_prior()
        else:
            self.tasks = None

    def __len__(self):
        """since getitem returns a batch, return 1"""
        return 1
    
    def _process_task_prior(self):
        """ 
        Process the task prior.
        """
        if self.config.task_prior == 'uniform':
            self.task_prior = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        elif self.config.task_prior == 'power_law':
            self.task_prior = F.normalize(torch.pow(torch.arange(1, self.num_tasks + 1, device=self.device), -self.config.task_prior_exponent), p=1, dim=0)
        else:
            raise ValueError("Unsupported prior type. Use 'uniform' or 'power_law'.")

    def __gettask__(self, batch_size):
        """
        Returns a batch of tasks from the dataset.
        """
        if self.infinite_task_diversity:
            return self.make_dataset(batch_size, save=False), None
        task_idx = torch.multinomial(self.task_prior, batch_size, replacement=True, generator=self.rng)
        return self.tasks[task_idx].to(self.device), task_idx

    def make_dataset(self, num_tasks=None, save=True):
        """
        Implemented in child class.
        """
        raise NotImplementedError("make_dataset method must be implemented in child class.")

##### Balls and urns dataset #####

class BallsAndUrnsDataset(MixtureDataset):
    def __init__(self, config: Config, overwrite_data: bool = False):
        """
        Args:
            config: Config object
            overwrite_data: Whether to overwrite the dataset if it already exists
        """
        super().__init__(config, overwrite_data)

        # Set up token mappings after parent initialization
        if hasattr(config, "tokenizer"):
            self.item_tokens = torch.tensor([config.tokenizer.encode(word, add_special_tokens=False)[0] for word in config.word_list])
            self.start_token = torch.as_tensor(config.tokenizer.bos_token_id).long()
        else:
            self.item_tokens = torch.arange(0, config.num_dims)
            self.start_token = torch.tensor(len(self.item_tokens)).long()

    @property
    def vocab_size(self):
        """Number of token ids the model must span: item tokens + start token."""
        return int(self.start_token.item()) + 1

    @property
    def seq_len(self):
        """Sequence length the model sees: context length + start token."""
        return self.config.training_kwargs["context_length"] + 1

    def make_dataset(self, num_tasks, save=False):
        """Generate balls and urns dataset"""
        def dirichlet_flat(n, d, generator):
            # Draw Exp(1) as -log(U), then normalize rows
            # Clamp to prevent log(0) which causes inf values
            x = -torch.log(torch.rand((n, d), generator=generator, device=generator.device).clamp(min=1e-10))
            return x / x.sum(dim=-1, keepdim=True)
        # Generate adjacency matrices for each task
        task_distributions = dirichlet_flat(num_tasks, self.config.num_dims, generator=self.rng) # (num_tasks, num_dims)
        if save:
            torch.save({"tasks": task_distributions.cpu()}, self.config.data_path)
        else:
            return task_distributions.to(self.device)

    def __getitem__(self, i=None):
        tasks, _ = self.__gettask__(self.config.training_kwargs["batch_size"])
        # sample from categorical distribution based on distributions provided in task
        input_ids = torch.multinomial(tasks, self.config.training_kwargs["context_length"], replacement=True, generator=self.rng) # (batch_size, context_length)
        # tokenize if tokenizer is provided
        if hasattr(self.config, "tokenizer"):
            input_ids = self.item_tokens[input_ids]
        # add start token
        input_ids = torch.cat([self.start_token.to(self.device).expand(input_ids.shape[0], 1), input_ids], dim=1)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "eval_labels": tasks.clone().unsqueeze(1).repeat(1, self.config.training_kwargs["context_length"], 1), # has size (batch_size, context_length, num_dims), given by duplicating the task distribution
        }
    
    @staticmethod
    def compute_metrics(eval_pred, return_all=True):
        """Compute KL divergence loss for next token prediction"""
        logits, labels = eval_pred
        logits = logits[:, :-1] # remove last token

        results = {}

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        kl_div_fn = nn.KLDivLoss(reduction="none")
        with torch.no_grad():
            loss = (
                kl_div_fn(log_probs, labels)
                .sum(dim=-1)
                .reshape(logits.shape[0], -1)
            )

        results['loss'] = loss.mean().item()

        if return_all:
            results['all_loss'] = loss.detach().cpu()

        return results