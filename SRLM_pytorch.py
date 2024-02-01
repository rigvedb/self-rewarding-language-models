from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, ConcatDataset

from numpy.lib.format import open_memmap

from beartype import beartype
from beartype.typing import Optional, List, Union

from self_rewarding_lm_pytorch.dpo import (
    DPO,
    EarlyStopper,
    DPOTrainer,
    set_dropout_,
    adam_optimizer_with_linear_decay
)

from einops import rearrange

from accelerate import Accelerator

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# constants
# llm-as-judge prompt
# https://openreview.net/forum?id=uccHPGDlao


# config, allowing for different types of reward prompting
# colocate with functions for extracting the response and reward

REWARD_PROMPT_CONFIG = dict(
    default = dict(
        prompt = DEFAULT_LLM_AS_JUDGE_PROMPT
    )
)

# sft trainer

class SFTTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        accelerator: Accelerator,
        train_dataset: Union[List[Dataset], Dataset],
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        weight_decay: float = 0.,
        ignore_index: int = -1,
        adam_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = accelerator
        self.model = model

        self.num_epochs = num_epochs
        self.ignore_index = ignore_index

        if isinstance(train_dataset, list):
            train_dataset = ConcatDataset(train_dataset)

        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        self.model, self.train_dataloader = self.accelerator.prepare(self.model, self.train_dataloader)

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            accelerator = accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        self.val_dataloader = None
        if exists(val_dataset):
            self.val_dataloader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

    def forward(self):
        for epoch in self.num_epochs:
            for seq in self.train_dataloader:
                seq, labels = seq[: :-1], seq[:, 1:]

                logits = self.model(seq)

                ce_loss = F.cross_entropy(
                    rearrange(logits, 'b n l -> b l n'),
                    labels,
                    ignore_index = self.ignore_index
                )

                self.accelerator.backward(ce_loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

# reward generator class

class RewardGenerator(Module):
    @beartype
    def __init__(
        self,
        model_generate_prompt: Module,
        model: Module,
        num_preference_pairs: int,
        num_candidate_responses: int = 4,
        gen_temperature: float = 0.7,
        gen_nucleus_p: float = 0.9,
        eval_temperature: float = 0.7,
        eval_nucleus_p: float = 0.9,
        num_evals_to_average: int = 3,
        *,
        reward_config: dict,
        dataset_file_location: str = './dpo-train-set.memmap.npy'
    ):
        super().__init__()
        self.model = model
        self.num_candidate_responses = num_candidate_responses

        self.num_preference_pairs = num_preference_pairs

        self.gen_nucleus_p = gen_nucleus_p
        self.gen_temperature = gen_temperature

        self.eval_nucleus_p = eval_nucleus_p
        self.eval_temperature = eval_temperature

        self.num_evals_to_average = num_evals_to_average
        self.dataset_file_location = dataset_file_location

    def forward(self) -> Dataset:
        raise NotImplementedError
