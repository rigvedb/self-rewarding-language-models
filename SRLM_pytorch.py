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