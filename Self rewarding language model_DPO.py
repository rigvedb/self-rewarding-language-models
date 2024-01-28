from copy import deepcopy
from collections import namedtuple

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from beartype import beartype
from beartype.typing import Optional, Callable

from einx import get_at

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

from pytorch_custom_utils.accelerate_utils import (
    auto_unwrap_model,
    model_forward_contexts
)

from torchtyping import TensorType



def exists(v):
    return v is not None

def dreeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False #disabling gradient calculations for that parameters
        
        
def log_prob_from_model_and_sequence(model,seq,eps=1e-20):
    logits = model(seq)
    prob = logits.softmax(dim = -1)
    return get_at('b n [c], b n -> b n', prob,indices).clamp(min = eps).log()
