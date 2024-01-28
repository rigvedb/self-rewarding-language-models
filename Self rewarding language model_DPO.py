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
