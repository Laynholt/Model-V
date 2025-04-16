from typing import Any, Dict, Tuple
from pydantic import BaseModel, ConfigDict
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from .base import BaseScheduler


class MultiStepLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.MultiStepLR`."""
    model_config = ConfigDict(frozen=True)

    milestones: Tuple[int, ...] = (30, 80)   # List of epoch indices for LR decay
    gamma: float = 0.1                     # Multiplicative factor of learning rate decay
    last_epoch: int = -1

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.MultiStepLR`."""
        return self.model_dump()


class MultiStepLRScheduler(BaseScheduler):
    """
    Wrapper around torch.optim.lr_scheduler.MultiStepLR.
    """

    def __init__(self, optimizer: optim.Optimizer, params: MultiStepLRParams):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            params (MultiStepLRParams): Scheduler parameters.
        """
        super().__init__(optimizer, params)
        self.scheduler = MultiStepLR(optimizer, **params.asdict())