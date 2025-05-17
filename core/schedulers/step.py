from .base import BaseScheduler

from typing import Any
from torch import optim
from torch.optim.lr_scheduler import StepLR
from pydantic import BaseModel, ConfigDict


class StepLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.StepLR`."""
    model_config = ConfigDict(frozen=True)

    step_size: int = 30     # Period of learning rate decay
    gamma: float = 0.1      # Multiplicative factor of learning rate decay
    last_epoch: int = -1

    def asdict(self) -> dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.StepLR`."""
        return self.model_dump()
    


class StepLRScheduler(BaseScheduler):
    """
    Wrapper around torch.optim.lr_scheduler.StepLR.
    """

    def __init__(self, optimizer: optim.Optimizer, params: StepLRParams) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            params (StepLRParams): Scheduler parameters.
        """
        super().__init__(optimizer, params)
        self.scheduler = StepLR(optimizer, **params.asdict())