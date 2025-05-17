from .base import BaseScheduler

from typing import Any
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from pydantic import BaseModel, ConfigDict


class ExponentialLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.ExponentialLR`."""
    model_config = ConfigDict(frozen=True)

    gamma: float = 0.95 # Multiplicative factor of learning rate decay
    last_epoch: int = -1

    def asdict(self) -> dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.ExponentialLR`."""
        return self.model_dump()


class ExponentialLRScheduler(BaseScheduler):
    """
    Wrapper around torch.optim.lr_scheduler.ExponentialLR.
    """

    def __init__(self, optimizer: optim.Optimizer, params: ExponentialLRParams) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            params (ExponentialLRParams): Scheduler parameters.
        """
        super().__init__(optimizer, params)
        self.scheduler = ExponentialLR(optimizer, **params.asdict())