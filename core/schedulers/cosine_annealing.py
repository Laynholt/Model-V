from .base import BaseScheduler

from typing import Any
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pydantic import BaseModel, ConfigDict


class CosineAnnealingLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.CosineAnnealingLR`."""
    model_config = ConfigDict(frozen=True)

    T_max: int = 100        # Maximum number of iterations
    eta_min: float = 0.0    # Minimum learning rate
    last_epoch: int = -1


    def asdict(self) -> dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.CosineAnnealingLR`."""
        return self.model_dump()


class CosineAnnealingLRScheduler(BaseScheduler):
    """
    Wrapper around torch.optim.lr_scheduler.CosineAnnealingLR.
    """

    def __init__(self, optimizer: optim.Optimizer, params: CosineAnnealingLRParams) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            params (CosineAnnealingLRParams): Scheduler parameters.
        """
        super().__init__(optimizer, params)
        self.scheduler = CosineAnnealingLR(optimizer, **params.asdict())