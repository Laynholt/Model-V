import torch.optim as optim
from pydantic import BaseModel
from typing import List, Optional


class BaseScheduler:
    """
    Abstract base class for learning rate schedulers.
    Wraps a PyTorch LR scheduler and provides a unified interface.
    """

    def __init__(self, optimizer: optim.Optimizer, params: BaseModel):
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

    def step(self) -> None:
        """
        Performs a single scheduler step. This typically updates the learning rate
        based on the current epoch or step count.
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        """
        Returns the most recent learning rate(s).
        """
        return self.scheduler.get_last_lr() if self.scheduler else []
