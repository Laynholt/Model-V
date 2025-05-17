from torch import optim
from pydantic import BaseModel


class BaseScheduler:
    """
    Abstract base class for learning rate schedulers.
    Wraps a PyTorch LR scheduler and provides a unified interface.
    """

    def __init__(self, optimizer: optim.Optimizer, params: BaseModel) -> None:
        self.scheduler: optim.lr_scheduler.LRScheduler | None = None

    def step(self) -> None:
        """
        Performs a single scheduler step. This typically updates the learning rate
        based on the current epoch or step count.
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def get_last_lr(self) -> list[float]:
        """
        Returns the most recent learning rate(s).
        """
        return self.scheduler.get_last_lr() if self.scheduler else []
