import torch
from torch import optim
from pydantic import BaseModel
from typing import Any, Iterable


class BaseOptimizer:
    """Custom loss function combining BCEWithLogitsLoss and MSE losses for cell recognition and distinction."""

    def __init__(self, model_params: Iterable[torch.nn.Parameter], optim_params: BaseModel) -> None:
        super().__init__()
        self.optim: optim.Optimizer | None = None
        

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clears the gradients of all optimized tensors.

        Args:
            set_to_none (bool): If True, sets gradients to None instead of zero.
                                This can reduce memory usage and improve performance.
                                (Introduced in PyTorch 1.7+)
        """
        if self.optim is not None:
            self.optim.zero_grad(set_to_none=set_to_none)


    def step(self, closure: Any | None = None) -> Any:
        """
        Performs a single optimization step (parameter update).

        Args:
            closure (Any | None): A closure that reevaluates the model and returns the loss.
                                          This is required for optimizers like LBFGS that need multiple forward passes.

        Returns:
            Any: The return value depends on the specific optimizer implementation.
        """
        if self.optim is not None:
            return self.optim.step(closure=closure)