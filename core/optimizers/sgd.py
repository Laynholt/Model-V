import torch
from torch import optim
from typing import Any, Iterable
from pydantic import BaseModel, ConfigDict

from .base import BaseOptimizer


class SGDParams(BaseModel):
    """Configuration for `torch.optim.SGD` optimizer."""
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-3                            # Learning rate
    momentum: float = 0.0                       # Momentum factor
    dampening: float = 0.0                      # Dampening for momentum
    weight_decay: float = 0.0                   # L2 penalty
    nesterov: bool = False                      # Enables Nesterov momentum

    def asdict(self) -> dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.SGD`."""
        return self.model_dump()
    
    
class SGDOptimizer(BaseOptimizer):
    """
    Wrapper around torch.optim.SGD.
    """

    def __init__(self, model_params: Iterable[torch.nn.Parameter], optim_params: SGDParams) -> None:
        """
        Initializes the SGD optimizer with given parameters.

        Args:
            model_params (Iterable[Parameter]): Parameters to optimize.
            optim_params (SGDParams): Optimizer parameters.
        """
        super().__init__(model_params, optim_params)
        self.optim = optim.SGD(model_params, **optim_params.asdict())