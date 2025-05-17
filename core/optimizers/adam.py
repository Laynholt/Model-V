import torch
from torch import optim
from typing import Any, Iterable
from pydantic import BaseModel, ConfigDict

from .base import BaseOptimizer

class AdamParams(BaseModel):
    """Configuration for `torch.optim.Adam` optimizer."""
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-3                            # Learning rate
    betas: tuple[float, float] = (0.9, 0.999)   # Coefficients for computing running averages
    eps: float = 1e-8                           # Term added to denominator for numerical stability
    weight_decay: float = 0.0                   # L2 regularization
    amsgrad: bool = False                       # Whether to use the AMSGrad variant

    def asdict(self) -> dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.Adam`."""
        return self.model_dump()
    
    
class AdamOptimizer(BaseOptimizer):
    """
    Wrapper around torch.optim.Adam.
    """

    def __init__(self, model_params: Iterable[torch.nn.Parameter], optim_params: AdamParams) -> None:
        """
        Initializes the Adam optimizer with given parameters.

        Args:
            model_params (Iterable[Parameter]): Parameters to optimize.
            optim_params (AdamParams): Optimizer parameters.
        """
        super().__init__(model_params, optim_params)
        self.optim = optim.Adam(model_params, **optim_params.asdict())