import torch
from torch import optim
from typing import Any, Iterable
from pydantic import BaseModel, ConfigDict

from .base import BaseOptimizer

class AdamWParams(BaseModel):
    """Configuration for `torch.optim.AdamW` optimizer."""
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-3                            # Learning rate
    betas: tuple[float, ...] = (0.9, 0.999)   # Adam coefficients
    eps: float = 1e-8                           # Numerical stability
    weight_decay: float = 1e-2                  # L2 penalty (AdamW uses decoupled weight decay)
    amsgrad: bool = False                       # Whether to use the AMSGrad variant

    def asdict(self) -> dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.AdamW`."""
        return self.model_dump()
    

class AdamWOptimizer(BaseOptimizer):
    """
    Wrapper around torch.optim.AdamW.
    """

    def __init__(self, model_params: Iterable[torch.nn.Parameter], optim_params: AdamWParams) -> None:
        """
        Initializes the AdamW optimizer with given parameters.

        Args:
            model_params (Iterable[Parameter]): Parameters to optimize.
            optim_params (AdamWParams): Optimizer parameters.
        """
        super().__init__(model_params, optim_params)
        self.optim = optim.AdamW(model_params, **optim_params.asdict())