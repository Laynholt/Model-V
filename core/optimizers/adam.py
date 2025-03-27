from typing import Any, Dict, Tuple
from pydantic import BaseModel, ConfigDict


class AdamParams(BaseModel):
    """Configuration for `torch.optim.Adam` optimizer."""
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-3                            # Learning rate
    betas: Tuple[float, float] = (0.9, 0.999)   # Coefficients for computing running averages
    eps: float = 1e-8                           # Term added to denominator for numerical stability
    weight_decay: float = 0.0                   # L2 regularization
    amsgrad: bool = False                       # Whether to use the AMSGrad variant

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.Adam`."""
        return self.model_dump()