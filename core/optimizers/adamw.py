from typing import Any, Dict, Tuple
from pydantic import BaseModel, ConfigDict

class AdamWParams(BaseModel):
    """Configuration for `torch.optim.AdamW` optimizer."""
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-3                            # Learning rate
    betas: Tuple[float, ...] = (0.9, 0.999)   # Adam coefficients
    eps: float = 1e-8                           # Numerical stability
    weight_decay: float = 1e-2                  # L2 penalty (AdamW uses decoupled weight decay)
    amsgrad: bool = False                       # Whether to use the AMSGrad variant

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.AdamW`."""
        return self.model_dump()