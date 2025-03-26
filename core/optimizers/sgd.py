import torch
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict


class SGDParams(BaseModel):
    """Configuration for `torch.optim.SGD` optimizer."""
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-3                            # Learning rate
    momentum: float = 0.0                       # Momentum factor
    dampening: float = 0.0                      # Dampening for momentum
    weight_decay: float = 0.0                   # L2 penalty
    nesterov: bool = False                      # Enables Nesterov momentum

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.SGD`."""
        return self.model_dump()