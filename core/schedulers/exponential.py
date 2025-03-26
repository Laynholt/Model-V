from typing import Any, Dict
from pydantic import BaseModel, ConfigDict


class ExponentialLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.ExponentialLR`."""
    model_config = ConfigDict(frozen=True)

    gamma: float = 0.95 # Multiplicative factor of learning rate decay

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.ExponentialLR`."""
        return self.model_dump()