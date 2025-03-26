from typing import Any, Dict
from pydantic import BaseModel, ConfigDict


class StepLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.StepLR`."""
    model_config = ConfigDict(frozen=True)

    step_size: int = 30     # Period of learning rate decay
    gamma: float = 0.1      # Multiplicative factor of learning rate decay

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.StepLR`."""
        return self.model_dump()