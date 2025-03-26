from typing import Any, Dict, Tuple
from pydantic import BaseModel, ConfigDict


class MultiStepLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.MultiStepLR`."""
    model_config = ConfigDict(frozen=True)

    milestones: Tuple[int, ...] = (30, 80)   # List of epoch indices for LR decay
    gamma: float = 0.1                     # Multiplicative factor of learning rate decay

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.MultiStepLR`."""
        return self.model_dump()