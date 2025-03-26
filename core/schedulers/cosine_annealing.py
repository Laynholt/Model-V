from typing import Any, Dict
from pydantic import BaseModel, ConfigDict


class CosineAnnealingLRParams(BaseModel):
    """Configuration for `torch.optim.lr_scheduler.CosineAnnealingLR`."""
    model_config = ConfigDict(frozen=True)

    T_max: int = 100        # Maximum number of iterations
    eta_min: float = 0.0    # Minimum learning rate

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid parameters for `torch.optim.lr_scheduler.CosineAnnealingLR`."""
        return self.model_dump()
