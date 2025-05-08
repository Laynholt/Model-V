from pydantic import BaseModel, model_validator
from typing import Any, Dict, Optional


class WandbConfig(BaseModel):
    """
    Configuration for Weights & Biases logging.
    """
    use_wandb: bool = False            # Whether to enable WandB logging
    project: Optional[str] = None      # WandB project name
    group: Optional[str] = None        # WandB group name
    entity: Optional[str] = None       # WandB entity (user or team)
    name: Optional[str] = None         # Name of the run
    tags: Optional[list[str]] = None   # List of tags for the run
    notes: Optional[str] = None        # Notes or description for the run
    save_code: bool = True             # Whether to save the code to WandB

    @model_validator(mode="after")
    def validate_wandb(self) -> "WandbConfig":
        if self.use_wandb:
            if not self.project:
                raise ValueError("When use_wandb=True, 'project' must be provided")
        return self
    
    def asdict(self) -> Dict[str, Any]:
        """
        Return a dict of all W&B parameters, excluding 'use_wandb' and any None values.
        """
        return self.model_dump(exclude_none=True, exclude={"use_wandb"})