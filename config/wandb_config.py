from pydantic import BaseModel, model_validator
from typing import Any


class WandbConfig(BaseModel):
    """
    Configuration for Weights & Biases logging.
    """
    use_wandb: bool = False         # Whether to enable WandB logging
    project: str | None = None      # WandB project name
    group: str | None = None        # WandB group name
    entity: str | None = None       # WandB entity (user or team)
    name: str | None = None         # Name of the run
    id: str | None = None           # Id of the run
    tags: list[str] | None = None   # List of tags for the run
    notes: str | None = None        # Notes or description for the run
    save_code: bool = True          # Whether to save the code to WandB

    @model_validator(mode="after")
    def validate_wandb(self) -> "WandbConfig":
        if self.use_wandb:
            if not self.project:
                raise ValueError("When use_wandb=True, 'project' must be provided")
        return self
    
    def asdict(self) -> dict[str, Any]:
        """
        Return a dict of all W&B parameters, excluding 'use_wandb' and any None values.
        """
        return self.model_dump(exclude_none=True, exclude={"use_wandb"})