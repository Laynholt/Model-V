from .base import BaseLoss
import torch
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
from monai.metrics.cumulative_average import CumulativeAverage


class BCELossParams(BaseModel):
    """
    Class for handling parameters for both `nn.BCELoss` and `nn.BCEWithLogitsLoss`.
    """
    model_config = ConfigDict(frozen=True)
    
    with_logits: bool = False
    
    weight: list[int | float] | None = None  # Sample weights
    reduction: Literal["none", "mean", "sum"] = "mean"  # Reduction method
    pos_weight: list[int | float] | None = None  # Used only for BCEWithLogitsLoss

    def asdict(self) -> dict[str, Any]:
        """
        Returns a dictionary of valid parameters for `nn.BCEWithLogitsLoss` and `nn.BCELoss`.

        - If `with_logits=False`, `pos_weight` is **removed** to avoid errors.
        - Ensures only the valid parameters are passed based on the loss function.

        Returns:
            dict(str, Any): Filtered dictionary of parameters.
        """
        loss_kwargs = self.model_dump()
        if not self.with_logits:
            loss_kwargs.pop("pos_weight", None)  # Remove pos_weight if using BCELoss
        loss_kwargs.pop("with_logits", None)
        
        weight = loss_kwargs.get("weight")
        pos_weight = loss_kwargs.get("pos_weight")
        
        if weight is not None:
            loss_kwargs["weight"] = torch.Tensor(weight)
            
        if pos_weight is not None:
            loss_kwargs["pos_weight"] = torch.Tensor(pos_weight) 
        
        return {k: v for k, v in loss_kwargs.items() if v is not None}  # Remove None values


class BCELoss(BaseLoss):
    """
    Custom loss function wrapper for `nn.BCELoss and nn.BCEWithLogitsLoss` with tracking of loss metrics.
    """

    def __init__(self, params: BCELossParams | None = None) -> None:
        """
        Initializes the loss function with optional BCELoss parameters.

        Args:
            params (BCELossParams | None): Parameters for nn.BCELoss (default: None).
        """
        super().__init__(params=params)
        with_logits = params.with_logits if params is not None else False
        _bce_params = params.asdict() if params is not None else {}
        
        # Initialize loss functions with user-provided parameters or PyTorch defaults
        self.bce_loss = (
            torch.nn.BCEWithLogitsLoss(**_bce_params) 
            if with_logits 
            else torch.nn.BCELoss(**_bce_params)
        )
        
        # Using CumulativeAverage from MONAI to track loss metrics
        self.loss_bce_metric = CumulativeAverage()


    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between true labels and prediction outputs.

        Args:
            outputs (torch.Tensor): Model predictions of shape (batch_size, channels, H, W).
            target (torch.Tensor): Ground truth labels in one-hot format.

        Returns:
            torch.Tensor: The total loss value.
        """
        # Ensure target is on the same device as outputs
        assert (
            target.device == outputs.device
        ), (
            "Target tensor must be moved to the same device as outputs "
            "before calling forward()."
        )
        
        loss = self.bce_loss(outputs, target)
        self.loss_bce_metric.append(loss.item())
        
        return loss


    def get_loss_metrics(self) -> dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            dict(str, float): A dictionary containing the average BCE loss.
        """
        return {
            "loss": round(self.loss_bce_metric.aggregate().item(), 4),
        }


    def reset_metrics(self) -> None:
        """Resets the stored loss metrics."""
        self.loss_bce_metric.reset()
