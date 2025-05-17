from .base import BaseLoss
import torch
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
from monai.metrics.cumulative_average import CumulativeAverage


class CrossEntropyLossParams(BaseModel):
    """
    Class for handling parameters for `nn.CrossEntropyLoss`.
    """
    model_config = ConfigDict(frozen=True)
    
    weight: list[int | float] | None = None
    ignore_index: int = -100
    reduction: Literal["none", "mean", "sum"] = "mean"
    label_smoothing: float = 0.0
    
    def asdict(self) -> dict[str, Any]:
        """
        Returns a dictionary of valid parameters for `nn.CrossEntropyLoss`.

        Returns:
            dict(str, Any): Dictionary of parameters for nn.CrossEntropyLoss.
        """
        loss_kwargs = self.model_dump()
        
        weight = loss_kwargs.get("weight")
        if weight is not None:
            loss_kwargs["weight"] = torch.Tensor(weight)
        
        return {k: v for k, v in loss_kwargs.items() if v is not None}  # Remove None values
    


class CrossEntropyLoss(BaseLoss):
    """
    Custom loss function wrapper for `nn.CrossEntropyLoss` with tracking of loss metrics.
    """

    def __init__(self, params: CrossEntropyLossParams | None = None) -> None:
        """
        Initializes the loss function with optional CrossEntropyLoss parameters.

        Args:
            params (CrossEntropyLossParams | None): Parameters for nn.CrossEntropyLoss (default: None).
        """
        super().__init__(params=params)
        _ce_params = params.asdict() if params is not None else {}
        
        # Initialize loss functions with user-provided parameters or PyTorch defaults
        self.ce_loss = torch.nn.CrossEntropyLoss(**_ce_params)
        
        # Using CumulativeAverage from MONAI to track loss metrics
        self.loss_ce_metric = CumulativeAverage()


    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between true labels and prediction outputs.

        Args:
            outputs (torch.Tensor): Model predictions of shape (batch_size, channels, H, W).
            target (torch.Tensor): Ground truth labels of shape (batch_size, H, W).

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
        
        loss = self.ce_loss(outputs, target)
        self.loss_ce_metric.append(loss.item())
        
        return loss


    def get_loss_metrics(self) -> dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            dict(str, float): A dictionary containing the average CrossEntropy loss.
        """
        return {
            "loss": round(self.loss_ce_metric.aggregate().item(), 4),
        }


    def reset_metrics(self) -> None:
        """Resets the stored loss metrics."""
        self.loss_ce_metric.reset()
