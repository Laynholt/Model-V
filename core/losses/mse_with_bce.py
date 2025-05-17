from .base import BaseLoss
from .bce import BCELossParams
from .mse import MSELossParams

import torch
from typing import Any
from pydantic import BaseModel, ConfigDict
from monai.metrics.cumulative_average import CumulativeAverage


class BCE_MSE_LossParams(BaseModel):
    """
    Class for handling parameters for `nn.MSELoss` with `nn.BCELoss`.
    """
    model_config = ConfigDict(frozen=True)
    
    num_classes: int = 1
    bce_params: BCELossParams = BCELossParams()
    mse_params: MSELossParams = MSELossParams()
    
    def asdict(self) -> dict[str, Any]:
        """
        Returns a dictionary of valid parameters for `nn.BCELoss` and `nn.MSELoss`.

        Returns:
            dict(str, Any): Dictionary of parameters.
        """
        
        return {
            "num_classes": self.num_classes,
            "bce_params": self.bce_params.asdict(),
            "mse_params": self.mse_params.asdict()
        }
    

class BCE_MSE_Loss(BaseLoss):
    """
    Custom loss function combining BCE (with or without logits) and MSE losses for cell recognition and distinction.
    """

    def __init__(self, params: BCE_MSE_LossParams | None = None):
        """
        Initializes the loss function with optional BCE and MSE parameters.
        """
        super().__init__(params=params)

        _params = params if params is not None else BCE_MSE_LossParams()

        self.num_classes = _params.num_classes

        # Process BCE parameters
        _bce_params = _params.bce_params.asdict()

        # Choose BCE loss function
        self.bce_loss = (
            torch.nn.BCEWithLogitsLoss(**_bce_params)
            if _params.bce_params.with_logits 
            else torch.nn.BCELoss(**_bce_params)
        )

        # Process MSE parameters
        _mse_params = _params.mse_params.asdict()

        # Initialize MSE loss
        self.mse_loss = torch.nn.MSELoss(**_mse_params)

        # Using CumulativeAverage from MONAI to track loss metrics
        self.loss_bce_metric = CumulativeAverage()
        self.loss_mse_metric = CumulativeAverage()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between true labels and prediction outputs.

        Args:
            outputs (torch.Tensor): Model predictions of shape (batch_size, channels, H, W).
            target (torch.Tensor): Ground truth labels of shape (batch_size, channels, H, W).

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
  
        # Cell Recognition Loss
        cellprob_loss = self.bce_loss(
            outputs[:, -self.num_classes:], (target[:, -self.num_classes:] > 0).float()
        )

        # Cell Distinction Loss
        gradflow_loss = 0.5 * self.mse_loss(
            outputs[:, :2 * self.num_classes], 5.0 * target[:, :2 * self.num_classes]
        )

        # Total loss
        total_loss = cellprob_loss + gradflow_loss

        # Track individual losses
        self.loss_bce_metric.append(cellprob_loss.item())
        self.loss_mse_metric.append(gradflow_loss.item())

        return total_loss

    def get_loss_metrics(self) -> dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            dict(str, float): A dictionary containing the average BCE and MSE loss.
        """
        return {
            "bce_loss": round(self.loss_bce_metric.aggregate().item(), 4),
            "mse_loss": round(self.loss_mse_metric.aggregate().item(), 4),
            "loss": round(
                self.loss_bce_metric.aggregate().item() + self.loss_mse_metric.aggregate().item(), 4
            ),
        }

    def reset_metrics(self):
        """Resets the stored loss metrics."""
        self.loss_bce_metric.reset()
        self.loss_mse_metric.reset()
