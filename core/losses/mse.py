from .base import *
from typing import Literal
from pydantic import BaseModel, ConfigDict


class MSELossParams(BaseModel):
    """
    Class for MSE loss parameters, compatible with `nn.MSELoss`.
    """
    model_config = ConfigDict(frozen=True)
    
    reduction: Literal["none", "mean", "sum"] = "mean"

    def asdict(self) -> Dict[str, Any]:
        """
        Returns a dictionary of valid parameters for `nn.MSELoss`.

        Returns:
            Dict[str, Any]: Dictionary of parameters for `nn.MSELoss`.
        """
        loss_kwargs = self.model_dump()
        return {k: v for k, v in loss_kwargs.items() if v is not None}  # Remove None values


class MSELoss(BaseLoss):
    """
    Custom loss function wrapper for `nn.MSELoss` with tracking of loss metrics.
    """

    def __init__(self, params: Optional[MSELossParams] = None):
        """
        Initializes the loss function with optional MSELoss parameters.

        Args:
            params (Optional[MSELossParams]): Parameters for `nn.MSELoss` (default: None).
        """
        super().__init__(params=params)
        _mse_params = params.asdict() if params is not None else {}

        # Initialize MSE loss with user-provided parameters or PyTorch defaults
        self.mse_loss = nn.MSELoss(**_mse_params)

        # Using CumulativeAverage from MONAI to track loss metrics
        self.loss_mse_metric = CumulativeAverage()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between true values and predictions.

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
        
        loss = self.mse_loss(outputs, target)
        self.loss_mse_metric.append(loss.item())

        return loss

    def get_loss_metrics(self) -> Dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            Dict[str, float]: A dictionary containing the average MSE loss.
        """
        return {
            "loss": round(self.loss_mse_metric.aggregate().item(), 4),
        }

    def reset_metrics(self):
        """Resets the stored loss metrics."""
        self.loss_mse_metric.reset()
