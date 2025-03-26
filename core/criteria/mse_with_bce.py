from .base import *
from .bce import BCELossParams
from .mse import MSELossParams


class BCE_MSE_Loss(BaseLoss):
    """
    Custom loss function combining BCE (with or without logits) and MSE losses for cell recognition and distinction.
    """

    def __init__(
        self,
        num_classes: int,
        bce_params: Optional[BCELossParams] = None,
        mse_params: Optional[MSELossParams] = None,
        bce_with_logits: bool = False,
    ):
        """
        Initializes the loss function with optional BCE and MSE parameters.

        Args:
            num_classes (int): Number of output classes, used for target shifting.
            bce_params (Optional[BCELossParams]): Parameters for BCEWithLogitsLoss or BCELoss (default: None).
            mse_params (Optional[MSELossParams]): Parameters for MSELoss (default: None).
            bce_with_logits (bool): If True, uses BCEWithLogitsLoss; otherwise, uses BCELoss.
        """
        super().__init__()

        self.num_classes = num_classes

        # Process BCE parameters
        _bce_params = bce_params.asdict(bce_with_logits) if bce_params is not None else {}

        # Choose BCE loss function
        self.bce_loss = (
            nn.BCEWithLogitsLoss(**_bce_params) if bce_with_logits else nn.BCELoss(**_bce_params)
        )

        # Process MSE parameters
        _mse_params = mse_params.asdict() if mse_params is not None else {}

        # Initialize MSE loss
        self.mse_loss = nn.MSELoss(**_mse_params)

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
            outputs[:, -self.num_classes:], target[:, self.num_classes:2 * self.num_classes].float()
        )

        # Cell Distinction Loss
        gradflow_loss = 0.5 * self.mse_loss(
            outputs[:, :2 * self.num_classes], 5.0 * target[:, 2 * self.num_classes:]
        )

        # Total loss
        total_loss = cellprob_loss + gradflow_loss

        # Track individual losses
        self.loss_bce_metric.append(cellprob_loss.item())
        self.loss_mse_metric.append(gradflow_loss.item())

        return total_loss

    def get_loss_metrics(self) -> Dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            Dict[str, float]: A dictionary containing the average BCE and MSE loss.
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
