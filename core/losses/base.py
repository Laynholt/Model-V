import abc
import torch
from pydantic import BaseModel


class BaseLoss(torch.nn.Module, abc.ABC):
    """Custom loss function combining BCEWithLogitsLoss and MSE losses for cell recognition and distinction."""

    def __init__(self, params: BaseModel | None = None) -> None:
        super().__init__()
        
        
    @abc.abstractmethod
    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between true labels and prediction outputs.

        Args:
            outputs (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth.

        Returns:
            torch.Tensor: The total loss value.
        """
        

    @abc.abstractmethod
    def get_loss_metrics(self) -> dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            dict(str, float): A dictionary containing the loss name and average loss value.
        """
        

    @abc.abstractmethod
    def reset_metrics(self) -> None:
        """Resets the stored loss metrics."""
    