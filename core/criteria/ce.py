from .base import *
from typing import List, Literal, Union
from pydantic import BaseModel, ConfigDict


class CrossEntropyLossParams(BaseModel):
    """
    Class for handling parameters for `nn.CrossEntropyLoss`.
    """
    model_config = ConfigDict(frozen=True)
    
    weight: Optional[List[Union[int, float]]] = None
    ignore_index: int = -100
    reduction: Literal["none", "mean", "sum"] = "mean"
    label_smoothing: float = 0.0
    
    def asdict(self):
        """
        Returns a dictionary of valid parameters for `nn.CrossEntropyLoss`.

        Returns:
            Dict[str, Any]: Dictionary of parameters for nn.CrossEntropyLoss.
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

    def __init__(self, ce_params: Optional[CrossEntropyLossParams] = None):
        """
        Initializes the loss function with optional CrossEntropyLoss parameters.

        Args:
            ce_params (Optional[Dict[str, Any]]): Parameters for nn.CrossEntropyLoss (default: None).
        """
        super().__init__()
        _ce_params = ce_params.asdict() if ce_params is not None else {}
        
        # Initialize loss functions with user-provided parameters or PyTorch defaults
        self.ce_loss = nn.CrossEntropyLoss(**_ce_params)
        
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


    def get_loss_metrics(self) -> Dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            Dict[str, float]: A dictionary containing the average CrossEntropy loss.
        """
        return {
            "loss": round(self.loss_ce_metric.aggregate().item(), 4),
        }


    def reset_metrics(self):
        """Resets the stored loss metrics."""
        self.loss_ce_metric.reset()
