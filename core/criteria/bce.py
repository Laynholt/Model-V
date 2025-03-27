from .base import *
from typing import List, Literal, Union
from pydantic import BaseModel, ConfigDict


class BCELossParams(BaseModel):
    """
    Class for handling parameters for both `nn.BCELoss` and `nn.BCEWithLogitsLoss`.
    """
    model_config = ConfigDict(frozen=True)
    
    weight: Optional[List[Union[int, float]]] = None  # Sample weights
    reduction: Literal["none", "mean", "sum"] = "mean"  # Reduction method
    pos_weight: Optional[List[Union[int, float]]] = None  # Used only for BCEWithLogitsLoss

    def asdict(self, with_logits: bool = False) -> Dict[str, Any]:
        """
        Returns a dictionary of valid parameters for `nn.BCEWithLogitsLoss` and `nn.BCELoss`.

        - If `with_logits=False`, `pos_weight` is **removed** to avoid errors.
        - Ensures only the valid parameters are passed based on the loss function.

        Args:
            with_logits (bool): If `True`, includes `pos_weight` (for `nn.BCEWithLogitsLoss`).
                               If `False`, removes `pos_weight` (for `nn.BCELoss`).

        Returns:
            Dict[str, Any]: Filtered dictionary of parameters.
        """
        loss_kwargs = self.model_dump()
        if not with_logits:
            loss_kwargs.pop("pos_weight", None)  # Remove pos_weight if using BCELoss
        
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

    def __init__(self, bce_params: Optional[BCELossParams] = None, with_logits: bool = False):
        """
        Initializes the loss function with optional BCELoss parameters.

        Args:
            bce_params (Optional[Dict[str, Any]]): Parameters for nn.BCELoss (default: None).
        """
        super().__init__()
        _bce_params = bce_params.asdict(with_logits=with_logits) if bce_params is not None else {}
        
        # Initialize loss functions with user-provided parameters or PyTorch defaults
        self.bce_loss = nn.BCEWithLogitsLoss(**_bce_params) if with_logits else nn.BCELoss(**_bce_params)
        
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


    def get_loss_metrics(self) -> Dict[str, float]:
        """
        Retrieves the tracked loss metrics.

        Returns:
            Dict[str, float]: A dictionary containing the average BCE loss.
        """
        return {
            "loss": round(self.loss_bce_metric.aggregate().item(), 4),
        }


    def reset_metrics(self):
        """Resets the stored loss metrics."""
        self.loss_bce_metric.reset()
