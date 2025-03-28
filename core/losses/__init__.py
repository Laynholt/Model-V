from typing import Dict, Final, Tuple, Type, List, Any, Union
from pydantic import BaseModel

from .base import BaseLoss
from .ce import CrossEntropyLoss, CrossEntropyLossParams
from .bce import BCELoss, BCELossParams
from .mse import MSELoss, MSELossParams
from .mse_with_bce import BCE_MSE_Loss

__all__ = [
    "CriterionRegistry",
    "CrossEntropyLoss", "BCELoss", "MSELoss", "BCE_MSE_Loss",
    "CrossEntropyLossParams", "BCELossParams", "MSELossParams"
]

class CriterionRegistry:
    """Registry of loss functions and their parameter classes with case-insensitive lookup."""
    
    __CRITERIONS: Final[Dict[str, Dict[str, Any]]] = {
        "CrossEntropyLoss": {
            "class": CrossEntropyLoss,
            "params": CrossEntropyLossParams,
        },
        "BCELoss": {
            "class": BCELoss,
            "params": BCELossParams,
        },
        "MSELoss": {
            "class": MSELoss,
            "params": MSELossParams,
        },
        "BCE_MSE_Loss": {
            "class": BCE_MSE_Loss,
            "params": (BCELossParams, MSELossParams),
        },
    }
    
    @classmethod
    def __get_entry(cls, name: str) -> Dict[str, Any]:
        """
        Private method to retrieve the criterion entry from the registry using case-insensitive lookup.
        
        Args:
            name (str): The name of the loss function.
        
        Returns:
            Dict[str, Any]: A dictionary containing the keys 'class' and 'params'.
        
        Raises:
            ValueError: If the loss function is not found.
        """
        name_lower = name.lower()
        mapping = {key.lower(): key for key in cls.__CRITERIONS}
        original_key = mapping.get(name_lower)
        if original_key is None:
            raise ValueError(
                f"Criterion '{name}' not found! Available options: {list(cls.__CRITERIONS.keys())}"
            )
        return cls.__CRITERIONS[original_key]
    
    @classmethod
    def get_criterion_class(cls, name: str) -> Type[BaseLoss]:
        """
        Retrieves the loss function class by name (case-insensitive).
        
        Args:
            name (str): Name of the loss function.
        
        Returns:
            Type[BaseLoss]: The loss function class.
        """
        entry = cls.__get_entry(name)
        return entry["class"]
    
    @classmethod
    def get_criterion_params(cls, name: str) -> Union[Type[BaseModel], Tuple[Type[BaseModel]]]:
        """
        Retrieves the loss function parameter class (or classes) by name (case-insensitive).
        
        Args:
            name (str): Name of the loss function.
        
        Returns:
            Union[Type[BaseModel], Tuple[Type[BaseModel]]]: The loss function parameter class or a tuple of parameter classes.
        """
        entry = cls.__get_entry(name)
        return entry["params"]
    
    @classmethod
    def get_available_criterions(cls) -> Tuple[str, ...]:
        """
        Returns a tuple of available loss function names in their original case.
        
        Returns:
            Tuple[str]: Tuple of available loss function names.
        """
        return tuple(cls.__CRITERIONS.keys())
