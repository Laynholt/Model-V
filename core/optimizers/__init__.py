import torch.optim as optim
from pydantic import BaseModel
from typing import Dict, Final, Tuple, Type, List, Any, Union

from .adam import AdamParams
from .adamw import AdamWParams
from .sgd import SGDParams

__all__ = [
    "OptimizerRegistry",
    "AdamParams", "AdamWParams", "SGDParams"
]

class OptimizerRegistry:
    """Registry for optimizers and their parameter classes with case-insensitive lookup."""
    
    # Single dictionary storing both optimizer classes and parameter classes.
    __OPTIMIZERS: Final[Dict[str, Dict[str, Type[Any]]]] = {
        "SGD": {
            "class": optim.SGD,
            "params": SGDParams,
        },
        "Adam": {
            "class": optim.Adam,
            "params": AdamParams,
        },
        "AdamW": {
            "class": optim.AdamW,
            "params": AdamWParams,
        },
    }
    
    @classmethod
    def __get_entry(cls, name: str) -> Dict[str, Type[Any]]:
        """
        Private method to retrieve the optimizer entry from the registry using case-insensitive lookup.
        
        Args:
            name (str): The name of the optimizer.
        
        Returns:
            Dict[str, Type[Any]]: A dictionary containing the keys 'class' and 'params'.
        
        Raises:
            ValueError: If the optimizer is not found.
        """
        name_lower = name.lower()
        mapping = {key.lower(): key for key in cls.__OPTIMIZERS}
        original_key = mapping.get(name_lower)
        if original_key is None:
            raise ValueError(
                f"Optimizer '{name}' not found! Available options: {list(cls.__OPTIMIZERS.keys())}"
            )
        return cls.__OPTIMIZERS[original_key]
    
    @classmethod
    def get_optimizer_class(cls, name: str) -> Type[optim.Optimizer]:
        """
        Retrieves the optimizer class by name (case-insensitive).
        
        Args:
            name (str): Name of the optimizer.
        
        Returns:
            Type[optim.Optimizer]: The optimizer class.
        """
        entry = cls.__get_entry(name)
        return entry["class"]
    
    @classmethod
    def get_optimizer_params(cls, name: str) -> Union[Type[BaseModel], Tuple[Type[BaseModel]]]:
        """
        Retrieves the optimizer parameter class by name (case-insensitive).
        
        Args:
            name (str): Name of the optimizer.
        
        Returns:
            Union[Type[BaseModel], Tuple[Type[BaseModel]]]: The optimizer parameter class or a tuple of parameter classes.
        """
        entry = cls.__get_entry(name)
        return entry["params"]
    
    @classmethod
    def get_available_optimizers(cls) -> Tuple[str, ...]:
        """
        Returns a tuple of available optimizer names in their original case.
        
        Returns:
            Tuple[str]: Tuple of available optimizer names.
        """
        return tuple(cls.__OPTIMIZERS.keys())
