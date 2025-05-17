from pydantic import BaseModel
from typing import Final, Type, Any

from .base import BaseOptimizer
from .adam import AdamParams, AdamOptimizer
from .adamw import AdamWParams, AdamWOptimizer
from .sgd import SGDParams, SGDOptimizer

__all__ = [
    "OptimizerRegistry", "BaseOptimizer",
    "AdamParams", "AdamWParams", "SGDParams",
    "AdamOptimizer", "AdamWOptimizer", "SGDOptimizer"
]

class OptimizerRegistry:
    """Registry for optimizers and their parameter classes with case-insensitive lookup."""
    
    # Single dictionary storing both optimizer classes and parameter classes.
    __OPTIMIZERS: Final[dict[str, dict[str, Type[Any]]]] = {
        "SGD": {
            "class": SGDOptimizer,
            "params": SGDParams,
        },
        "Adam": {
            "class": AdamOptimizer,
            "params": AdamParams,
        },
        "AdamW": {
            "class": AdamWOptimizer,
            "params": AdamWParams,
        },
    }
    
    @classmethod
    def __get_entry(cls, name: str) -> dict[str, Type[Any]]:
        """
        Private method to retrieve the optimizer entry from the registry using case-insensitive lookup.
        
        Args:
            name (str): The name of the optimizer.
        
        Returns:
            dict(str, Type(Any)): A dictionary containing the keys 'class' and 'params'.
        
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
    def get_optimizer_class(cls, name: str) -> Type[BaseOptimizer]:
        """
        Retrieves the optimizer class by name (case-insensitive).
        
        Args:
            name (str): Name of the optimizer.
        
        Returns:
            Type(BaseOptimizer): The optimizer class.
        """
        entry = cls.__get_entry(name)
        return entry["class"]
    
    @classmethod
    def get_optimizer_params(cls, name: str) -> Type[BaseModel]:
        """
        Retrieves the optimizer parameter class by name (case-insensitive).
        
        Args:
            name (str): Name of the optimizer.
        
        Returns:
            Type(BaseModel): The optimizer parameter class.
        """
        entry = cls.__get_entry(name)
        return entry["params"]
    
    @classmethod
    def get_available_optimizers(cls) -> tuple[str, ...]:
        """
        Returns a tuple of available optimizer names in their original case.
        
        Returns:
            Tuple(str): Tuple of available optimizer names.
        """
        return tuple(cls.__OPTIMIZERS.keys())
