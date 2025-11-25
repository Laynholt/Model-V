from torch import nn
from typing import Final, Type, Any
from pydantic import BaseModel

from .mediar_v import MediarV, MediarVParams


__all__ = [
    "ModelRegistry",
    "MediarV",
    "MediarVParams"
]


class ModelRegistry:
    """Registry for models and their parameter classes with case-insensitive lookup."""
    
    # Single dictionary storing both model classes and parameter classes.
    __MODELS: Final[dict[str, dict[str, Type[Any]]]] = {
        "MediarV": {
            "class": MediarV,
            "params": MediarVParams,
        },
    }
    
    @classmethod
    def __get_entry(cls, name: str) -> dict[str, Type[Any]]:
        """
        Private method to retrieve the model entry from the registry using case-insensitive lookup.
        
        Args:
            name (str): The name of the model.
        
        Returns:
            dict(str, Type[Any]): A dictionary containing the keys 'class' and 'params'.
        
        Raises:
            ValueError: If the model is not found.
        """
        name_lower = name.lower()
        mapping = {key.lower(): key for key in cls.__MODELS}
        original_key = mapping.get(name_lower)
        if original_key is None:
            raise ValueError(
                f"Model '{name}' not found! Available options: {list(cls.__MODELS.keys())}"
            )
        return cls.__MODELS[original_key]
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[nn.Module]:
        """
        Retrieves the model class by name (case-insensitive).
        
        Args:
            name (str): Name of the model.
        
        Returns:
            Type(torch.nn.Module): The model class.
        """
        entry = cls.__get_entry(name)
        return entry["class"]
    
    @classmethod
    def get_model_params(cls, name: str) -> Type[BaseModel]:
        """
        Retrieves the model parameter class by name (case-insensitive).
        
        Args:
            name (str): Name of the model.
        
        Returns:
            Type(BaseModel): The model parameter class.
        """
        entry = cls.__get_entry(name)
        return entry["params"]
    
    @classmethod
    def get_available_models(cls) -> tuple[str, ...]:
        """
        Returns a tuple of available model names in their original case.
        
        Returns:
            Tuple(str): Tuple of available model names.
        """
        return tuple(cls.__MODELS.keys())
