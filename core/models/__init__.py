import torch.nn as nn
from typing import Dict, Final, Tuple, Type, Any, List, Union
from pydantic import BaseModel

from .model_v import ModelV, ModelVParams


__all__ = [
    "ModelRegistry",
    "ModelV",
    "ModelVParams"
]


class ModelRegistry:
    """Registry for models and their parameter classes with case-insensitive lookup."""
    
    # Single dictionary storing both model classes and parameter classes.
    __MODELS: Final[Dict[str, Dict[str, Type[Any]]]] = {
        "ModelV": {
            "class": ModelV,
            "params": ModelVParams,
        },
    }
    
    @classmethod
    def __get_entry(cls, name: str) -> Dict[str, Type[Any]]:
        """
        Private method to retrieve the model entry from the registry using case-insensitive lookup.
        
        Args:
            name (str): The name of the model.
        
        Returns:
            Dict[str, Type[Any]]: A dictionary containing the keys 'class' and 'params'.
        
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
            Type[nn.Module]: The model class.
        """
        entry = cls.__get_entry(name)
        return entry["class"]
    
    @classmethod
    def get_model_params(cls, name: str) -> Union[Type[BaseModel], Tuple[Type[BaseModel]]]:
        """
        Retrieves the model parameter class by name (case-insensitive).
        
        Args:
            name (str): Name of the model.
        
        Returns:
            Union[Type[BaseModel], Tuple[Type[BaseModel]]]: The model parameter class or a tuple of parameter classes.
        """
        entry = cls.__get_entry(name)
        return entry["params"]
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Returns a list of available model names in their original case.
        
        Returns:
            List[str]: List of available model names.
        """
        return list(cls.__MODELS.keys())
