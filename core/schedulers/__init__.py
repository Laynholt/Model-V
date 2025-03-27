import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Final, Tuple, Type, List, Any, Union
from pydantic import BaseModel

from .step import StepLRParams
from .multi_step import MultiStepLRParams
from .exponential import ExponentialLRParams
from .cosine_annealing import CosineAnnealingLRParams

__all__ = [
    "SchedulerRegistry",
    "StepLRParams", "MultiStepLRParams", "ExponentialLRParams", "CosineAnnealingLRParams"
]

class SchedulerRegistry:
    """Registry for learning rate schedulers and their parameter classes with case-insensitive lookup."""
    
    __SCHEDULERS: Final[Dict[str, Dict[str, Type[Any]]]] = {
        "Step": {
            "class": lr_scheduler.StepLR,
            "params": StepLRParams,
        },
        "Exponential": {
            "class": lr_scheduler.ExponentialLR,
            "params": ExponentialLRParams,
        },
        "MultiStep": {
            "class": lr_scheduler.MultiStepLR,
            "params": MultiStepLRParams,
        },
        "CosineAnnealing": {
            "class": lr_scheduler.CosineAnnealingLR,
            "params": CosineAnnealingLRParams,
        },
    }
    
    @classmethod
    def __get_entry(cls, name: str) -> Dict[str, Type[Any]]:
        """
        Private method to retrieve the scheduler entry from the registry using case-insensitive lookup.
        
        Args:
            name (str): The name of the scheduler.
        
        Returns:
            Dict[str, Type[Any]]: A dictionary containing the keys 'class' and 'params'.
        
        Raises:
            ValueError: If the scheduler is not found.
        """
        name_lower = name.lower()
        mapping = {key.lower(): key for key in cls.__SCHEDULERS}
        original_key = mapping.get(name_lower)
        if original_key is None:
            raise ValueError(
                f"Scheduler '{name}' not found! Available options: {list(cls.__SCHEDULERS.keys())}"
            )
        return cls.__SCHEDULERS[original_key]
    
    @classmethod
    def get_scheduler_class(cls, name: str) -> Type[lr_scheduler.LRScheduler]:
        """
        Retrieves the scheduler class by name (case-insensitive).
        
        Args:
            name (str): Name of the scheduler.
        
        Returns:
            Type[lr_scheduler.LRScheduler]: The scheduler class.
        """
        entry = cls.__get_entry(name)
        return entry["class"]
    
    @classmethod
    def get_scheduler_params(cls, name: str) -> Union[Type[BaseModel], Tuple[Type[BaseModel]]]:
        """
        Retrieves the scheduler parameter class by name (case-insensitive).
        
        Args:
            name (str): Name of the scheduler.
        
        Returns:
            Union[Type[BaseModel], Tuple[Type[BaseModel]]]: The scheduler parameter class or a tuple of parameter classes.
        """
        entry = cls.__get_entry(name)
        return entry["params"]
    
    @classmethod
    def get_available_schedulers(cls) -> Tuple[str, ...]:
        """
        Returns a tuple of available scheduler names in their original case.
        
        Returns:
            Tuple[str]: Tuple of available scheduler names.
        """
        return tuple(cls.__SCHEDULERS.keys())
