import json
from typing import Any, Dict, Optional
from pydantic import BaseModel

from .dataset_config import DatasetConfig


__all__ = ["Config", "ComponentConfig"]


class ComponentConfig(BaseModel):
    name: str
    params: BaseModel

    def dump(self) -> Dict[str, Any]:
        """
        Recursively serializes the component into a dictionary.

        Returns:
            dict: A dictionary containing the component name and its serialized parameters.
        """
        if isinstance(self.params, BaseModel):
            params_dump = self.params.model_dump()
        else:
            params_dump = self.params
        return {
            "name": self.name,
            "params": params_dump
        }
        


class Config(BaseModel):
    model: ComponentConfig
    dataset_config: DatasetConfig
    criterion: Optional[ComponentConfig] = None
    optimizer: Optional[ComponentConfig] = None
    scheduler: Optional[ComponentConfig] = None

    def save_json(self, file_path: str, indent: int = 4) -> None:
        """
        Saves the configuration to a JSON file using dumps of each individual field.
        
        Args:
            file_path (str): Destination path for the JSON file.
            indent (int): Indentation level for the JSON file.
        """
        config_dump = {
            "model": self.model.dump(),
            "dataset_config": self.dataset_config.model_dump()
        }
        if self.criterion is not None:
            config_dump["criterion"] = self.criterion.dump()
        if self.optimizer is not None:
            config_dump["optimizer"] = self.optimizer.dump()
        if self.scheduler is not None:
            config_dump["scheduler"] = self.scheduler.dump()
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(config_dump, indent=indent))


    @classmethod
    def load_json(cls, file_path: str) -> "Config":
        """
        Loads a configuration from a JSON file and re-instantiates each section using
        the registry keys to recover the original parameter class(es).

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Config: An instance of Config with the proper parameter classes.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse dataset_config using its Pydantic model.
        dataset_config = DatasetConfig(**data.get("dataset_config", {}))

        # Helper function to parse registry fields.
        def parse_field(component_data: Dict[str, Any], registry_getter) -> Optional[ComponentConfig]:
            name = component_data.get("name")
            params_data = component_data.get("params", {})
            
            if name is not None:
                expected = registry_getter(name)
                params = expected(**params_data)
                return ComponentConfig(name=name, params=params)
            return None

        from core import (
            ModelRegistry, CriterionRegistry, OptimizerRegistry, SchedulerRegistry
        )

        parsed_model = parse_field(data.get("model", {}), lambda key: ModelRegistry.get_model_params(key))
        parsed_criterion = parse_field(data.get("criterion", {}), lambda key: CriterionRegistry.get_criterion_params(key))
        parsed_optimizer = parse_field(data.get("optimizer", {}), lambda key: OptimizerRegistry.get_optimizer_params(key))
        parsed_scheduler = parse_field(data.get("scheduler", {}), lambda key: SchedulerRegistry.get_scheduler_params(key))

        if parsed_model is None:
            raise ValueError('Failed to load model information')

        return cls(
            model=parsed_model,
            dataset_config=dataset_config,
            criterion=parsed_criterion,
            optimizer=parsed_optimizer,
            scheduler=parsed_scheduler
        )
