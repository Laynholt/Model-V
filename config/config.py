import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from .dataset_config import DatasetConfig

class Config(BaseModel):
    model: Dict[str, Union[BaseModel, List[BaseModel]]]
    dataset_config: DatasetConfig
    criterion: Optional[Dict[str, Union[BaseModel, List[BaseModel]]]] = None
    optimizer: Optional[Dict[str, Union[BaseModel, List[BaseModel]]]] = None
    scheduler: Optional[Dict[str, Union[BaseModel, List[BaseModel]]]] = None

    @staticmethod
    def __dump_field(value: Any) -> Any:
        """
        Recursively dumps a field if it is a BaseModel or a list/dict of BaseModels.
        """
        if isinstance(value, BaseModel):
            return value.model_dump()
        elif isinstance(value, list):
            return [Config.__dump_field(item) for item in value]
        elif isinstance(value, dict):
            return {k: Config.__dump_field(v) for k, v in value.items()}
        else:
            return value

    def save_json(self, file_path: str, indent: int = 4) -> None:
        """
        Saves the configuration to a JSON file using dumps of each individual field.
        
        Args:
            file_path (str): Destination path for the JSON file.
            indent (int): Indentation level for the JSON file.
        """
        config_dump = {
            "model": self.__dump_field(self.model),
            "dataset_config": self.dataset_config.model_dump()
        }
        if self.criterion is not None:
            config_dump.update({"criterion": self.__dump_field(self.criterion)})
        if self.optimizer is not None:
            config_dump.update({"optimizer": self.__dump_field(self.optimizer)})
        if self.scheduler is not None:
            config_dump.update({"scheduler": self.__dump_field(self.scheduler)})
        
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
        def parse_field(field_data: Dict[str, Any], registry_getter) -> Dict[str, Union[BaseModel, List[BaseModel]]]:
            result = {}
            for key, value in field_data.items():
                expected = registry_getter(key)
                # If the registry returns a tuple, then we expect a list of dictionaries.
                if isinstance(expected, tuple):
                    result[key] = [cls_param(**item) for cls_param, item in zip(expected, value)]
                else:
                    result[key] = expected(**value)
            return result

        from core import (
            ModelRegistry, CriterionRegistry, OptimizerRegistry, SchedulerRegistry
        )

        parsed_model = parse_field(data.get("model", {}), lambda key: ModelRegistry.get_model_params(key))
        parsed_criterion = parse_field(data.get("criterion", {}), lambda key: CriterionRegistry.get_criterion_params(key))
        parsed_optimizer = parse_field(data.get("optimizer", {}), lambda key: OptimizerRegistry.get_optimizer_params(key))
        parsed_scheduler = parse_field(data.get("scheduler", {}), lambda key: SchedulerRegistry.get_scheduler_params(key))

        return cls(
            model=parsed_model,
            dataset_config=dataset_config,
            criterion=parsed_criterion,
            optimizer=parsed_optimizer,
            scheduler=parsed_scheduler
        )
