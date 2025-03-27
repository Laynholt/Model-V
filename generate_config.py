import os
from pydantic import BaseModel
from typing import Any, Dict, Tuple, Type, Union, List

from config.config import Config
from config.dataset_config import DatasetConfig

from core import (
    ModelRegistry, CriterionRegistry, OptimizerRegistry, SchedulerRegistry
)


def instantiate_params(param: Any) -> Union[BaseModel, List[BaseModel]]:
    """
    Instantiates the parameter class(es) with default values.
    
    If 'param' is a tuple, instantiate each class and return a list of instances.
    Otherwise, instantiate the single class and return the instance.
    """
    if isinstance(param, tuple):
        return [cls() for cls in param]
    else:
        return param()

def prompt_choice(prompt_message: str, options: Tuple[str, ...]) -> str:
    """
    Prompt the user with a list of options and return the selected option.
    """
    print(prompt_message)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # Determine the directory of this script.
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Ask the user whether this is training mode.
    training_input = input("Is this training mode? (y/n): ").strip().lower()
    is_training = training_input in ("y", "yes")

    # Create a default DatasetConfig based on the training mode.
    # The DatasetConfig.default_config method fills in required fields with zero-values.
    dataset_config = DatasetConfig(is_training=is_training)

    # Prompt the user to select a model.
    model_options = ModelRegistry.get_available_models()
    chosen_model = prompt_choice("\nSelect a model:", model_options)
    model_param_class = ModelRegistry.get_model_params(chosen_model)
    model_instance = instantiate_params(model_param_class)

    if is_training is False:
        config = Config(
            model={chosen_model: model_instance},
            dataset_config=dataset_config
        )
        
        # Construct a base filename from the selected registry names.
        base_filename = f"{chosen_model}"

    else:
        # Prompt the user to select a criterion.
        criterion_options = CriterionRegistry.get_available_criterions()
        chosen_criterion = prompt_choice("\nSelect a criterion:", criterion_options)
        criterion_param_class = CriterionRegistry.get_criterion_params(chosen_criterion)
        criterion_instance = instantiate_params(criterion_param_class)

        # Prompt the user to select an optimizer.
        optimizer_options = OptimizerRegistry.get_available_optimizers()
        chosen_optimizer = prompt_choice("\nSelect an optimizer:", optimizer_options)
        optimizer_param_class = OptimizerRegistry.get_optimizer_params(chosen_optimizer)
        optimizer_instance = instantiate_params(optimizer_param_class)

        # Prompt the user to select a scheduler.
        scheduler_options = SchedulerRegistry.get_available_schedulers()
        chosen_scheduler = prompt_choice("\nSelect a scheduler:", scheduler_options)
        scheduler_param_class = SchedulerRegistry.get_scheduler_params(chosen_scheduler)
        scheduler_instance = instantiate_params(scheduler_param_class)

        # Assemble the overall configuration using the registry names as keys.
        config = Config(
            model={chosen_model: model_instance},
            dataset_config=dataset_config,
            criterion={chosen_criterion: criterion_instance},
            optimizer={chosen_optimizer: optimizer_instance},
            scheduler={chosen_scheduler: scheduler_instance}
        )
        
        # Construct a base filename from the selected registry names.
        base_filename = f"{chosen_model}_{chosen_criterion}_{chosen_optimizer}_{chosen_scheduler}"

    # Determine the output directory relative to this script.
    base_dir = os.path.join(script_path, "config/jsons", "train" if is_training else "predict")
    os.makedirs(base_dir, exist_ok=True)

    filename = f"{base_filename}.json"
    full_path = os.path.join(base_dir, filename)
    counter = 1

    # Append a counter if a file with the same name exists.
    while os.path.exists(full_path):
        filename = f"{base_filename}_{counter}.json"
        full_path = os.path.join(base_dir, filename)
        counter += 1

    # Save the configuration as a JSON file.
    config.save_json(full_path)

    print(f"\nConfiguration saved to: {full_path}")

if __name__ == "__main__":
    main()
