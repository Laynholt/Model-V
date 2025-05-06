import os
import wandb

from config.config import Config
from core.data import *
from core.segmentator import CellSegmentator


if __name__ == "__main__":
    config_path = 'config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing.json'
    # config_path = 'config/templates/predict/ModelV.json'
    
    config = Config.load_json(config_path)
    # config = Config.load_json(config_path)
    
    if config.dataset_config.wandb.use_wandb:
        # Initialize W&B
        wandb.init(config=config.asdict(), **config.dataset_config.wandb.asdict())

        # How many batches to wait before logging training status
        wandb.config.log_interval = 10
    
    segmentator = CellSegmentator(config)
    segmentator.create_dataloaders()
    
    # Watch parameters & gradients of model
    if config.dataset_config.wandb.use_wandb:
        wandb.watch(segmentator._model, log="all", log_graph=True)
    
    segmentator.run()
    
    weights_dir = "weights" if not config.dataset_config.wandb.use_wandb else wandb.run.dir # type: ignore
    saving_path = os.path.join(
        weights_dir, os.path.basename(config.dataset_config.common.predictions_dir) + '.pth'
    )
    segmentator.save_checkpoint(saving_path)
    
    if config.dataset_config.wandb.use_wandb:
        wandb.save(saving_path)
    
    
