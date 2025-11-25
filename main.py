import os
import sys
import argparse
import wandb

from config import Config
from core.data import (
    get_train_transforms,
    get_valid_transforms,
    get_test_transforms,
    get_predict_transforms
)
from core.segmentator import CellSegmentator


def main(
    manual: bool = False,
    config_path: str | None = None,
    mode: str | None = None,
    save_masks: bool = True,
    only_masks: bool = False
) -> None:
    
    if not manual:
        parser = argparse.ArgumentParser(
            description="Train or predict cell segmentator with specified config file."
        )
        parser.add_argument(
            '-c', '--config',
            type=str,
            help='Path to the JSON config file'
        )
        parser.add_argument(
            '-m', '--mode',
            choices=['train', 'test', 'predict'],
            default='train',
            help='Run mode: train, test or predict'
        )
        parser.add_argument(
            '--no-save-masks',
            action='store_false',
            dest='save_masks',
            help='If set, do NOT save predicted masks (saving is enabled by default)'
        )
        parser.add_argument(
            '--only-masks',
            action='store_true',
            help=('If set and save-masks set, save only the raw predicted'
                ' masks without additional visualizations')
        )

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

        mode = args.mode
        config_path = args.config
        save_masks = args.save_masks
        only_masks = args.only_masks
    
    else:
        if mode is None or config_path is None:
            raise ValueError("In manual mode, you must specify the path to the config and mode!")
        
    config = Config.load_json(config_path)

    if mode == 'train' and not config.dataset_config.is_training:
        raise ValueError(
            "Config is not set for training (is_training=False), but mode 'train' was requested."
        )
    if mode in ('test', 'predict') and config.dataset_config.is_training:
        raise ValueError(
            f"Config is set for training (is_training=True), but mode '{mode}' was requested."
        )

    if config.wandb_config.use_wandb:
        # Initialize W&B
        wandb.init(config=config.asdict(), reinit="finish_previous", **config.wandb_config.asdict())
        # How many batches to wait before logging training status
        wandb.config.log_interval = 10

    segmentator = CellSegmentator(config)
    segmentator.create_dataloaders(
        train_transforms=get_train_transforms(
            roi_size=config.dataset_config.common.roi_size) if mode == "train" else None,
        valid_transforms=get_valid_transforms() if mode == "train" else None,
        test_transforms=get_test_transforms() if mode in ("train", "test") else None,
        predict_transforms=get_predict_transforms() if mode == "predict" else None
    )
    
    segmentator.print_data_info(
        loader_type=mode, index=0
    )

    # Watch parameters & gradients of model
    if config.wandb_config.use_wandb:
        wandb.watch(segmentator._model, log="all", log_graph=True)

    try:
        segmentator.run(save_results=save_masks, only_masks=only_masks)
    except Exception as e:
        raise e
    finally:
        if config.dataset_config.is_training:
            # Prepare saving path
            weights_dir = (
                wandb.run.dir if config.wandb_config.use_wandb else "weights" # type: ignore
            )
            saving_path = os.path.join(
                weights_dir,
                os.path.basename(config.dataset_config.common.predictions_dir) + '.pth'
            )
            segmentator.save_checkpoint(saving_path)

            if config.wandb_config.use_wandb:
                wandb.save(saving_path)
                wandb.finish()



if __name__ == "__main__":
    train_configs = [
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cA.json",
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cB.json",
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cSoma.json",
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cAB.json",
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cABSoma.json",
        
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cytoCell.json",
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cytoNuc.json",
        "/workspace/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing_cytoCellNuc.json",
    ]
    
    predict_configs = [
        "/workspace/model-v/config/templates/predict/ModelV_cA.json",
        "/workspace/model-v/config/templates/predict/ModelV_cB.json",
        "/workspace/model-v/config/templates/predict/ModelV_cSoma.json",
        "/workspace/model-v/config/templates/predict/ModelV_cAB.json",
        "/workspace/model-v/config/templates/predict/ModelV_cABSoma.json",
        
        "/workspace/model-v/config/templates/predict/ModelV_cytoCell.json",
        "/workspace/model-v/config/templates/predict/ModelV_cytoNuc.json",
        "/workspace/model-v/config/templates/predict/ModelV_cytoCellNuc.json",
    ]
        
    for config in train_configs:
        print(f"Hande config {config}")
        main(
            manual=True,
            config_path=config,
            mode="train",
            save_masks=True,
            only_masks=False
        )
        
    for config in predict_configs:
        print(f"Hande config {config}")
        main(
            manual=True,
            config_path=config,
            mode="predict",
            save_masks=True,
            only_masks=False
        )
    