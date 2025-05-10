import os
import argparse
import wandb

from config import Config
from core.data import *
from core.segmentator import CellSegmentator


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict cell segmentator with specified config file."
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing.json',
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

    args = parser.parse_args()

    mode = args.mode
    config_path = args.config
    config = Config.load_json(config_path)

    if mode == 'train' and not config.dataset_config.is_training:
        raise ValueError(
            f"Config is not set for training (is_training=False), but mode 'train' was requested."
        )
    if mode in ('test', 'predict') and config.dataset_config.is_training:
        raise ValueError(
            f"Config is set for training (is_training=True), but mode '{mode}' was requested."
        )

    if config.wandb_config.use_wandb:
        # Initialize W&B
        wandb.init(config=config.asdict(), **config.wandb_config.asdict())
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
        segmentator.run(save_results=args.save_masks, only_masks=args.only_masks)
    except Exception:
        raise
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



if __name__ == "__main__":
    main()
