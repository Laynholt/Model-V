import random
import numpy as np
from numba import njit, prange

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

import fastremap

import fill_voids
from skimage import morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import mean, find_objects

from monai.data.dataset import Dataset
from monai.transforms import * # type: ignore
from monai.inferers.utils import sliding_window_inference
from monai.metrics.cumulative_average import CumulativeAverage

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import glob
import copy
import tifffile as tiff

from pprint import pformat
from tabulate import tabulate
from typing import Any, Dict, Literal, Optional, Tuple, List, Union

from tqdm import tqdm
import wandb

from config import Config
from core.models import *
from core.losses import *
from core.optimizers import *
from core.schedulers import *
from core.utils import (
    compute_batch_segmentation_tp_fp_fn,
    compute_f1_score,
    compute_average_precision_score
)

from core.logger import get_logger


logger = get_logger()


class CellSegmentator:
    def __init__(self, config: Config) -> None:
        self.__set_seed(config.dataset_config.common.seed)
        self.__parse_config(config)

        self._device: torch.device = torch.device(self._dataset_setup.common.device or "cpu")
        self._scaler = (
            torch.amp.GradScaler(self._device.type) # type: ignore
            if self._dataset_setup.is_training and self._dataset_setup.common.use_amp 
            else None
        )
        
        self._train_dataloader:     Optional[DataLoader] = None
        self._valid_dataloader:     Optional[DataLoader] = None
        self._test_dataloader:      Optional[DataLoader] = None
        self._predict_dataloader:   Optional[DataLoader] = None


    def create_dataloaders(
        self,
        train_transforms: Optional[Compose] = None,
        valid_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None
    ) -> None:
        """
        Creates train, validation, test, and prediction dataloaders based on dataset configuration
        and provided transforms.

        Args:
            train_transforms (Optional[Compose]): Transformations for training data.
            valid_transforms (Optional[Compose]): Transformations for validation data.
            test_transforms (Optional[Compose]): Transformations for testing data.
            predict_transforms (Optional[Compose]): Transformations for prediction data.

        Raises:
            ValueError: If required transforms are missing.
            RuntimeError: If critical dataset config values are missing.
        """
        if self._dataset_setup.is_training and train_transforms is None:
            raise ValueError("Training mode requires 'train_transforms' to be provided.")
        elif not self._dataset_setup.is_training and test_transforms is None and predict_transforms is None:
            raise ValueError("In inference mode, at least one of 'test_transforms' or 'predict_transforms' must be provided.")

        if self._dataset_setup.is_training:
            # Training mode: handle either pre-split datasets or splitting on the fly
            if self._dataset_setup.training.is_split:
                # Validate presence of validation transforms if validation directory and size are set
                if (
                    self._dataset_setup.training.pre_split.valid_dir and 
                    self._dataset_setup.training.valid_size and 
                    valid_transforms is None
                ):
                    raise ValueError("Validation transforms must be provided when using pre-split validation data.")
                
                # Use explicitly split directories
                train_dir = self._dataset_setup.training.pre_split.train_dir
                valid_dir = self._dataset_setup.training.pre_split.valid_dir
                test_dir = self._dataset_setup.training.pre_split.test_dir

                train_offset = self._dataset_setup.training.train_offset
                valid_offset = self._dataset_setup.training.valid_offset
                test_offset = self._dataset_setup.training.test_offset

                shuffle = False
            else:
                # Same validation for split mode with full data directory
                if (
                    self._dataset_setup.training.split.all_data_dir and 
                    self._dataset_setup.training.valid_size and 
                    valid_transforms is None
                ):
                    raise ValueError("Validation transforms must be provided when splitting dataset.")
                
                # Automatically split dataset from one directory
                train_dir = valid_dir = test_dir = self._dataset_setup.training.split.all_data_dir

                number_of_images = len(os.listdir(os.path.join(train_dir, 'images')))
                if number_of_images == 0:
                    raise FileNotFoundError(f"No images found in '{train_dir}/images'")

                # Calculate train/valid sizes
                train_size = (
                    self._dataset_setup.training.train_size
                    if isinstance(self._dataset_setup.training.train_size, int)
                    else int(number_of_images * self._dataset_setup.training.train_size)
                )
                valid_size = (
                    self._dataset_setup.training.valid_size
                    if isinstance(self._dataset_setup.training.valid_size, int)
                    else int(number_of_images * self._dataset_setup.training.valid_size)
                )

                train_offset = self._dataset_setup.training.train_offset
                valid_offset = self._dataset_setup.training.valid_offset + train_size
                test_offset = self._dataset_setup.training.test_offset + train_size + valid_size

                shuffle = True

            # Train dataloader
            train_dataset = self.__get_dataset(
                images_dir=os.path.join(train_dir, 'images'),
                masks_dir=os.path.join(train_dir, 'masks'),
                transforms=train_transforms,  # type: ignore
                size=self._dataset_setup.training.train_size,
                offset=train_offset,
                shuffle=shuffle
            )
            self._train_dataloader = DataLoader(train_dataset, batch_size=self._dataset_setup.training.batch_size, shuffle=True)
            logger.info(f"Loaded training dataset with {len(train_dataset)} samples.")

            # Validation dataloader
            if valid_transforms is not None:
                if not valid_dir or not self._dataset_setup.training.valid_size:
                    raise RuntimeError("Validation directory or size is not properly configured.")
                valid_dataset = self.__get_dataset(
                    images_dir=os.path.join(valid_dir, 'images'),
                    masks_dir=os.path.join(valid_dir, 'masks'),
                    transforms=valid_transforms,
                    size=self._dataset_setup.training.valid_size,
                    offset=valid_offset,
                    shuffle=shuffle
                )
                self._valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
                logger.info(f"Loaded validation dataset with {len(valid_dataset)} samples.")

            # Test dataloader
            if test_transforms is not None:
                if not test_dir or not self._dataset_setup.training.test_size:
                    raise RuntimeError("Test directory or size is not properly configured.")
                test_dataset = self.__get_dataset(
                    images_dir=os.path.join(test_dir, 'images'),
                    masks_dir=os.path.join(test_dir, 'masks'),
                    transforms=test_transforms,
                    size=self._dataset_setup.training.test_size,
                    offset=test_offset,
                    shuffle=shuffle
                )
                self._test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                logger.info(f"Loaded test dataset with {len(test_dataset)} samples.")

            # Prediction dataloader
            if predict_transforms is not None:
                if not test_dir or not self._dataset_setup.training.test_size:
                    raise RuntimeError("Prediction directory or size is not properly configured.")
                predict_dataset = self.__get_dataset(
                    images_dir=os.path.join(test_dir, 'images'),
                    masks_dir=None,
                    transforms=predict_transforms,
                    size=self._dataset_setup.training.test_size,
                    offset=test_offset,
                    shuffle=shuffle
                )
                self._predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
                logger.info(f"Loaded prediction dataset with {len(predict_dataset)} samples.")

        else:
            # Inference mode (no training)
            test_images = os.path.join(self._dataset_setup.testing.test_dir, 'images')
            test_masks = os.path.join(self._dataset_setup.testing.test_dir, 'masks')

            if test_transforms is not None:
                test_dataset = self.__get_dataset(
                    images_dir=test_images,
                    masks_dir=test_masks,
                    transforms=test_transforms,
                    size=self._dataset_setup.testing.test_size,
                    offset=self._dataset_setup.testing.test_offset,
                    shuffle=self._dataset_setup.testing.shuffle
                )
                self._test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                logger.info(f"Loaded test dataset with {len(test_dataset)} samples.")

            if predict_transforms is not None:
                predict_dataset = self.__get_dataset(
                    images_dir=test_images,
                    masks_dir=None,
                    transforms=predict_transforms,
                    size=self._dataset_setup.testing.test_size,
                    offset=self._dataset_setup.testing.test_offset,
                    shuffle=self._dataset_setup.testing.shuffle
                )
                self._predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
                logger.info(f"Loaded prediction dataset with {len(predict_dataset)} samples.")


    def print_data_info(
        self,
        loader_type: Literal["train", "valid", "test", "predict"],
        index: Optional[int] = None
    ) -> None:
        """
        Prints statistics for a single sample from the specified dataloader.

        Args:
            loader_type: One of "train", "valid", "test", "predict".
            index: The sample index; if None, a random index is selected.
        """
        # Retrieve the dataloader attribute, e.g., self._train_dataloader
        loader: Optional[torch.utils.data.DataLoader] = getattr(self, f"_{loader_type}_dataloader", None)
        if loader is None:
            logger.error(f"Dataloader '{loader_type}' is not initialized.")
            return

        dataset = loader.dataset
        total = len(dataset) # type: ignore
        if total == 0:
            logger.error(f"Dataset for '{loader_type}' is empty.")
            return

        # Choose index
        idx = index if index is not None else random.randint(0, total - 1)
        if not (0 <= idx < total):
            logger.error(f"Index {idx} is out of range [0, {total}).")
            return

        # Fetch the sample and apply transforms
        sample = dataset[idx]
        # Expecting a dict with {'image': ..., 'mask': ...} or {'image': ...}
        img = sample["image"]
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.asarray(img)

        # Compute image statistics
        img_min, img_max = img.min(), img.max()
        img_mean, img_std = float(img.mean()), float(img.std())
        img_shape = img.shape

        # Prepare log lines
        lines = [
            "=" * 40,
            f"Dataloader: '{loader_type}', sample index: {idx} / {total - 1}",
            f"Image  — shape: {img_shape}, min: {img_min:.4f}, max: {img_max:.4f}, mean: {img_mean:.4f}, std: {img_std:.4f}"
        ]

        # For 'predict', no mask is available
        if loader_type != "predict":
            mask = sample.get("mask", None)
            if mask is not None:
                # Convert tensor to numpy if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = np.asarray(mask)
                m_min, m_max = mask.min(), mask.max()
                m_mean, m_std = float(mask.mean()), float(mask.std())
                m_shape = mask.shape
                lines.append(
                    f"Mask   — shape: {m_shape}, min: {m_min:.4f}, "
                    f"max: {m_max:.4f}, mean: {m_mean:.4f}, std: {m_std:.4f}"
                )
            else:
                lines.append("Mask   — not available for this sample.")

        lines.append("=" * 40)

        # Output via logger
        for l in lines:
            logger.info(l)

    
    def train(self) -> None:
        """
        Train the model over multiple epochs, including validation and test.
        """
        # Ensure training is enabled in dataset setup
        if not self._dataset_setup.is_training:
            raise RuntimeError("Dataset setup indicates training is disabled.")
        
        # Determine device name for logging
        if self._device.type == "cpu":
            device_name = "cpu"
        else:
            idx = self._device.index if hasattr(self._device, 'index') else torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(idx)

        logger.info(f"\n{'=' * 50}\n")
        logger.info(f"Training starts on device: {device_name}")
        logger.info(f"\n{'=' * 50}")

        best_f1_score = 0.0
        best_weights = None

        for epoch in range(1, self._dataset_setup.training.num_epochs + 1):
            train_metrics = self.__run_epoch("train", epoch)
            self.__print_with_logging(train_metrics, epoch)

            # Step the scheduler after training
            if self._scheduler is not None:
                self._scheduler.step()

            # Periodic validation or tuning
            if epoch % self._dataset_setup.training.val_freq == 0:
                if self._valid_dataloader is not None:
                    valid_metrics = self.__run_epoch("valid", epoch)
                    self.__print_with_logging(valid_metrics, epoch)

                    # Update best model on improved F1
                    f1 = valid_metrics.get("valid_f1_score", 0.0)
                    if f1 > best_f1_score:
                        best_f1_score = f1
                        # Deep copy weights to avoid reference issues
                        best_weights = copy.deepcopy(self._model.state_dict())
                        logger.info(f"Updated best model weights with F1 score: {f1:.4f}")
            
        # Restore best model weights if available
        if best_weights is not None:
            self._model.load_state_dict(best_weights)

        if self._test_dataloader is not None:
            test_metrics = self.__run_epoch("test")
            self.__print_with_logging(test_metrics, 0)

  
    def evaluate(self) -> None:
        """
        Run a full test epoch and display/log the resulting metrics.
        """
        test_metrics = self.__run_epoch("test")
        self.__print_with_logging(test_metrics, 0)

 
    def predict(self) -> None:
        """
        Run inference on the predict set and save the resulting instance masks.
        """
        # Ensure the predict DataLoader has been set
        if self._predict_dataloader is None:
            raise RuntimeError("DataLoader for mode 'predict' is not set.")

        batch_counter = 0
        for batch in tqdm(self._predict_dataloader, desc="Predicting"):
            # Move input images to the configured device (CPU/GPU)
            inputs = batch["img"].to(self._device)

            # Use automatic mixed precision if enabled in dataset setup
            with torch.amp.autocast( # type: ignore
                self._device.type,
                enabled=self._dataset_setup.common.use_amp
            ):
                # Disable gradient computation for inference
                with torch.no_grad():
                    # Run the model’s forward pass in ‘predict’ mode
                    raw_output = self.__run_inference(inputs, mode="predict")

                    # Convert logits/probabilities to discrete instance masks
                    # ground_truth is not passed here; only predictions are needed
                    preds, _ = self.__post_process_predictions(raw_output)

                    # Save out the predicted masks, using batch_counter to index files
                    self.__save_prediction_masks(batch, preds, batch_counter)

            # Increment counter by batch size for unique file naming
            batch_counter += inputs.shape[0]


    def run(self) -> None:
        """
        Orchestrate the full workflow:  
        - If training is enabled in the dataset setup, start training.  
        - Otherwise, if a test DataLoader is provided, run evaluation.  
        - Else if a prediction DataLoader is provided, run inference/prediction.  
        - If neither loader is available in non‐training mode, raise an error.
        """
        # 1) TRAINING PATH
        if self._dataset_setup.is_training:
            # Launch the full training loop (with validation, scheduler steps, etc.)
            self.train()
            return

        # 2) NON-TRAINING PATH (TEST or PREDICT)
        # Prefer test if available
        if self._test_dataloader is not None:
            # Run a single evaluation epoch on the test set and log metrics
            self.evaluate()
            return

        # If no test loader, fall back to prediction if available
        if self._predict_dataloader is not None:
            # Run inference on the predict set and save outputs
            self.predict()
            return

        # 3) ERROR: no appropriate loader found
        raise RuntimeError(
            "Neither test nor predict DataLoader is set for non‐training mode."
        )


    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads model weights from a specified checkpoint into the current model.

        Args:
            checkpoint_path (str): Path to the checkpoint file containing the model weights.
        """
        # Load the checkpoint onto the correct device (CPU or GPU)
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        # Load the state dict into the model, allowing for missing keys
        self._model.load_state_dict(checkpoint['state_dict'], strict=False)


    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Saves the current model weights to a checkpoint file.

        Args:
            checkpoint_path (str): Path where the checkpoint file will be saved.
        """
        # Create a checkpoint dictionary containing the model’s state_dict
        checkpoint = {
            'state_dict': self._model.state_dict()
        }
        # Write the checkpoint to disk
        torch.save(checkpoint, checkpoint_path)


    def __parse_config(self, config: Config) -> None:
        """
        Parses the given configuration object to initialize model, criterion,
        optimizer, scheduler, and dataset setup.

        Args:
            config (Config): Configuration object with model, optimizer,
                            scheduler, criterion, and dataset setup information.
        """
        model = config.model
        criterion = config.criterion
        optimizer = config.optimizer
        scheduler = config.scheduler
        
        logger.info("========== Parsed Configuration ==========")
        logger.info("Model Config:\n%s", pformat(model.dump(), indent=2))
        if criterion:
            logger.info("Criterion Config:\n%s", pformat(criterion.dump(), indent=2))
        if optimizer:
            logger.info("Optimizer Config:\n%s", pformat(optimizer.dump(), indent=2))
        if scheduler:
            logger.info("Scheduler Config:\n%s", pformat(scheduler.dump(), indent=2))
        logger.info("Dataset Config:\n%s", pformat(config.dataset_config.model_dump(), indent=2))
        logger.info("==========================================")

        # Initialize model using the model registry
        self._model = ModelRegistry.get_model_class(model.name)(model.params)
        
        # Loads model weights from a specified checkpoint
        if config.dataset_config.is_training:
            if config.dataset_config.training.pretrained_weights:
                self.load_from_checkpoint(config.dataset_config.training.pretrained_weights)
        
        # Initialize loss criterion if specified
        self._criterion = (
            CriterionRegistry.get_criterion_class(criterion.name)(params=criterion.params)
            if criterion is not None
            else None
        )
        
        if hasattr(self._criterion, "num_classes"):
            nc_model = self._model.num_classes
            nc_crit  = getattr(self._criterion, "num_classes")
            if nc_model != nc_crit:
                raise ValueError(
                    f"Number of classes mismatch: model.num_classes={nc_model} "
                    f"but criterion.num_classes={nc_crit}"
                )

        # Initialize optimizer if specified
        self._optimizer = (
            OptimizerRegistry.get_optimizer_class(optimizer.name)(
                model_params=self._model.parameters(),
                optim_params=optimizer.params
            )
            if optimizer is not None
            else None
        )

        # Initialize scheduler only if both scheduler and optimizer are defined
        self._scheduler = (
            SchedulerRegistry.get_scheduler_class(scheduler.name)(
                optimizer=self._optimizer.optim,
                params=scheduler.params
            )
            if scheduler is not None and self._optimizer is not None and self._optimizer.optim is not None
            else None
        )

        logger.info("========== Model Components Initialization ==========")
        logger.info("├─ Model:      " + (f"{model.name}" if self._model else "Not specified"))
        logger.info("├─ Criterion:  " + (f"{criterion.name}" if self._criterion else "Not specified")) # type: ignore
        logger.info("├─ Optimizer:  " + (f"{optimizer.name}" if self._optimizer else "Not specified")) # type: ignore
        logger.info("└─ Scheduler:  " + (f"{scheduler.name}" if self._scheduler else "Not specified")) # type: ignore
        logger.info("=====================================================")
        

        # Save dataset config
        self._dataset_setup = config.dataset_config
        common = config.dataset_config.common
        
        logger.info("========== Dataset Setup ==========")
        logger.info("[COMMON]")
        logger.info(f"├─ Seed: {common.seed}")
        logger.info(f"├─ Device: {common.device}")
        logger.info(f"├─ Use AMP: {'yes' if common.use_amp else 'no'}")
        logger.info(f"└─ Predictions output dir: {common.predictions_dir}")

        if config.dataset_config.is_training:
            training = config.dataset_config.training
            logger.info("[MODE] Training")
            logger.info(f"├─ Batch size: {training.batch_size}")
            logger.info(f"├─ Epochs: {training.num_epochs}")
            logger.info(f"├─ Validation frequency: {training.val_freq}")
            logger.info(f"├─ Pretrained weights: {training.pretrained_weights or 'None'}")

            if training.is_split:
                logger.info(f"├─ Using pre-split directories:")
                logger.info(f"│  ├─ Train dir: {training.pre_split.train_dir}")
                logger.info(f"│  ├─ Valid dir: {training.pre_split.valid_dir}")
                logger.info(f"│  └─ Test dir:  {training.pre_split.test_dir}")
            else:
                logger.info(f"├─ Using unified dataset with splits:")
                logger.info(f"│  ├─ All data dir: {training.split.all_data_dir}")
                logger.info(f"│  └─ Shuffle: {'yes' if training.split.shuffle else 'no'}")

            logger.info(f"└─ Dataset split:")
            logger.info(f"   ├─ Train size: {training.train_size}, offset: {training.train_offset}")
            logger.info(f"   ├─ Valid size: {training.valid_size}, offset: {training.valid_offset}")
            logger.info(f"   └─ Test size:  {training.test_size}, offset: {training.test_offset}")

        else:
            testing = config.dataset_config.testing
            logger.info("[MODE] Inference")
            logger.info(f"├─ Test dir: {testing.test_dir}")
            logger.info(f"├─ Test size: {testing.test_size} (offset: {testing.test_offset})")
            logger.info(f"├─ Shuffle: {'yes' if testing.shuffle else 'no'}")
            logger.info(f"├─ Use ensemble: {'yes' if testing.use_ensemble else 'no'}")
            logger.info(f"└─ Pretrained weights:")
            logger.info(f"   ├─ Single model: {testing.pretrained_weights}")
            logger.info(f"   ├─ Ensemble model 1: {testing.ensemble_pretrained_weights1}")
            logger.info(f"   └─ Ensemble model 2: {testing.ensemble_pretrained_weights2}")

        wandb_cfg = config.dataset_config.wandb
        if wandb_cfg.use_wandb:
            logger.info("[W&B]")
            logger.info(f"├─ Project: {wandb_cfg.project}")
            logger.info(f"├─ Entity:  {wandb_cfg.entity}")
            if wandb_cfg.name:
                logger.info(f"├─ Run name: {wandb_cfg.name}")
            if wandb_cfg.tags:
                logger.info(f"├─ Tags:     {', '.join(wandb_cfg.tags)}")
            if wandb_cfg.notes:
                logger.info(f"├─ Notes:    {wandb_cfg.notes}")
            logger.info(f"└─ Save code: {'yes' if wandb_cfg.save_code else 'no'}")
        else:
            logger.info("[W&B] Logging disabled")

        logger.info("===================================")


    def __set_seed(self, seed: Optional[int]) -> None:
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

        Args:
            seed (Optional[int]): Seed value. If None, no seeding is performed.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"Random seed set to {seed}")
        else:
            logger.info("Seed not set (None provided)")


    def __get_dataset(
        self,
        images_dir: str,
        masks_dir: Optional[str],
        transforms: Compose,
        size: Union[int, float],
        offset: int,
        shuffle: bool
    ) -> Dataset:
        """
        Loads and returns a dataset object from image and optional mask directories.

        Args:
            images_dir (str): Path to directory or glob pattern for input images.
            masks_dir (Optional[str]): Path to directory or glob pattern for masks.
            transforms (Compose): Transformations to apply to each image or pair.
            size (Union[int, float]): Either an integer or a fraction of the dataset.
            offset (int): Number of images to skip from the start.
            shuffle (bool): Whether to shuffle the dataset before slicing.

        Returns:
            Dataset: A dataset containing image and optional mask paths.

        Raises:
            FileNotFoundError: If no images are found.
            ValueError: If masks are provided but do not match image count.
            ValueError: If dataset is too small for requested size or offset.
        """
        # Collect sorted list of image paths
        images = sorted(glob.glob(images_dir))
        if not images:
            raise FileNotFoundError(f"No images found in path or pattern: '{images_dir}'")

        if masks_dir is not None:
            # Collect and validate sorted list of mask paths
            masks = sorted(glob.glob(masks_dir))
            if len(images) != len(masks):
                raise ValueError(f"Number of masks ({len(masks)}) does not match number of images ({len(images)})")

        # Convert float size (fraction) to absolute count
        size = size if isinstance(size, int) else int(size * len(images))

        if size <= 0:
            raise ValueError(f"Size must be positive, got: {size}")

        if len(images) < size:
            raise ValueError(f"Not enough images ({len(images)}) for requested size ({size})")

        if len(images) < size + offset:
            raise ValueError(f"Offset ({offset}) + size ({size}) exceeds dataset length ({len(images)})")

        # Shuffle image-mask pairs if requested
        if shuffle:
            if masks_dir is not None:
                combined = list(zip(images, masks))  # type: ignore
                random.shuffle(combined)
                images, masks = zip(*combined)
            else:
                random.shuffle(images)

        # Apply offset and limit by size
        images = images[offset: offset + size]
        if masks_dir is not None:
            masks = masks[offset: offset + size]  # type: ignore

        # Prepare data structure for Dataset class
        if masks_dir is not None:
            data = [
                {"image": image, "mask": mask}
                for image, mask in zip(images, masks)  # type: ignore
            ]
        else:
            data = [{"image": image} for image in images]

        return Dataset(data, transforms)
    
    
    def __print_with_logging(self, results: Dict[str, float], step: int) -> None:
        """
        Print metrics in a tabular format and log to W&B.

        Args:
            results (Dict[str, float]): results dictionary.
            step (int): epoch index.
        """
        table = tabulate(
            tabular_data=results.items(),
            headers=["Metric", "Value"],
            floatfmt=".4f",
            tablefmt="fancy_grid"
        )
        print(table, "\n")
        if self._dataset_setup.wandb.use_wandb:
            wandb.log(results, step=step)
    
    
    def __run_epoch(self,
        mode: Literal["train", "valid", "test"],
        epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Execute one epoch of training, validation, or testing.

        Args:
            mode (str): One of 'train', 'valid', or 'test'.
            epoch (int, optional): Current epoch number for logging.

        Returns:
            Dict[str, float]: Loss metrics and F1 score for valid/test.
        """
        # Ensure required components are available
        if mode in ("train", "valid") and (self._optimizer is None or self._criterion is None):
            raise RuntimeError("Optimizer and loss function must be initialized for train/valid.")

        # Set model mode and choose the appropriate data loader
        if mode == "train":
            self._model.train()
            loader = self._train_dataloader
        else:
            self._model.eval()
            loader = self._valid_dataloader if mode == "valid" else self._test_dataloader

        # Check that the data loader is provided
        if loader is None:
            raise RuntimeError(f"DataLoader for mode '{mode}' is not set.")

        all_tp, all_fp, all_fn = [], [], []

        # Prepare tqdm description with epoch info if available
        if epoch is not None:
            desc = f"Epoch {epoch}/{self._dataset_setup.training.num_epochs} [{mode}]"
        else:
            desc = f"Epoch ({mode})"

        # Iterate over batches
        batch_counter = 0
        for batch in tqdm(loader, desc=desc):
            inputs = batch["img"].to(self._device)
            targets = batch["label"].to(self._device)

            # Zero gradients for training
            if self._optimizer is not None:
                self._optimizer.zero_grad()

            # Mixed precision context if enabled
            with torch.amp.autocast(                            # type: ignore
                self._device.type, 
                enabled=self._dataset_setup.common.use_amp
            ):
                # Only compute gradients in training phase
                with torch.set_grad_enabled(mode == "train"):
                    # Forward pass
                    raw_output = self.__run_inference(inputs, mode)

                    if self._criterion is not None:
                        # Convert label masks to flow representations (one-hot)
                        flow_targets = self.__compute_flows_from_masks(targets)

                        # Compute loss for this batch
                        batch_loss = self._criterion(raw_output, flow_targets)  # type: ignore

                    # Post-process and compute F1 during validation and testing
                    if mode in ("valid", "test"):
                        preds, labels_post = self.__post_process_predictions(
                            raw_output, targets
                        )
                        
                        # Collecting statistics on the batch
                        tp, fp, fn = self.__compute_stats(
                            predicted_masks=preds,
                            ground_truth_masks=labels_post, # type: ignore
                            iou_threshold=0.5
                        )
                        all_tp.append(tp)
                        all_fp.append(fp)
                        all_fn.append(fn)
                        
                        if mode == "test":
                            self.__save_prediction_masks(
                                batch, preds, batch_counter
                            )

                # Backpropagation and optimizer step in training
                if mode == "train":
                    if self._dataset_setup.common.use_amp and self._scaler is not None:
                        self._scaler.scale(batch_loss).backward() # type: ignore
                        self._scaler.unscale_(self._optimizer.optim) # type: ignore
                        self._scaler.step(self._optimizer.optim) # type: ignore
                        self._scaler.update()
                    else:    
                        batch_loss.backward() # type: ignore
                        if self._optimizer is not None:
                            self._optimizer.step()
                        
            batch_counter += inputs.shape[0]

        if self._criterion is not None:
            # Collect loss metrics
            epoch_metrics = {f"{mode}_{name}": value for name, value in self._criterion.get_loss_metrics().items()}
            # Reset internal loss metrics accumulator
            self._criterion.reset_metrics()
        else:
            epoch_metrics = {}

        # Include F1 and mAP for validation and testing
        if mode in ("valid", "test"):
            # Concatenating by batch: shape (num_batches*B, C)
            tp_array = np.vstack(all_tp)
            fp_array = np.vstack(all_fp)
            fn_array = np.vstack(all_fn)
            
            epoch_metrics[f"{mode}_f1_score"] = self.__compute_f1_metric( # type: ignore
                tp_array, fp_array, fn_array, reduction="micro"
            )
            epoch_metrics[f"{mode}_mAP"] = self.__compute_average_precision_metric( # type: ignore
                tp_array, fp_array, fn_array, reduction="macro"
            )

        return epoch_metrics
    
    
    def __run_inference(
        self,
        inputs: torch.Tensor,
        mode: Literal["train", "valid", "test", "predict"] = "train"
    ) -> torch.Tensor:
        """
        Perform model inference for different stages.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).
            stage (Literal[...]): One of "train", "valid", "test", "predict".

        Returns:
            torch.Tensor: Model outputs tensor.
        """
        if mode != "train":
            # Use sliding window inference for non-training phases
            outputs = sliding_window_inference(
                inputs,
                roi_size=512,
                sw_batch_size=4,
                predictor=self._model,
                padding_mode="constant",
                mode="gaussian",
                overlap=0.5,
            )
        else:
            # Direct forward pass during training
            outputs = self._model(inputs)
        return outputs # type: ignore
    

    def __post_process_predictions(
        self,
        raw_outputs: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Post-process raw network outputs to extract instance segmentation masks.

        Args:
            raw_outputs (torch.Tensor): Raw model outputs of shape (B, С, H, W).
            ground_truth (torch.Tensor): Ground truth masks of shape (B, С, H, W).

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - instance_masks: Instance-wise masks array of shape (B, С, H, W).
                - labels_np: Converted ground truth of shape (B, С, H, W) or None if 
                ground_truth was not provided.
        """
        # Move outputs to CPU and convert to numpy
        outputs_np = raw_outputs.cpu().numpy()
        # Split channels: gradient flows then class logits
        gradflow = outputs_np[:, :2 * self._model.num_classes]
        logits = outputs_np[:, -self._model.num_classes :]
        # Apply sigmoid to logits to get probabilities
        probabilities = self.__sigmoid(logits)

        batch_size, _, height, width = probabilities.shape
        # Prepare container for instance masks
        instance_masks = np.zeros((batch_size, self._model.num_classes, height, width), dtype=np.uint16)
        for idx in range(batch_size):
            instance_masks[idx] = self.__segment_instances(
                probability_map=probabilities[idx],
                flow=gradflow[idx],
                prob_threshold=0.0,
                flow_threshold=0.4,
                min_object_size=15
            )

        # Convert ground truth to numpy
        labels_np = ground_truth.cpu().numpy() if ground_truth is not None else None
        return instance_masks, labels_np

    
    def __compute_stats(
        self,
        predicted_masks: np.ndarray,
        ground_truth_masks: np.ndarray,
        iou_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute batch-wise true positives, false positives, and false negatives
        for instance segmentation, using a configurable IoU threshold.

        Args:
            predicted_masks (np.ndarray): Predicted instance masks of shape (B, C, H, W).
            ground_truth_masks (np.ndarray): Ground truth instance masks of shape (B, C, H, W).
            iou_threshold (float): Intersection-over-Union threshold for matching predictions
                to ground truths (default: 0.5).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - tp: True positives per batch and class, shape (B, C)
                - fp: False positives per batch and class, shape (B, C)
                - fn: False negatives per batch and class, shape (B, C)
        """
        stats = compute_batch_segmentation_tp_fp_fn(
            batch_ground_truth=ground_truth_masks,
            batch_prediction=predicted_masks,
            iou_threshold=iou_threshold,
            remove_boundary_objects=True
        )
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        return tp, fp, fn
    
    
    def __compute_f1_metric(
        self,
        true_positives: np.ndarray,
        false_positives: np.ndarray,
        false_negatives: np.ndarray,
        reduction: Literal["micro", "macro", "weighted", "imagewise", "none"] = "micro"
    ) -> Union[float, np.ndarray]:
        """
        Compute F1-score from batch-wise TP/FP/FN using various aggregation schemes.

        Args:
            true_positives: array of TP counts per sample and class.
            false_positives: array of FP counts per sample and class.
            false_negatives: array of FN counts per sample and class.
            reduction:
                - 'none':    return F1 for each sample, class → shape (batch_size, num_classes)
                - 'micro':   global F1 over all samples & classes
                - 'imagewise':  F1 per sample (summing over classes), then average over samples
                - 'macro':   average class-wise F1 (classes summed over batch)
                - 'weighted': class-wise F1 weighted by support (TP+FN) 
        Returns:
            float for reductions 'micro', 'imagewise', 'macro', 'weighted'; 
            or np.ndarray of shape (batch_size, num_classes) if reduction='none'.
        """
        batch_size, num_classes = true_positives.shape

        # 1) No reduction: compute F1 for each (sample, class)
        if reduction == "none":
            f1_matrix = np.zeros((batch_size, num_classes), dtype=float)
            for i in range(batch_size):
                for c in range(num_classes):
                    tp_val = int(true_positives[i, c])
                    fp_val = int(false_positives[i, c])
                    fn_val = int(false_negatives[i, c])
                    _, _, f1_val = compute_f1_score(tp_val, fp_val, fn_val)
                    f1_matrix[i, c] = f1_val
            return f1_matrix

        # 2) Micro: sum all TP/FP/FN and compute a single F1
        if reduction == "micro":
            tp_total = int(true_positives.sum())
            fp_total = int(false_positives.sum())
            fn_total = int(false_negatives.sum())
            _, _, f1_global = compute_f1_score(tp_total, fp_total, fn_total)
            return f1_global

        # 3) Imagewise: compute per-sample F1 (sum over classes), then average
        if reduction == "imagewise":
            f1_per_image = np.zeros(batch_size, dtype=float)
            for i in range(batch_size):
                tp_i = int(true_positives[i].sum())
                fp_i = int(false_positives[i].sum())
                fn_i = int(false_negatives[i].sum())
                _, _, f1_i = compute_f1_score(tp_i, fp_i, fn_i)
                f1_per_image[i] = f1_i
            return float(f1_per_image.mean())

        # For macro/weighted, first aggregate per class across the batch
        tp_per_class = true_positives.sum(axis=0).astype(int)   # shape (num_classes,)
        fp_per_class = false_positives.sum(axis=0).astype(int)
        fn_per_class = false_negatives.sum(axis=0).astype(int)

        # 4) Macro: average F1 across classes equally
        if reduction == "macro":
            f1_per_class = np.zeros(num_classes, dtype=float)
            for c in range(num_classes):
                _, _, f1_c = compute_f1_score(
                    tp_per_class[c],
                    fp_per_class[c],
                    fn_per_class[c]
                )
                f1_per_class[c] = f1_c
            return float(f1_per_class.mean())

        # 5) Weighted: class-wise F1 weighted by support = TP + FN
        if reduction == "weighted":
            f1_per_class = np.zeros(num_classes, dtype=float)
            support = np.zeros(num_classes, dtype=float)
            for c in range(num_classes):
                tp_c = tp_per_class[c]
                fp_c = fp_per_class[c]
                fn_c = fn_per_class[c]
                _, _, f1_c = compute_f1_score(tp_c, fp_c, fn_c)
                f1_per_class[c] = f1_c
                support[c] = tp_c + fn_c
            total_support = support.sum()
            if total_support == 0:
                # fallback to unweighted macro if no positives
                return float(f1_per_class.mean())
            weights = support / total_support
            return float((f1_per_class * weights).sum())

        raise ValueError(f"Unknown reduction mode: {reduction}")
    
    
    def __compute_average_precision_metric(
        self,
        true_positives: np.ndarray,
        false_positives: np.ndarray,
        false_negatives: np.ndarray,
        reduction: Literal["micro", "macro", "weighted", "imagewise", "none"] = "micro"
    ) -> Union[float, np.ndarray]:
        """
        Compute Average Precision (AP) from batch-wise TP/FP/FN using various aggregation schemes.

        AP is defined here as:
            AP = TP / (TP + FP + FN)

        Args:
            true_positives: array of true positives per sample and class.
            false_positives: array of false positives per sample and class.
            false_negatives: array of false negatives per sample and class.
            reduction:
                - 'none':      return AP for each sample and class → shape (batch_size, num_classes)
                - 'micro':     global AP over all samples & classes
                - 'imagewise': AP per sample (summing stats over classes), then average over batch
                - 'macro':     average class-wise AP (each class summed over batch)
                - 'weighted':  class-wise AP weighted by support (TP+FN)

        Returns:
            float for reductions 'micro', 'imagewise', 'macro', 'weighted';
            or np.ndarray of shape (batch_size, num_classes) if reduction='none'.
        """
        batch_size, num_classes = true_positives.shape

        # 1) No reduction: AP per (sample, class)
        if reduction == "none":
            ap_matrix = np.zeros((batch_size, num_classes), dtype=float)
            for i in range(batch_size):
                for c in range(num_classes):
                    tp_val = int(true_positives[i, c])
                    fp_val = int(false_positives[i, c])
                    fn_val = int(false_negatives[i, c])
                    ap_val = compute_average_precision_score(tp_val, fp_val, fn_val)
                    ap_matrix[i, c] = ap_val
            return ap_matrix

        # 2) Micro: sum all TP/FP/FN and compute one AP
        if reduction == "micro":
            tp_total = int(true_positives.sum())
            fp_total = int(false_positives.sum())
            fn_total = int(false_negatives.sum())
            return compute_average_precision_score(tp_total, fp_total, fn_total)

        # 3) Imagewise: compute per-sample AP (sum over classes), then mean
        if reduction == "imagewise":
            ap_per_image = np.zeros(batch_size, dtype=float)
            for i in range(batch_size):
                tp_i = int(true_positives[i].sum())
                fp_i = int(false_positives[i].sum())
                fn_i = int(false_negatives[i].sum())
                ap_per_image[i] = compute_average_precision_score(tp_i, fp_i, fn_i)
            return float(ap_per_image.mean())

        # For macro and weighted: first aggregate per class across batch
        tp_per_class = true_positives.sum(axis=0).astype(int)   # shape (num_classes,)
        fp_per_class = false_positives.sum(axis=0).astype(int)
        fn_per_class = false_negatives.sum(axis=0).astype(int)

        # 4) Macro: average AP across classes equally
        if reduction == "macro":
            ap_per_class = np.zeros(num_classes, dtype=float)
            for c in range(num_classes):
                ap_per_class[c] = compute_average_precision_score(
                    tp_per_class[c],
                    fp_per_class[c],
                    fn_per_class[c]
                )
            return float(ap_per_class.mean())

        # 5) Weighted: class-wise AP weighted by support = TP + FN
        if reduction == "weighted":
            ap_per_class = np.zeros(num_classes, dtype=float)
            support = np.zeros(num_classes, dtype=float)
            for c in range(num_classes):
                tp_c = tp_per_class[c]
                fp_c = fp_per_class[c]
                fn_c = fn_per_class[c]
                ap_per_class[c] = compute_average_precision_score(tp_c, fp_c, fn_c)
                support[c] = tp_c + fn_c
            total_support = support.sum()
            if total_support == 0:
                # fallback to unweighted macro if no positive instances
                return float(ap_per_class.mean())
            weights = support / total_support
            return float((ap_per_class * weights).sum())

        raise ValueError(f"Unknown reduction mode: {reduction}")
    
    
    @staticmethod
    def __sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid activation for numpy arrays.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of the input.
        """
        return 1 / (1 + np.exp(-z))
    
    
    def __save_prediction_masks(
        self,
        sample: Dict[str, Any],
        predicted_mask: Union[np.ndarray, torch.Tensor],
        start_index: int = 0,
    ) -> None:
        """
        Save multi-channel predicted masks as TIFFs and corresponding visualizations as PNGs in separate folders.

        Args:
            sample (Dict[str, Any]): Batch sample from MONAI LoadImaged (contains 'image', optional 'mask', and 'image_meta_dict').
            predicted_mask (np.ndarray or torch.Tensor): Array of shape (C, H, W) or (B, C, H, W).
            start_index (int): Starting index for naming when metadata is missing.
        """
        # Determine base paths
        base_output_dir = self._dataset_setup.common.predictions_dir
        masks_dir = base_output_dir
        plots_dir = os.path.join(base_output_dir, "plots")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Extract image (C, H, W) or batch of images (B, C, H, W), and metadata
        image_obj = sample.get("image")  # Expected shape: (C, H, W) or (B, C, H, W)
        mask_obj = sample.get("mask")    # Expected shape: (C, H, W) or (B, C, H, W)
        image_meta = sample.get("image_meta_dict")

        # Convert tensors to numpy
        def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        image_array = to_numpy(image_obj) if image_obj is not None else None
        mask_array = to_numpy(mask_obj) if mask_obj is not None else None
        pred_array = to_numpy(predicted_mask)

        # Handle batch dimension: (B, C, H, W)
        if pred_array.ndim == 4:
            for idx in range(pred_array.shape[0]):
                batch_sample = dict(sample)
                if image_array is not None and image_array.ndim == 4:
                    batch_sample["image"] = image_array[idx]
                    if isinstance(image_meta, list):
                        batch_sample["image_meta_dict"] = image_meta[idx]
                if mask_array is not None and mask_array.ndim == 4:
                    batch_sample["mask"] = mask_array[idx]
                self.__save_prediction_masks(
                    batch_sample,
                    pred_array[idx],
                    start_index=start_index+idx
                )
            return

        # Determine base filename
        if image_meta and "filename_or_obj" in image_meta:
            base_name = os.path.splitext(os.path.basename(image_meta["filename_or_obj"]))[0]
        else:
            # Use provided start_index when metadata missing
            base_name = f"prediction_{start_index:04d}"

       # Now pred_array shape is (C, H, W)
        num_channels = pred_array.shape[0]
        for channel_idx in range(num_channels):
            channel_mask = pred_array[channel_idx]

            # File names
            mask_filename = f"{base_name}_ch{channel_idx:02d}.tif"
            plot_filename = f"{base_name}_ch{channel_idx:02d}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            plot_path = os.path.join(plots_dir, plot_filename)

            # Save mask TIFF (16-bit)
            tiff.imwrite(mask_path, channel_mask.astype(np.uint16), compression="zlib")

            # Extract corresponding true mask channel if exists
            true_mask = None
            if mask_array is not None and mask_array.ndim == 3:
                true_mask = mask_array[channel_idx]

            # Generate and save visualization
            self.__plot_mask(
                file_path=plot_path,
                image_data=image_array, # type: ignore
                predicted_mask=channel_mask,
                true_mask=true_mask,
            )


    def __plot_mask(
        self,
        file_path: str,
        image_data: np.ndarray,
        predicted_mask: np.ndarray,
        true_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Create and save grid visualization: 1x3 if no true mask, or 2x3 if true mask provided.
        """
        img = np.moveaxis(image_data, 0, -1) if image_data.ndim == 3 else image_data

        if true_mask is None:
            fig, axs = plt.subplots(1, 3, figsize=(15,5))
            plt.subplots_adjust(wspace=0.02, hspace=0)
            self.__plot_panels(axs, img, predicted_mask, 'red',
                              ('Original Image','Predicted Mask','Predicted Contours'))
        else:
            fig, axs = plt.subplots(2,3,figsize=(15,10))
            plt.subplots_adjust(wspace=0.02, hspace=0.1)
            # row 0: predicted
            self.__plot_panels(axs[0], img, predicted_mask, 'red',
                              ('Original Image','Predicted Mask','Predicted Contours'))
            # row 1: true
            self.__plot_panels(axs[1], img, true_mask, 'blue',
                              ('Original Image','True Mask','True Contours'))
        fig.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        

    def __plot_panels(
        self,
        axes,
        img: np.ndarray,
        mask: np.ndarray,
        contour_color: str,
        titles: Tuple[str, ...]
    ):
        """
        Plot a row of three panels: original image, mask, and mask boundaries on image.

        Args:
            axes: list/array of three Axis objects.
            img: Image array (H, W or H, W, C).
            mask: Label mask (H, W).
            contour_color: Color for boundaries.
            titles: (title_img, title_mask, title_contours).
        """
        # Panel 1: Original image
        ax0, ax1, ax2 = axes
        ax0.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax0.set_title(titles[0]); ax0.axis('off')
        
        # Compute boundaries once
        boundaries = find_boundaries(mask, mode='thick')
        
        # Panel 2: Mask with black boundaries
        cmap = plt.get_cmap("gist_ncar")
        cmap = mcolors.ListedColormap([
            cmap(i/len(np.unique(mask))) for i in range(len(np.unique(mask)))
        ])
        ax1.imshow(mask, cmap=cmap)
        ax1.contour(boundaries, colors='black', linewidths=0.5)
        ax1.set_title(titles[1])
        ax1.axis('off')
        
        # Panel 3: Original image with black contour overlay
        ax2.imshow(img)
        # Draw boundaries as contour lines
        ax2.contour(boundaries, colors=contour_color, linewidths=0.5)
        ax2.set_title(titles[2])
        ax2.axis('off')
        

    def __compute_flows_from_masks(
        self,
        true_masks: Tensor
    ) -> np.ndarray:
        """
        Convert segmentation masks to flow fields for training.

        Args:
            true_masks: Torch tensor of shape (batch, C, H, W) containing integer masks.

        Returns:
            numpy array of concatenated [flow_vectors, renumbered_true_masks] per image.
            renumbered_true_masks is labels, flow_vectors[0] is Y flow, flow_vectors[1] is X flow.
        """
        # Move to CPU numpy
        _true_masks: np.ndarray = true_masks.cpu().numpy().astype(np.int16)
        batch_size = _true_masks.shape[0]

        # Ensure each label has a channel dimension
        if _true_masks.ndim == 3:
            # shape (batch, H, W) -> (batch, 1, H, W)
            _true_masks = _true_masks[:, np.newaxis, :, :]

        batch_size, *_ = _true_masks.shape
        
        # Renumber labels to ensure uniqueness
        renumbered: np.ndarray = np.stack([fastremap.renumber(_true_masks[i], in_place=True)[0]
                                        for i in range(batch_size)])
        # Compute vector flows per image
        flow_vectors = np.stack([self.__compute_flow_from_mask(renumbered[i])
                        for i in range(batch_size)])
        
        return np.concatenate((flow_vectors, renumbered), axis=1).astype(np.float32)
    
    
    def __compute_flow_from_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized flow vectors from a labeled mask.

        Args:
            mask: 3D array of instance-labeled mask of shape (C, H, W).

        Returns:
            flow: Array of shape (2 * C, H, W).
        """
        if mask.max() == 0 or np.count_nonzero(mask) <= 1:
            # No flow to compute
            logger.warning("Empty mask!")
            C, H, W = mask.shape
            return np.zeros((2*C, H, W), dtype=np.float32)

        # Delegate to GPU or CPU routine
        if self._device.type == "cuda" or self._device.type == "mps":
            return self.__mask_to_flow_gpu(mask)
        else:
            return self.__mask_to_flow_cpu(mask)
        
    
    def __mask_to_flow_gpu(self, mask: np.ndarray) -> np.ndarray:
        """Convert masks to flows using diffusion from center pixel.

        Center of masks where diffusion starts is defined by pixel closest to median within the mask.

        Args:
            masks (3D array): Labelled masks of shape (C, H, W).
            
        Returns:
            np.ndarray: A 3D array where for each channel the flows for each pixel
            are represented along the X and Y axes.
        """
        
        channels, height, width = mask.shape
        flows = np.zeros((2*channels, height, width), np.float32)
        
        for channel in range(channels):
            padded_height, padded_width = height + 2, width + 2

            # Pad the mask with a 1-pixel border
            masks_padded = torch.from_numpy(mask.astype(np.int64)).to(self._device)
            masks_padded = F.pad(masks_padded, (1, 1, 1, 1))

            # Get coordinates of all non-zero pixels in the padded mask
            y, x = torch.nonzero(masks_padded, as_tuple=True)
            y = y.int();  x = x.int()   # ensure integer type

            # Generate 8-connected neighbors (including center) via broadcasted offsets
            offsets = torch.tensor([
                [ 0,  0],  # center
                [-1,  0],  # up
                [ 1,  0],  # down
                [ 0, -1],  # left
                [ 0,  1],  # right
                [-1, -1],  # up-left
                [-1,  1],  # up-right
                [ 1, -1],  # down-left
                [ 1,  1],  # down-right
            ], dtype=torch.int32, device=self._device)           # (9, 2)

            # coords: (N, 2)
            coords = torch.stack((y, x), dim=1)

            # neighbors: (9, N, 2)
            neighbors = offsets[:, None, :] + coords[None, :, :]

            # transpose into (2, 9, N) for the GPU kernel
            neighbors = neighbors.permute(2, 0, 1)       # first dim is y/x, second is neighbor index

            # Build connectivity mask: True where neighbor label == center label
            center_labels   = masks_padded[y, x][None, :]                   # (1, N)
            neighbor_labels = masks_padded[neighbors[0], neighbors[1]]      # (9, N)
            is_neighbor     = neighbor_labels == center_labels              # (9, N)

            # Compute object slices and pack into array for get_centers
            slices = find_objects(mask)
            slices_arr = np.array([
                [i, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop]
                for i, sl in enumerate(slices) if sl is not None
            ], dtype=int)
            
            # Compute centers (pixel indices) and extents via the provided helper
            centers, ext = self.__get_mask_centers_and_extents(mask, slices_arr)
            # Move centers to GPU and shift by +1 for padding
            meds_p = torch.from_numpy(centers).to(self._device).long() + 1    # (M, 2); +1 for padding

            # Determine number of diffusion iterations
            n_iter = 2 * ext.max()

            # Run the GPU diffusion kernel
            mu = self.__propagate_centers_gpu(
                neighbor_indices=neighbors,
                center_indices=meds_p.T,
                valid_neighbor_mask=is_neighbor,
                output_shape=(padded_height, padded_width),
                num_iterations=n_iter
            )

            # Cast to float64 and normalize flow vectors
            mu = mu.astype(np.float64)
            mu /= np.sqrt((mu**2).sum(axis=0)) + 1e-60

            # Remove the padding and write into final output
            flow_output = np.zeros((2, height, width), dtype=np.float32)
            ys_np = y.cpu().numpy() - 1
            xs_np = x.cpu().numpy() - 1
            flow_output[:, ys_np, xs_np] = mu
            flows[2*channel: 2*channel + 2] = flow_output
            
        return flows
        
    
    @staticmethod
    @njit(nogil=True)
    def __get_mask_centers_and_extents(
        label_map: np.ndarray,
        slices_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the centroids and extents of labeled regions in a 2D mask array.

        Args:
            label_map (np.ndarray): 2D array where each connected region has a unique integer label (1…K).
            slices_arr (np.ndarray): Array of shape (K, 5), where each row is
                                    (label_id, row_start, row_stop, col_start, col_stop).

        Returns:
            centers (np.ndarray): Integer array of shape (K, 2) with (row, col) center for each label.
            extents (np.ndarray): Integer array of shape (K,) giving the sum of height and width + 2 for each region.
        """
        num_regions = slices_arr.shape[0]
        centers = np.zeros((num_regions, 2), dtype=np.int32)
        extents = np.zeros(num_regions, dtype=np.int32)

        for idx in prange(num_regions):
            # Unpack slice info
            label_id   = slices_arr[idx, 0]
            row_start  = slices_arr[idx, 1]
            row_stop   = slices_arr[idx, 2]
            col_start  = slices_arr[idx, 3]
            col_stop   = slices_arr[idx, 4]

            # Extract binary submask for this label
            submask = (label_map[row_start:row_stop, col_start:col_stop] == label_id)

            # Get local coordinates of all pixels in the region
            ys, xs = np.nonzero(submask)

            # Compute the floating-point centroid within the submask
            y_mean = ys.mean()
            x_mean = xs.mean()

            # Find the pixel closest to the centroid by minimizing squared distance
            dist_sq = (ys - y_mean) ** 2 + (xs - x_mean) ** 2
            closest_idx = dist_sq.argmin()

            # Convert to global coordinates
            center_row = ys[closest_idx] + row_start
            center_col = xs[closest_idx] + col_start
            centers[idx, 0] = center_row
            centers[idx, 1] = center_col

            # Compute extent as height + width + 2 (to include one-pixel border)
            height = row_stop - row_start
            width  = col_stop - col_start
            extents[idx] = height + width + 2

        return centers, extents
    
    
    def __propagate_centers_gpu(
        self,
        neighbor_indices: torch.Tensor,
        center_indices: torch.Tensor,
        valid_neighbor_mask: torch.Tensor,
        output_shape: Tuple[int, int],
        num_iterations: int = 200
    ) -> np.ndarray:
        """
        Propagates center points across a mask using GPU-based diffusion.

        Args:
            neighbor_indices (torch.Tensor): Tensor of shape (2, 9, N) containing row and column indices for 9 neighbors per pixel.
            center_indices (torch.Tensor): Tensor of shape (2, N) with row and column indices of mask centers.
            valid_neighbor_mask (torch.Tensor): Boolean tensor of shape (9, N) indicating if each neighbor is valid.
            output_shape (Tuple[int, int]): Desired 2D shape of the diffusion tensor, e.g., (H, W).
            num_iterations (int, optional): Number of diffusion iterations. Defaults to 200.
            
        Returns:
            np.ndarray: Array of shape (2, N) with the computed flows.
        """
        # Determine total number of elements and choose dtype accordingly
        total_elems = torch.prod(torch.tensor(output_shape))
        if total_elems > 4e7 or self._device.type == "mps":
            diffusion_tensor = torch.zeros(output_shape, dtype=torch.float, device=self._device)
        else:
            diffusion_tensor = torch.zeros(output_shape, dtype=torch.double, device=self._device)

        # Unpack center row and column indices
        center_rows, center_cols = center_indices

        # Unpack neighbor row and column indices for 9 neighbors per pixel
        # Order: [0: center, 1: up, 2: down, 3: left, 4: right,
        #         5: up-left, 6: up-right, 7: down-left, 8: down-right]
        neigh_rows, neigh_cols = neighbor_indices  # each of shape (9, N)

        # Perform diffusion iterations
        for _ in range(num_iterations):
            # Add source at each mask center
            diffusion_tensor[center_rows, center_cols] += 1

            # Sample neighbor values for each pixel
            neighbor_vals = diffusion_tensor[neigh_rows, neigh_cols]  # shape (9, N)

            # Zero out invalid neighbors
            neighbor_vals *= valid_neighbor_mask

            # Update the first neighbor (index 0) with the average of valid neighbor values
            diffusion_tensor[neigh_rows[0], neigh_cols[0]] = neighbor_vals.mean(dim=0)

        # Compute spatial gradients for 2D flow: dy and dx
        # Using neighbor indices: up = 1, down = 2, left = 3, right = 4
        grad_samples = diffusion_tensor[
            neigh_rows[[2, 1, 4, 3]],  # indices [down, up, right, left]
            neigh_cols[[2, 1, 4, 3]]
        ]  # shape (4, N)

        dy = grad_samples[0] - grad_samples[1]
        dx = grad_samples[2] - grad_samples[3]

        # Stack and convert to numpy flow field with shape (2, N)
        flow_field = np.stack((dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=0)

        return flow_field


    def __mask_to_flow_cpu(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert labeled masks to flow vectors by simulating diffusion from mask centers.

        Each mask's center is chosen as the pixel closest to its geometric centroid.
        A diffusion process is run on a padded local patch, and flows are derived
        as gradients (dy, dx) of the resulting density map.

        Args:
            masks (np.ndarray): 3D integer array of labels `(C x H x W)`,
                where 0 = background and positive integers = mask IDs.

        Returns:
            flow_field (np.ndarray): Array of shape `(2*C, H, W)` containing
                flow components [dy, dx] normalized per pixel.
        """
        channels, height, width = mask.shape
        flows = np.zeros((2*channels, height, width), np.float32)
        
        for channel in range(channels):
            # Initialize flow_field with two channels: dy and dx
            flow_field = np.zeros((2, height, width), dtype=np.float64)

            # Find bounding box for each labeled mask
            mask_slices = find_objects(mask)
            # centers: List[Tuple[int, int]] = []

            # Iterate over mask labels in parallel
            for label_idx in prange(len(mask_slices)):
                slc = mask_slices[label_idx]
                if slc is None:
                    continue

                # Extract row and column slice for this mask
                row_slice, col_slice = slc
                # Add 1-pixel border around the patch
                patch_height = (row_slice.stop - row_slice.start) + 2
                patch_width  = (col_slice.stop - col_slice.start) + 2

                # Get local coordinates of mask pixels within the patch
                local_rows, local_cols = np.nonzero(
                    mask[row_slice, col_slice] == (label_idx + 1)
                )
                # Shift coords by +1 for the border padding
                local_rows = local_rows.astype(np.int32) + 1
                local_cols = local_cols.astype(np.int32) + 1

                # Compute centroid and find nearest pixel as diffusion seed
                centroid_row = local_rows.mean()
                centroid_col = local_cols.mean()
                distances = (local_cols - centroid_col) ** 2 + (local_rows - centroid_row) ** 2
                seed_index = distances.argmin()
                center_row = int(local_rows[seed_index])
                center_col = int(local_cols[seed_index])

                # Determine number of iterations
                total_iter = 2 * (patch_height + patch_width)
                
                # Initialize flat diffusion map for the local patch
                diffusion_map = np.zeros(patch_height * patch_width, dtype=np.float64)
                # Run diffusion from the seed center
                diffusion_map = self.__diffuse_from_center(
                    diffusion_map,
                    local_rows,
                    local_cols,
                    center_row,
                    center_col,
                    patch_width,
                    total_iter
                )

                # Compute flow as finite differences (gradient) on the diffusion map
                dy = (
                    diffusion_map[(local_rows + 1) * patch_width + local_cols] -
                    diffusion_map[(local_rows - 1) * patch_width + local_cols]
                )
                dx = (
                    diffusion_map[local_rows * patch_width + (local_cols + 1)] -
                    diffusion_map[local_rows * patch_width + (local_cols - 1)]
                )

                # Write flows back into the global flow_field array
                flow_field[0,
                        row_slice.start + local_rows - 1,
                        col_slice.start + local_cols - 1] = dy
                flow_field[1,
                        row_slice.start + local_rows - 1,
                        col_slice.start + local_cols - 1] = dx

                # Store center location in original image coordinates
                # centers.append(
                #     (row_slice.start + center_row - 1,
                #     col_slice.start + center_col - 1)
                # )

            # Normalize each vector [dy,dx] by its magnitude
            magnitudes = np.sqrt((flow_field**2).sum(axis=0)) + 1e-60
            flow_field /= magnitudes
            
            flows[2*channel: 2*channel + 2] = flow_field
            
        return flows


    @staticmethod
    @njit("(float64[:], int32[:], int32[:], int32, int32, int32, int32)", nogil=True)
    def __diffuse_from_center(
        diffusion_map: np.ndarray,
        row_coords: np.ndarray,
        col_coords: np.ndarray,
        center_row: int,
        center_col: int,
        patch_width: int,
        num_iterations: int
    ) -> np.ndarray:
        """
        Perform diffusion of particles from a seed pixel across a local mask patch.

        At each iteration, one particle is added at the seed, and each mask pixel's
        value is updated to the average of itself and its 8-connected neighbors.

        Args:
            diffusion_map (np.ndarray): Flat array of length patch_height * patch_width.
            row_coords (np.ndarray): 1D array of row indices for mask pixels (local coords).
            col_coords (np.ndarray): 1D array of column indices for mask pixels (local coords).
            center_row (int): Row index of the seed point in local patch coords.
            center_col (int): Column index of the seed point in local patch coords.
            patch_width (int): Width (number of columns) in the local patch.
            num_iterations (int): Number of diffusion iterations to perform.

        Returns:
            np.ndarray: Updated diffusion_map after performing diffusion.
        """
        # Compute linear indices for each mask pixel and its neighbors
        base_idx     = row_coords * patch_width + col_coords
        up           = (row_coords - 1) * patch_width + col_coords
        down         = (row_coords + 1) * patch_width + col_coords
        left         = row_coords * patch_width + (col_coords - 1)
        right        = row_coords * patch_width + (col_coords + 1)
        up_left      = (row_coords - 1) * patch_width + (col_coords - 1)
        up_right     = (row_coords - 1) * patch_width + (col_coords + 1)
        down_left    = (row_coords + 1) * patch_width + (col_coords - 1)
        down_right   = (row_coords + 1) * patch_width + (col_coords + 1)

        for _ in range(num_iterations):
            # Inject one particle at the seed location
            diffusion_map[center_row * patch_width + center_col] += 1.0

            # Update each mask pixel as the average over itself and neighbors
            diffusion_map[base_idx] = (
                diffusion_map[base_idx] +
                diffusion_map[up] + diffusion_map[down] +
                diffusion_map[left] + diffusion_map[right] +
                diffusion_map[up_left] + diffusion_map[up_right] +
                diffusion_map[down_left] + diffusion_map[down_right]
            ) * (1.0 / 9.0)

        return diffusion_map


    def __segment_instances(
        self,
        probability_map: np.ndarray,
        flow: np.ndarray,
        prob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        num_iters: int = 200,
        min_object_size: int = 15
    ) -> np.ndarray:
        """
        Generate instance segmentation masks from probability and flow fields.

        Args:
            probability_map: 3D array (channels, height, width) of cell probabilities.
            flow: 3D array (2*channels, height, width) of forward flow vectors.
            prob_threshold: threshold to binarize probability_map. (Default 0.0)
            flow_threshold: threshold for filtering bad flow masks. (Default 0.4)
            num_iters: number of iterations for flow-following. (Default 200)
            min_object_size: minimum area to keep small instances. (Default 15)

        Returns:
            3D array of uint16 instance labels for each channel.
        """
        # Create a binary mask of likely cell locations
        probability_mask = probability_map > prob_threshold

        # If no cells exceed the threshold, return an empty mask
        if not np.any(probability_mask):
            logger.warning("No cell pixels found.")
            return np.zeros_like(probability_map, dtype=np.uint16)

        # Prepare output array for instance labels
        labeled_instances = np.zeros_like(probability_map, dtype=np.uint16)

        # Process each channel independently
        for channel_index in range(probability_mask.shape[0]):
            # Extract flow vectors for this channel (two components per channel)
            channel_flow_vectors = flow[2 * channel_index : 2 * channel_index + 2]
            # Extract binary mask for this channel
            channel_mask = probability_mask[channel_index]

            nonzero_coords = np.stack(np.nonzero(channel_mask))

            # Follow the flow vectors to generate coordinate mappings
            flow_coordinates = self.__follow_flows(
                flow_field=channel_flow_vectors * channel_mask / 5.0,
                initial_coords=nonzero_coords,
                num_iters=num_iters
            )
            # If flow following fails, leave this channel empty
            if flow_coordinates is None:
                labeled_instances[channel_index] = np.zeros(
                    probability_map.shape[1:], dtype=np.uint16
                )
                continue

            if not torch.is_tensor(flow_coordinates):
                flow_coordinates = torch.from_numpy(
                    flow_coordinates).to(self._device, dtype=torch.int32)
            else:
                flow_coordinates = flow_coordinates.int()

            # Obtain preliminary instance masks by clustering the coordinates
            channel_instances_mask = self.__get_mask(
                pixel_positions=flow_coordinates,
                valid_indices=nonzero_coords,
                original_shape=probability_map.shape[1:]
            )

            # Filter out bad flow-derived instances if requested
            if channel_instances_mask.max() > 0 and flow_threshold > 0:
                channel_instances_mask = self.__remove_inconsistent_flow_masks(
                    mask=channel_instances_mask,
                    flow_network=channel_flow_vectors,
                    error_threshold=flow_threshold
                )

                # Remove small objects or holes below the minimum size
                if min_object_size > 0:
                    # channel_instances_mask = morphology.remove_small_holes(
                    #     channel_instances_mask, area_threshold=min_object_size
                    # )
                    # channel_instances_mask = morphology.remove_small_objects(
                    #     channel_instances_mask, min_size=min_object_size
                    # )
                    channel_instances_mask = self.__fill_holes_and_prune_small_masks(
                        channel_instances_mask, minimum_size=min_object_size
                    )

                labeled_instances[channel_index] = channel_instances_mask
            else:
                # No valid instances found, leave the channel empty
                labeled_instances[channel_index] = np.zeros(
                    probability_map.shape[1:], dtype=np.uint16
                )

        return labeled_instances


    def __follow_flows(
        self,
        flow_field: np.ndarray,
        initial_coords: np.ndarray,
        num_iters: int = 200
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Trace pixel positions through a flow field via iterative interpolation.

        Args:
            flow_field (np.ndarray): Array of shape (2, H, W) containing flow vectors.
            initial_coords (np.ndarray): Array of shape (2, num_points) with starting (y, x) positions.
            num_iters (int): Number of integration steps.

        Returns:
            np.ndarray or torch.Tensor: Final (y, x) positions of each point.
        """
        dims = 2
        # Extract spatial dimensions
        height, width = flow_field.shape[1:]

        # Choose GPU/MPS path if available
        if self._device.type in ("cuda", "mps"):
            # Prepare point tensor: shape [1, 1, num_points, 2]
            pts = torch.zeros((1, 1, initial_coords.shape[1], dims),
                            dtype=torch.float32, device=self._device)
            # Prepare flow volume: shape [1, 2, height, width]
            flow_vol = torch.zeros((1, dims, height, width),
                                dtype=torch.float32, device=self._device)

            # Load initial positions and flow into tensors (flip order for grid_sample)
            # dim 0 = x
            # dim 1 = y
            for i in range(dims):
                pts[0, 0, :, dims - i - 1] = (
                    torch.from_numpy(initial_coords[i])
                        .to(self._device, torch.float32)
                )
                flow_vol[0, dims - i - 1] = (
                    torch.from_numpy(flow_field[i])
                        .to(self._device, torch.float32)
                )

            # Prepare normalization factors for x and y (max index)
            max_indices = torch.tensor([width - 1, height - 1],
                                    dtype=torch.float32, device=self._device)
            # Reshape for broadcasting to point tensor dims
            max_idx_pt = max_indices.view(1, 1, 1, dims)
            # Reshape for broadcasting to flow volume dims
            max_idx_flow = max_indices.view(1, dims, 1, 1)

            # Normalize flow values to [-1, 1] range
            flow_vol = (flow_vol * 2) / max_idx_flow
            # Normalize points to [-1, 1]
            pts = (pts / max_idx_pt) * 2 - 1

            # Iterate: sample flow and update points
            for _ in range(num_iters):
                sampled = torch.nn.functional.grid_sample(
                    flow_vol, pts, align_corners=False
                )
                # Update each coordinate and clamp to valid range
                for i in range(dims):
                    pts[..., i] = torch.clamp(pts[..., i] + sampled[0, i], -1.0, 1.0)

            # Denormalize back to original pixel coordinates
            pts = (pts + 1) * 0.5 * max_idx_pt
            # Swap channels back to (y, x) and flatten
            final_pts = pts[..., [1, 0]].squeeze()
            # Convert from (num_points, 2) to (2, num_points)
            return final_pts.T if final_pts.ndim > 1 else final_pts.unsqueeze(0).T

        # CPU fallback using numpy and scipy
        current_pos = initial_coords.copy().astype(np.float32)
        temp_delta = np.zeros_like(current_pos, dtype=np.float32)

        for _ in range(num_iters):
            # Interpolate flow at current positions
            self.__map_coordinates(flow_field, current_pos[0], current_pos[1], temp_delta)
            # Update positions and clamp to image bounds
            current_pos[0] = np.clip(current_pos[0] + temp_delta[0], 0, height - 1)
            current_pos[1] = np.clip(current_pos[1] + temp_delta[1], 0, width - 1)

        return current_pos


    @staticmethod
    @njit([
        "(int16[:,:,:], float32[:], float32[:], float32[:,:])",
        "(float32[:,:,:], float32[:], float32[:], float32[:,:])"
    ], cache=True)
    def __map_coordinates(
        image_data: np.ndarray,
        y_coords: np.ndarray,
        x_coords: np.ndarray,
        output: np.ndarray
    ) -> None:
        """
        Perform in-place bilinear interpolation on an image volume.

        Args:
            image_data (np.ndarray): Input volume with shape (C, H, W).
            y_coords (np.ndarray): Array of new y positions (num_points).
            x_coords (np.ndarray): Array of new x positions (num_points).
            output (np.ndarray): Output array of shape (C, num_points) to fill.

        Returns:
            None. Results written directly into `output`.
        """
        channels, height, width = image_data.shape
        # Compute integer (floor) and fractional parts for coords
        y_floor = y_coords.astype(np.int32)
        x_floor = x_coords.astype(np.int32)
        y_frac = y_coords - y_floor
        x_frac = x_coords - x_floor

        # Loop over each sample point
        for idx in range(y_floor.shape[0]):
            # Clamp base indices to valid range
            y0 = min(max(y_floor[idx], 0), height - 1)
            x0 = min(max(x_floor[idx], 0), width - 1)
            y1 = min(y0 + 1, height - 1)
            x1 = min(x0 + 1, width - 1)

            wy = y_frac[idx]
            wx = x_frac[idx]

            # Interpolate per channel
            for c in range(channels):
                v00 = np.float32(image_data[c, y0, x0])
                v10 = np.float32(image_data[c, y0, x1])
                v01 = np.float32(image_data[c, y1, x0])
                v11 = np.float32(image_data[c, y1, x1])
                # Bilinear interpolation formula
                output[c, idx] = (
                    v00 * (1 - wy) * (1 - wx) +
                    v10 * (1 - wy) * wx +
                    v01 * wy * (1 - wx) +
                    v11 * wy * wx
                )


    def __get_mask(
        self,
        pixel_positions: torch.Tensor,
        valid_indices: np.ndarray,
        original_shape: Tuple[int, ...],
        pad_radius: int = 20,
        max_size_fraction: float = 0.4
    ) -> np.ndarray:
        """
        Generate labeled masks by clustering pixel trajectories via histogram peaks and region growing.

        This function executes the following steps:
        1. Pads and clamps pixel final positions to avoid border effects.
        2. Builds a dense histogram of pixel counts over spatial bins.
        3. Identifies local maxima in the histogram as seed points.
        4. Extracts local patches around each seed and grows regions by iteratively adding neighbors
            that exceed an intensity threshold.
        5. Maps grown patches back to original image indices.
        6. Removes any masks that exceed a maximum size fraction of the image.

        Args:
            pixel_positions (torch.Tensor): Tensor of shape `[2, N_pixels]`, dtype=int, containing
                final pixel coordinates after dynamics for each dimension.
            valid_indices (np.ndarray): Integer array of shape `[2, N_pixels]` 
                giving indices of pixels in the original image grid.
            original_shape (tuple of ints): Spatial dimensions of the original image, e.g. (H, W).
            pad_radius (int): Number of zero-padding pixels added on each side of the histogram.
                Defaults to 20.
            max_size_fraction (float): If any mask has a pixel count > max_size_fraction * total_pixels,
                it will be removed. Defaults to 0.4.

        Returns:
            np.ndarray: Integer mask array of shape `original_shape` with labels 0 (background) and 1..M.

        Raises:
            ValueError: If input dimensions are inconsistent or pixel_positions shape is invalid.
        """
        # Validate inputs
        ndim = len(original_shape)
        if pixel_positions.ndim != 2 or pixel_positions.size(0) != ndim:
            msg = f"pixel_positions must be shape [{ndim}, N], got {tuple(pixel_positions.shape)}"
            logger.error(msg)
            raise ValueError(msg)
        if pad_radius < 0:
            msg = f"pad_radius must be non-negative, got {pad_radius}"
            logger.error(msg)
            raise ValueError(msg)

        # Step 1: Pad and clamp pixel positions
        padded_positions = pixel_positions.clone().to(torch.int64) + pad_radius
        for dim in range(ndim):
            max_val = original_shape[dim] + pad_radius - 1
            padded_positions[dim] = torch.clamp(padded_positions[dim], min=0, max=max_val)

        # Build histogram dimensions
        hist_shape = tuple(s + 2 * pad_radius for s in original_shape)

        # Step 2: Create sparse tensor and densify to get per-pixel counts
        try:
            counts_sparse = torch.sparse_coo_tensor(
                padded_positions,
                torch.ones(padded_positions.shape[1], dtype=torch.int32, device=pixel_positions.device),
                size=hist_shape
            )
            histogram = counts_sparse.to_dense()
        except Exception as e:
            logger.error("Failed to build dense histogram: %s", e)
            raise

        # Step 3: Find peaks via 5x5 max-pooling
        k = 5
        pooled = F.max_pool2d(
            histogram.unsqueeze(0),
            kernel_size=k,
            stride=1,
            padding=k // 2
        ).squeeze()
        # Seeds are positions where histogram equals local max and count > threshold
        seed_positions = torch.nonzero((histogram - pooled == 0) & (histogram > 10))
        if seed_positions.numel() == 0:
            logger.warning("No seeds found: returning empty mask")
            return np.zeros(original_shape, dtype=np.uint16)

        # Sort seeds by ascending count to process small peaks first
        seed_counts = histogram[tuple(seed_positions.T)]
        order = torch.argsort(seed_counts)
        seed_positions = seed_positions[order]
        del pooled, counts_sparse

        # Step 4: Extract local patches and perform region growing
        num_seeds = seed_positions.shape[0]
        # Tensor to hold local patches
        patches = torch.zeros((num_seeds, 11, 11), device=pixel_positions.device)
        for idx in range(num_seeds):
            coords = seed_positions[idx]
            slices = tuple(slice(c - 5, c + 6) for c in coords)
            patches[idx] = histogram[slices]
        del histogram

        # Initialize seed mask (center pixel of each patch)
        seed_masks = torch.zeros_like(patches, device=pixel_positions.device)
        seed_masks[:, 5, 5] = 1
        # Iterative dilation and thresholding
        for _ in range(5):
            seed_masks = F.max_pool2d(
                seed_masks,
                kernel_size=3,
                stride=1,
                padding=1
            )
            seed_masks = seed_masks & (patches > 2)
        # Compute final mask coordinates
        final_coords = []
        for idx in range(num_seeds):
            coords_local = torch.nonzero(seed_masks[idx])
            # Shift back to global positions
            coords_global = coords_local + seed_positions[idx] - 5
            final_coords.append(tuple(coords_global.T))

        # Step 5: Paint masks into padded volume
        dtype = torch.int32 if num_seeds < 2**16 else torch.int64
        mask_padded = torch.zeros(hist_shape, dtype=dtype, device=pixel_positions.device)
        for label_idx, coords in enumerate(final_coords, start=1):
            mask_padded[coords] = label_idx

        # Extract only the padded positions that correspond to original pixels
        mask_values = mask_padded[tuple(padded_positions)]
        mask_values = mask_values.cpu().numpy()

        # Step 6: Map to original image and remove oversized masks
        mask_final = np.zeros(original_shape, dtype=np.uint16 if num_seeds < 2**16 else np.uint32)
        mask_final[valid_indices] = mask_values

        # Prune masks that are too large
        labels, counts = fastremap.unique(mask_final, return_counts=True)
        total_pixels = np.prod(original_shape)
        oversized = labels[counts > (total_pixels * max_size_fraction)]
        if oversized.size > 0:
            mask_final = fastremap.mask(mask_final, oversized)
        fastremap.renumber(mask_final, in_place=True)

        return mask_final

   
    def __remove_inconsistent_flow_masks(
        self,
        mask: np.ndarray,
        flow_network: np.ndarray,
        error_threshold: float = 0.4
    ) -> np.ndarray:
        """
        Remove labeled masks that have inconsistent optical flows compared to network-predicted flows.

        This performs a quality control step by computing flows from the provided masks
        and comparing them to the flows predicted by the network. Masks with a mean squared
        flow error above `error_threshold` are discarded (set to 0).

        Args:
            mask (np.ndarray): Integer mask array with shape [H, W].
                Values: 0 = no mask; 1,2,... = mask labels.
            flow_network (np.ndarray): Float array of network-predicted flows with shape
                [2, H, W].
            error_threshold (float): Maximum allowed mean squared flow error per mask label.
                Defaults to 0.4.

        Returns:
            np.ndarray: The input mask with inconsistent masks removed (labels set to 0).

        Raises:
            MemoryError: If the mask size exceeds available GPU memory.
        """
        # If mask is very large and running on CUDA, check memory
        num_pixels = mask.size
        if (
            num_pixels > 10000 * 10000

            and self._device.type == 'cuda'
        ):
            # Clear unused GPU cache
            torch.cuda.empty_cache()
            # Determine PyTorch version
            major, minor = map(int, torch.__version__.split('.')[:2])
            # Determine current CUDA device index
            device_index = (
                self._device.index
                if hasattr(self._device, 'index')
                else torch.cuda.current_device()
            )
            # Get free and total memory
            if major == 1 and minor < 10:
                total_mem = torch.cuda.get_device_properties(device_index).total_memory
                used_mem = torch.cuda.memory_allocated(device_index)
                free_mem = total_mem - used_mem
            else:
                free_mem, total_mem = torch.cuda.mem_get_info(device_index)
            # Estimate required memory for mask-based flow computation
            # Assume float32 per pixel
            required_bytes = num_pixels * np.dtype(np.float32).itemsize
            if required_bytes > free_mem:
                logger.error(
                    'Image too large for GPU memory in flow QC step (required: %d B, available: %d B)',
                    required_bytes, free_mem
                )
                raise MemoryError('Insufficient GPU memory for flow QC computation')

        # Compute flow errors per mask label
        flow_errors, _ = self.__compute_flow_error(mask, flow_network)

        # Identify labels with error above threshold
        bad_labels = np.nonzero(flow_errors > error_threshold)[0] + 1

        # Remove bad masks by setting their label to 0
        mask[np.isin(mask, bad_labels)] = 0
        return mask


    def __compute_flow_error(
        self,
        mask: np.ndarray,
        flow_network: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean squared error between network-predicted flows and flows derived from masks.

        Args:
            mask (np.ndarray): Integer masks, shape must match flow_network spatial dims.
            flow_network (np.ndarray): Network predicted flows of shape [axis, ...].

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - flow_errors: 1D array (length = max label) of mean squared error per label.
                - computed_flows: Array of flows derived from the mask, same shape as flow_network.

        Raises:
            ValueError: If the spatial dimensions of `mask_array` and `flow_network` do not match.
        """
        # Ensure mask and flow shapes match
        if flow_network.shape[1:] != mask.shape:
            logger.error(
                'Shape mismatch: network flow shape %s vs mask shape %s',
                flow_network.shape[1:], mask.shape
            )
            raise ValueError('Network flow and mask shapes must match')

        # Compute flows from mask labels (user-provided function)
        computed_flows = self.__compute_flow_from_mask(mask[None, ...])

        # Prepare array for errors (one value per mask label)
        num_labels = int(mask.max())
        flow_errors = np.zeros(num_labels, dtype=float)

        # Accumulate mean squared error over each flow axis
        for axis_index in range(computed_flows.shape[0]):
            # MSE per label: mean((computed - predicted/5)^2)
            flow_errors += mean(
                (computed_flows[axis_index] - flow_network[axis_index] / 5.0) ** 2,
                mask,
                index=np.arange(1, num_labels + 1)
            )

        return flow_errors, computed_flows


    def __fill_holes_and_prune_small_masks(
        self,
        masks: np.ndarray,
        minimum_size: int = 15
    ) -> np.ndarray:
        """
        Fill holes in labeled masks and remove masks smaller than a given size.

        This function performs two steps:
        1. Fills internal holes in each labeled mask using `fill_voids.fill`.
        2. Discards any mask whose pixel count is below `minimum_size`.

        Args:
            masks (np.ndarray): Integer mask array of dimension 2 or 3 (shape [H, W] or [D, H, W]).
                Values: 0 = background; 1,2,... = mask labels.
            minimum_size (int): Minimum number of pixels required to keep a mask. 
                Masks smaller than this will be removed. 
                Set to -1 to skip size-based pruning. Defaults to 15.

        Returns:
            np.ndarray: Processed mask array with holes filled and small masks removed.

        Raises:
            ValueError: If `masks` is not a 2D or 3D integer array.
        """
        # Validate input dimensions
        if masks.ndim not in (2, 3):
            msg = f"Expected 2D or 3D mask array, got {masks.ndim}D."
            logger.error(msg)
            raise ValueError(msg)

        # Optionally remove masks smaller than minimum_size
        if minimum_size >= 0:
            # Compute label counts (skipping background at index 0)
            labels, counts = fastremap.unique(masks, return_counts=True)
            # Identify labels to remove: those with count < minimum_size
            small_labels = labels[counts < minimum_size]
            if small_labels.size > 0:
                masks = fastremap.mask(masks, small_labels)
                fastremap.renumber(masks, in_place=True)

        # Find bounding boxes for each mask label
        object_slices = find_objects(masks)
        new_label = 1
        output_masks = np.zeros_like(masks, dtype=masks.dtype)

        # Loop over each original slice, fill holes, and assign new labels
        for original_label, slc in enumerate(object_slices, start=1):
            if slc is None:
                continue
            # Extract sub-volume or sub-image
            region = masks[slc] == original_label
            if not np.any(region):
                continue
            # Fill internal holes
            filled_region = fill_voids.fill(region)
            # Write back into output mask with sequential labels
            output_masks[slc][filled_region] = new_label
            new_label += 1

        # Final pruning of small masks after filling (optional)
        if minimum_size >= 0:
            labels, counts = fastremap.unique(output_masks, return_counts=True)
            small_labels = labels[counts < minimum_size]
            if small_labels.size > 0:
                output_masks = fastremap.mask(output_masks, small_labels)
                fastremap.renumber(output_masks, in_place=True)

        return output_masks