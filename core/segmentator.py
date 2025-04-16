import torch 
import random
import numpy as np
from monai.data.dataset import Dataset
from monai.transforms import * # type: ignore
from torch.utils.data import DataLoader

import os
import glob
from pprint import pformat
from typing import Optional, Union

from config import Config
from core.models import *
from core.losses import *
from core.optimizers import *
from core.schedulers import *

from core.logger import get_logger


logger = get_logger()


class CellSegmentator:
    def __init__(self, config: Config) -> None:
        self.__set_seed(config.dataset_config.common.seed)
        self.__parse_config(config)

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

    
    def train(self) -> None:
        pass

  
    def evaluate(self) -> None:
        pass

 
    def predict(self) -> None:
        pass


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
        
        # Log the full configuration dictionary
        full_config_dict = {
            "model": model.dump(),
            "criterion": criterion.dump() if criterion else None,
            "optimizer": optimizer.dump() if optimizer else None,
            "scheduler": scheduler.dump() if scheduler else None,
            "dataset_config": config.dataset_config.model_dump()
        }
        logger.info("========== Parsed Configuration ==========")
        logger.info(pformat(full_config_dict, width=120))
        logger.info("==========================================")

        # Initialize model using the model registry
        self._model = ModelRegistry.get_model_class(model.name)(model.params)
        logger.info(f"Initialized model: {model.name}")
        
        # Initialize loss criterion if specified
        self._criterion = (
            CriterionRegistry.get_criterion_class(criterion.name)(params=criterion.params)
            if criterion is not None
            else None
        )
        if self._criterion is not None and criterion is not None:
            logger.info(f"Initialized criterion: {criterion.name}")
        else:
            logger.info("Criterion: not specified")

        # Initialize optimizer if specified
        self._optimizer = (
            OptimizerRegistry.get_optimizer_class(optimizer.name)(
                model_params=self._model.parameters(),
                optim_params=optimizer.params
            )
            if optimizer is not None
            else None
        )
        if self._optimizer is not None and optimizer is not None:
            logger.info(f"Initialized optimizer: {optimizer.name}")
        else:
            logger.info("Optimizer: not specified")

        # Initialize scheduler only if both scheduler and optimizer are defined
        self._scheduler = (
            SchedulerRegistry.get_scheduler_class(scheduler.name)(
                optimizer=self._optimizer.optim,
                params=scheduler.params
            )
            if scheduler is not None and self._optimizer is not None and self._optimizer.optim is not None
            else None
        )
        if self._scheduler is not None and scheduler is not None:
            logger.info(f"Initialized scheduler: {scheduler.name}")
        else:
            logger.info("Scheduler: not specified")

        # Save dataset config
        self._dataset_setup = config.dataset_config
        logger.info("Dataset setup loaded")
        common = config.dataset_config.common
        logger.info(f"Seed: {common.seed}")
        logger.info(f"Device: {common.device}")
        logger.info(f"Predictions output dir: {common.predictions_dir}")

        if config.dataset_config.is_training:
            training = config.dataset_config.training
            logger.info("Mode: Training")
            logger.info(f"  Batch size: {training.batch_size}")
            logger.info(f"  Epochs: {training.num_epochs}")
            logger.info(f"  Validation frequency: {training.val_freq}")
            logger.info(f"  Use AMP: {'yes' if training.use_amp else 'no'}")
            logger.info(f"  Pretrained weights: {training.pretrained_weights}")

            if training.is_split:
                logger.info("  Using pre-split directories:")
                logger.info(f"    Train dir: {training.pre_split.train_dir}")
                logger.info(f"    Valid dir: {training.pre_split.valid_dir}")
                logger.info(f"    Test dir:  {training.pre_split.test_dir}")
            else:
                logger.info("  Using unified dataset with splits:")
                logger.info(f"    All data dir: {training.split.all_data_dir}")
                logger.info(f"    Shuffle: {'yes' if training.split.shuffle else 'no'}")

            logger.info("  Dataset split:")
            logger.info(f"    Train size: {training.train_size}, offset: {training.train_offset}")
            logger.info(f"    Valid size: {training.valid_size}, offset: {training.valid_offset}")
            logger.info(f"    Test size:  {training.test_size}, offset: {training.test_offset}")
            
        else:
            testing = config.dataset_config.testing
            logger.info("Mode: Inference")
            logger.info(f"  Test dir: {testing.test_dir}")
            logger.info(f"  Test size: {testing.test_size} (offset: {testing.test_offset})")
            logger.info(f"  Shuffle: {'yes' if testing.shuffle else 'no'}")
            logger.info(f"  Use ensemble: {'yes' if testing.use_ensemble else 'no'}")
            logger.info(f"  Pretrained weights:")
            logger.info(f"    Single model: {testing.pretrained_weights}")
            logger.info(f"    Ensemble model 1: {testing.ensemble_pretrained_weights1}")
            logger.info(f"    Ensemble model 2: {testing.ensemble_pretrained_weights2}")



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
