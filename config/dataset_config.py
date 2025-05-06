from pydantic import BaseModel, model_validator, field_validator
from typing import Any, Dict, Optional, Union
import os


class DatasetCommonConfig(BaseModel):
    """
    Common configuration fields shared by both training and testing.
    """
    seed: Optional[int] = 0          # Seed for splitting if data is not pre-split (and all random operations)
    device: str = "cuda0"           # Device used for training/testing (e.g., 'cpu' or 'cuda')
    use_tta: bool = False           # Flag to use Test-Time Augmentation (TTA)
    use_amp: bool = False           # Flag to use Automatic Mixed Precision (AMP)
    predictions_dir: str = "."      # Directory to save predictions

    @model_validator(mode="after")
    def validate_common(self) -> "DatasetCommonConfig":
        """
        Validates that device is non-empty.
        """
        if not self.device:
            raise ValueError("device must be provided and non-empty")
        return self


class TrainingSplitInfo(BaseModel):
    """
    Configuration for training mode when data is NOT pre-split (is_split is False).
    Contains:
        - all_data_dir: Directory containing all data.
    """
    shuffle: bool = True             # Shuffle data before splitting
    all_data_dir: str = "."          # Directory containing all data if not pre-split

class TrainingPreSplitInfo(BaseModel):
    """
    Configuration for training mode when data is pre-split (is_split is True).
    Contains:
        - train_dir, valid_dir, test_dir: Directories for training, validation, and testing data.
    """
    train_dir: str = "."             # Directory for training data if data is pre-split
    valid_dir: str = ""             # Directory for validation data if data is pre-split
    test_dir: str = ""              # Directory for testing data if data is pre-split


class DatasetTrainingConfig(BaseModel):
    """
    Main training configuration.
    Contains:
        - is_split: Determines whether data is pre-split.
        - pre_split: Configuration for when data is NOT pre-split.
        - split: Configuration for when data is pre-split.
        - train_size, valid_size, test_size: Data split ratios or counts.
        - train_offset, valid_offset, test_offset: Offsets for respective splits.
        - Other training parameters: batch_size, num_epochs, val_freq, use_amp, pretrained_weights.
    
    Both pre_split and split objects are always created, but only the one corresponding
    to is_split is validated.
    """
    is_split: bool = False          # Whether the data is already split into train/validation sets
    pre_split: TrainingPreSplitInfo = TrainingPreSplitInfo()
    split: TrainingSplitInfo = TrainingSplitInfo()

    train_size: Union[int, float] = 0.7    # Training data size (int for static, float in (0,1] for dynamic)
    valid_size: Union[int, float] = 0.2    # Validation data size (int for static, float in (0,1] for dynamic)
    test_size: Union[int, float] = 0.1     # Testing data size (int for static, float in (0,1] for dynamic)
    train_offset: int = 0           # Offset for training data
    valid_offset: int = 0           # Offset for validation data
    test_offset: int = 0            # Offset for testing data 

    batch_size: int = 1             # Batch size for training
    num_epochs: int = 100           # Number of training epochs
    val_freq: int = 1               # Frequency of validation during training
    pretrained_weights: str = ""    # Path to pretrained weights for training


    @field_validator("train_size", "valid_size", "test_size", mode="before")
    def validate_sizes(cls, v: Union[int, float]) -> Union[int, float]:
        """
        Validates size values:
        - If provided as a float, must be in the range (0, 1].
        - If provided as an int, must be non-negative.
        """
        if isinstance(v, float):
            if not (0 <= v <= 1):
                raise ValueError("When provided as a float, size must be in the range (0, 1]")
        elif isinstance(v, int):
            if v < 0:
                raise ValueError("When provided as an int, size must be non-negative")
        else:
            raise ValueError("Size must be either an int or a float")
        return v

    @model_validator(mode="after")
    def validate_split_info(self) -> "DatasetTrainingConfig":
        """
        Conditionally validates the nested split objects:
          - If is_split is True, validates pre_split (train_dir must be non-empty and exist; if provided, valid_dir and test_dir must exist).
          - If is_split is False, validates split (all_data_dir must be non-empty and exist).
        """
        if any(isinstance(s, float) for s in (self.train_size, self.valid_size, self.test_size)):
            if (self.train_size + self.valid_size + self.test_size) > 1:
                raise ValueError("The total sample size with dynamically defined sizes must be <= 1")
        
        if not self.is_split:
            if not self.split.all_data_dir:
                raise ValueError("When is_split is False, all_data_dir must be provided and non-empty in pre_split")
            if not os.path.exists(self.split.all_data_dir):
                raise ValueError(f"Path for all_data_dir does not exist: {self.split.all_data_dir}")
        else:
            if not self.pre_split.train_dir:
                raise ValueError("When is_split is True, train_dir must be provided and non-empty in split")
            if not os.path.exists(self.pre_split.train_dir):
                raise ValueError(f"Path for train_dir does not exist: {self.pre_split.train_dir}")
            if self.pre_split.valid_dir and not os.path.exists(self.pre_split.valid_dir):
                raise ValueError(f"Path for valid_dir does not exist: {self.pre_split.valid_dir}")
            if self.pre_split.test_dir and not os.path.exists(self.pre_split.test_dir):
                raise ValueError(f"Path for test_dir does not exist: {self.pre_split.test_dir}")
        return self

    @model_validator(mode="after")
    def validate_numeric_fields(self) -> "DatasetTrainingConfig":
        """
        Validates numeric fields:
        - batch_size and num_epochs must be > 0.
        - val_freq must be >= 0.
        - offsets must be >= 0.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.val_freq < 0:
            raise ValueError("val_freq must be >= 0")
        if self.train_offset < 0 or self.valid_offset < 0 or self.test_offset < 0:
            raise ValueError("offsets must be >= 0")
        return self

    @model_validator(mode="after")
    def validate_pretrained(self) -> "DatasetTrainingConfig":
        """
        Validates that pretrained_weights is provided and exists.
        """
        if self.pretrained_weights and not os.path.exists(self.pretrained_weights):
            raise ValueError(f"Path for pretrained_weights does not exist: {self.pretrained_weights}")
        return self


class DatasetTestingConfig(BaseModel):
    """
    Configuration fields used only in testing mode.
    """
    test_dir: str = "."                    # Test data directory; must be non-empty
    test_size: Union[int, float] = 1.0     # Testing data size (int for static, float in (0,1] for dynamic)
    test_offset: int = 0                # Offset for testing data 
    shuffle: bool = True                # Shuffle data
    
    use_ensemble: bool = False          # Flag to use ensemble mode in testing
    ensemble_pretrained_weights1: str = "."
    ensemble_pretrained_weights2: str = "."
    pretrained_weights: str = "."

    @field_validator("test_size", mode="before")
    def validate_test_size(cls, v: Union[int, float]) -> Union[int, float]:
        """
        Validates the test_size value.
        """
        if isinstance(v, float):
            if not (0 < v <= 1):
                raise ValueError("When provided as a float, test_size must be in the range (0, 1]")
        elif isinstance(v, int):
            if v < 0:
                raise ValueError("When provided as an int, test_size must be non-negative")
        else:
            raise ValueError("test_size must be either an int or a float")
        return v
    
    @model_validator(mode="after")
    def validate_numeric_fields(self) -> "DatasetTestingConfig":
        """
        Validates numeric fields:
        - test_offset must be >= 0.
        """
        if self.test_offset < 0:
            raise ValueError("test_offset must be >= 0")
        return self

    @model_validator(mode="after")
    def validate_testing(self) -> "DatasetTestingConfig":
        """
        Validates the testing configuration:
        - test_dir must be non-empty and exist.
        - If use_ensemble is True, both ensemble_pretrained_weights1 and ensemble_pretrained_weights2 must be provided and exist.
        - If use_ensemble is False, pretrained_weights must be provided and exist.
        """
        if not self.test_dir:
            raise ValueError("In testing configuration, test_dir must be provided and non-empty")
        if not os.path.exists(self.test_dir):
            raise ValueError(f"Path for test_dir does not exist: {self.test_dir}")
        if self.use_ensemble:
            for field in ["ensemble_pretrained_weights1", "ensemble_pretrained_weights2"]:
                value = getattr(self, field)
                if not value:
                    raise ValueError(f"When use_ensemble is True, {field} must be provided and non-empty")
                if not os.path.exists(value):
                    raise ValueError(f"Path for {field} does not exist: {value}")
        else:
            if not self.pretrained_weights:
                raise ValueError("When use_ensemble is False, pretrained_weights must be provided and non-empty")
            if not os.path.exists(self.pretrained_weights):
                raise ValueError(f"Path for pretrained_weights does not exist: {self.pretrained_weights}")
        if self.test_offset < 0:
            raise ValueError("test_offset must be >= 0")
        return self


class DatasetConfig(BaseModel):
    """
    Main dataset configuration that groups fields into nested models for a structured and readable JSON.
    """
    is_training: bool = True   # Flag indicating whether the configuration is for training (True) or testing (False)
    common: DatasetCommonConfig = DatasetCommonConfig()
    training: DatasetTrainingConfig = DatasetTrainingConfig()
    testing: DatasetTestingConfig = DatasetTestingConfig()

    @model_validator(mode="after")
    def validate_config(self) -> "DatasetConfig":
        """
        Validates the overall dataset configuration:
        """
        if self.is_training:
            if self.training is None:
                raise ValueError("Training configuration must be provided when is_training is True")
            if self.training.train_size == 0:
                raise ValueError("train_size must be provided when is_training is True")
            if (self.training.is_split and self.training.pre_split.test_dir) or (not self.training.is_split):
                if self.training.test_size > 0 and not self.common.predictions_dir:
                    raise ValueError("predictions_dir must be provided when test_size is non-zero")
            if self.common.predictions_dir and not os.path.exists(self.common.predictions_dir):
                raise ValueError(f"Path for predictions_dir does not exist: {self.common.predictions_dir}")
        else:
            if self.testing is None:
                raise ValueError("Testing configuration must be provided when is_training is False")
            if self.testing.test_size > 0 and not self.common.predictions_dir:
                raise ValueError("predictions_dir must be provided when test_size is non-zero")
            if self.common.predictions_dir and not os.path.exists(self.common.predictions_dir):
                raise ValueError(f"Path for predictions_dir does not exist: {self.common.predictions_dir}")
        return self

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Dumps only the relevant configuration depending on the is_training flag.
        Only the nested configuration (training or testing) along with common fields is returned.
        """
        if self.is_training:
            return {
                "is_training": self.is_training,
                "common": self.common.model_dump(),
                "training": self.training.model_dump() if self.training else {}
            }
        else:
            return {
                "is_training": self.is_training,
                "common": self.common.model_dump(),
                "testing": self.testing.model_dump() if self.testing else {}
            }
