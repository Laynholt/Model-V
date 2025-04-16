from .cell_aware import IntensityDiversification
from .load_image import CustomLoadImage, CustomLoadImaged
from .normalize_image import CustomNormalizeImage, CustomNormalizeImaged

from monai.transforms import * # type: ignore


__all__ = [
    "get_train_transforms",
    "get_valid_transforms",
    "get_test_transforms",
    "get_predict_transforms",
]


def get_train_transforms():
    """
    Returns the transformation pipeline for training data.

    The training pipeline applies a series of image and label preprocessing steps:
      1. Load image and label data.
      2. Normalize the image intensities.
      3. Ensure the image and label have channel-first format.
      4. Scale image intensities.
      5. Apply spatial transformations (zoom, padding, cropping, flipping, and rotation).
      6. Diversify intensities for selected cell regions.
      7. Apply additional intensity perturbations (noise, contrast, smoothing, histogram shift, and sharpening).
      8. Convert the data types to the desired format.

    Returns:
        Compose: The composed transformation pipeline for training.
    """
    train_transforms = Compose(
        [
            # Load image and label data in (H, W, C) format (image loaded as image-only).
            CustomLoadImaged(keys=["image", "mask"], image_only=True),
            # Normalize the (H, W, C) image using the specified percentiles.
            CustomNormalizeImaged(
                keys=["image"],
                allow_missing_keys=True,
                channel_wise=False,
                percentiles=[0.0, 99.5],
            ),
            # Ensure both image and label are in channel-first format.
            EnsureChannelFirstd(keys=["image", "mask"], channel_dim=-1),
            # Scale image intensities (do not scale the label).
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            # Apply random zoom to both image and label.
            RandZoomd(
                keys=["image", "mask"],
                prob=0.5,
                min_zoom=0.25,
                max_zoom=1.5,
                mode=["area", "nearest"],
                keep_size=False,
            ),
            # Pad spatial dimensions to ensure a size of 512.
            SpatialPadd(keys=["image", "mask"], spatial_size=512),
            # Randomly crop a region of interest of size 512.
            RandSpatialCropd(keys=["image", "mask"], roi_size=512, random_size=False),
            # Randomly flip the image and label along an axis.
            RandAxisFlipd(keys=["image", "mask"], prob=0.5),
            # Randomly rotate the image and label by 90 degrees.
            RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=(0, 1)),
            # Diversify intensities for selected cell regions.
            IntensityDiversification(keys=["image", "mask"], allow_missing_keys=True),
            # Apply random Gaussian noise to the image.
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0, std=0.1),
            # Randomly adjust the contrast of the image.
            RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(1, 2)),
            # Apply random Gaussian smoothing to the image.
            RandGaussianSmoothd(keys=["image"], prob=0.25, sigma_x=(1, 2)),
            # Randomly shift the histogram of the image.
            RandHistogramShiftd(keys=["image"], prob=0.25, num_control_points=3),
            # Apply random Gaussian sharpening to the image.
            RandGaussianSharpend(keys=["image"], prob=0.25),
            # Ensure that the data types are correct.
            EnsureTyped(keys=["image", "mask"]),
        ]
    )
    return train_transforms


def get_valid_transforms():
    """
    Returns the transformation pipeline for validation data.

    The validation pipeline includes the following steps:
      1. Load image and label data (with missing keys allowed).
      2. Normalize the image intensities.
      3. Ensure the image and label are in channel-first format.
      4. Scale image intensities.
      5. Convert the data types to the desired format.

    Returns:
        Compose: The composed transformation pipeline for validation.
    """
    valid_transforms = Compose(
        [
            # Load image and label data in (H, W, C) format (image loaded as image-only; allow missing keys).
            CustomLoadImaged(keys=["image", "mask"], allow_missing_keys=True, image_only=True),
            # Normalize the (H, W, C) image using the specified percentiles.
            CustomNormalizeImaged(
                keys=["image"],
                allow_missing_keys=True,
                channel_wise=False,
                percentiles=[0.0, 99.5],
            ),
            # Ensure both image and label are in channel-first format.
            EnsureChannelFirstd(keys=["image", "mask"], allow_missing_keys=True, channel_dim=-1),
            # Scale image intensities.
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            # Ensure that the data types are correct.
            EnsureTyped(keys=["image", "mask"], allow_missing_keys=True),
        ]
    )
    return valid_transforms


def get_test_transforms():
    """
    Returns the transformation pipeline for test data.

    The test pipeline is similar to the validation pipeline and includes:
      1. Load image and label data (with missing keys allowed).
      2. Normalize the image intensities.
      3. Ensure the image and label are in channel-first format.
      4. Scale image intensities.
      5. Convert the data types to the desired format.

    Returns:
        Compose: The composed transformation pipeline for testing.
    """
    test_transforms = Compose(
        [
            # Load image and label data in (H, W, C) format (image loaded as image-only; allow missing keys).
            CustomLoadImaged(keys=["image", "mask"], allow_missing_keys=True, image_only=True),
            # Normalize the (H, W, C) image using the specified percentiles.
            CustomNormalizeImaged(
                keys=["image"],
                allow_missing_keys=True,
                channel_wise=False,
                percentiles=[0.0, 99.5],
            ),
            # Ensure both image and label are in channel-first format.
            EnsureChannelFirstd(keys=["image", "mask"], allow_missing_keys=True, channel_dim=-1),
            # Scale image intensities.
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            # Ensure that the data types are correct.
            EnsureTyped(keys=["image", "mask"], allow_missing_keys=True),
        ]
    )
    return test_transforms


def get_predict_transforms():
    """
    Returns the transformation pipeline for prediction preprocessing.

    The prediction pipeline includes the following steps:
      1. Load the image data.
      2. Normalize the image intensities.
      3. Ensure the image is in channel-first format.
      4. Scale image intensities.
      5. Convert the image to the required tensor type.

    Returns:
        Compose: The composed transformation pipeline for prediction.
    """
    pred_transforms = Compose(
        [
            # Load the image data in (H, W, C) format (image loaded as image-only).
            CustomLoadImage(image_only=True),
            # Normalize the (H, W, C) image using the specified percentiles.
            CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
            # Ensure the image is in channel-first format.
            EnsureChannelFirst(channel_dim=-1),  # image shape: (C, H, W)
            # Scale image intensities.
            ScaleIntensity(),
            # Convert the image to the required tensor type.
            EnsureType(data_type="tensor"),
        ]
    )
    return pred_transforms
