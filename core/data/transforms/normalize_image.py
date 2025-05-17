import numpy as np
from skimage import exposure
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import Transform, MapTransform
from typing import Hashable, Mapping, Sequence

__all__ = [
    "CustomNormalizeImage",
    "CustomNormalizeImaged",
    "CustomNormalizeImageD",
    "CustomNormalizeImageDict",
]


class CustomNormalizeImage(Transform):
    """
    Normalize the image by rescaling intensity values based on specified percentiles.

    The normalization can be applied either on the entire image or channel-wise.
    If the image is 2D (only height and width), a channel dimension is added for consistency.
    """

    def __init__(self, percentiles: Sequence[float] = (0, 99), channel_wise: bool = False) -> None:
        """
        Args:
            percentiles (Sequence(float)): Lower and upper percentiles used for intensity scaling.
                                           Default is (0, 99).
            channel_wise (bool): Whether to apply normalization on each channel individually.
                                 Default is False.
        """
        self.lower, self.upper = percentiles  # Unpack the lower and upper percentile values.
        self.channel_wise = channel_wise       # Flag for channel-wise normalization.

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Rescale image intensity using non-zero values for percentile calculation.

        Args:
            img (np.ndarray): A numpy array representing a single-channel image.

        Returns:
            np.ndarray: A uint8 numpy array with rescaled intensity values.
        """
        # Extract non-zero values to avoid background influence.
        non_zero_vals = img[np.nonzero(img)]
        # Calculate the specified percentiles from the non-zero values.
        computed_percentiles: np.ndarray = np.percentile(non_zero_vals, [self.lower, self.upper])
        # Rescale the intensity values to the full uint8 range.
        img_norm = exposure.rescale_intensity(
            img, in_range=(computed_percentiles[0], computed_percentiles[1]), out_range="uint8" # type: ignore
        )
        return img_norm.astype(np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply normalization to the input image.

        If the input image is 2D (height, width), a channel dimension is added.
        Depending on the 'channel_wise' flag, normalization is applied either to each channel individually or to the entire image.

        Args:
            img (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Normalized image as a numpy array.
        """
        # Check if the image is 2D (grayscale). If so, add a new axis for the channel.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # Added channel dimension for consistency.

        if self.channel_wise:
            # Initialize an empty array with the same shape as the input image to store normalized channels.
            normalized_img = np.zeros(img.shape, dtype=np.uint8)
            
            # Process each channel individually.
            for i in range(img.shape[-1]):
                channel_img: np.ndarray = img[:, :, i]
                
                # Only normalize the channel if there are non-zero values present.
                if np.count_nonzero(channel_img) > 0:
                    normalized_img[:, :, i] = self._normalize(channel_img)
                    
            img = normalized_img
        else:
            # Apply normalization to the entire image.
            img = self._normalize(img)

        return img


class CustomNormalizeImaged(MapTransform):
    """
    Dictionary-based wrapper for CustomNormalizeImage.

    This transform applies normalization to one or more images contained in a dictionary,
    where the keys point to the image data.
    """

    def __init__(
        self,
        keys: KeysCollection,
        percentiles: Sequence[float] = (1, 99),
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys (KeysCollection): Keys identifying the image entries in the dictionary.
            percentiles (Sequence(float)): Lower and upper percentiles used for intensity scaling.
                                           Default is (1, 99).
            channel_wise (bool): Whether to apply normalization on each channel individually.
                                 Default is False.
            allow_missing_keys (bool): If True, missing keys in the dictionary will be ignored.
                                       Default is False.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        # Create an instance of the normalization transform with specified parameters.
        self.normalizer: CustomNormalizeImage = CustomNormalizeImage(percentiles, channel_wise)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        """
        Apply the normalization transform to each image in the input dictionary.

        Args:
            data (Mapping[Hashable, np.ndarray]): A dictionary mapping keys to numpy arrays representing images.

        Returns:
            dict(Hashable, np.ndarray): A dictionary with the same keys where the images have been normalized.
        """
        # Copy the input dictionary to avoid modifying the original data.
        d: dict[Hashable, np.ndarray] = dict(data)
        # Iterate over each key specified in the transform and normalize the corresponding image.
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d


# Create aliases for the dictionary-based normalization transform.
CustomNormalizeImageD = CustomNormalizeImageDict = CustomNormalizeImaged
