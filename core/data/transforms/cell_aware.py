import copy
import torch
import numpy as np
from typing import Sequence
from skimage.segmentation import find_boundaries
from monai.transforms import RandScaleIntensity, Compose, MapTransform # type: ignore

from core.logger import get_logger

__all__ = ["BoundaryExclusion", "IntensityDiversification"]


logger = get_logger(__name__)


class BoundaryExclusion(MapTransform):
    """
    Map the cell boundary pixel labels to the background class (0).

    This transform processes a label image by first detecting boundaries of cell regions
    and then excluding those boundary pixels by setting them to 0. However, it retains
    the original cell label if the cell is too small (less than 14x14 pixels) or if the cell
    touches the image boundary.
    """

    def __init__(self, keys: Sequence[str] = ("mask",), allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys (Sequence(str)): Keys in the input dictionary corresponding to the label image.
                                  Default is ("mask",).
            allow_missing_keys (bool): If True, missing keys in the input will be ignored.
                                       Default is False.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply the boundary exclusion transform to the label image.

        The process involves:
          1. Deep-copying the original label.
          2. Finding boundaries using a thick mode with connectivity=1.
          3. Setting the boundary pixels to background (0).
          4. Restoring original labels for cells that are too small (< 14x14 pixels).
          5. Ensuring that cells touching the image boundary are not excluded.
          6. Assigning the transformed label back into the input dictionary.

        Args:
            data (Dict(str, np.ndarray)): Dictionary containing at least the "mask" key with a label image.

        Returns:
            Dict(str, np.ndarray): The input dictionary with the "mask" key updated after boundary exclusion.
        """
        # Retrieve the original label image.
        label_original: np.ndarray = data["mask"]
        # Create a deep copy of the original label for processing.
        label: np.ndarray = copy.deepcopy(label_original)
        # Detect cell boundaries with a thick boundary.
        boundary: np.ndarray = find_boundaries(label, connectivity=1, mode="thick")
        # Exclude boundary pixels by setting them to 0.
        label[boundary] = 0

        # Create a new label copy for selective exclusion.
        new_label: np.ndarray = copy.deepcopy(label_original)
        new_label[label == 0] = 0

        # Obtain unique cell indices and their pixel counts.
        cell_idx, cell_counts = np.unique(label_original, return_counts=True)

        # If a cell is too small (< 196 pixels, approx. 14x14), restore its original label.
        for k in range(len(cell_counts)):
            if cell_counts[k] < 196:
                new_label[label_original == cell_idx[k]] = cell_idx[k]

        # Ensure that cells at the image boundaries are not excluded.
        # Get the dimensions of the label image.
        H, W, _ = label_original.shape
        # Create a binary mask with a border of 2 pixels preserved.
        bd: np.ndarray = np.zeros_like(label_original, dtype=label.dtype)
        bd[2 : H - 2, 2 : W - 2, :] = 1
        # Combine the preserved boundaries with the new label.
        new_label += label_original * bd

        # Update the input dictionary with the transformed label.
        data["mask"] = new_label

        return data


class IntensityDiversification(MapTransform):
    """
    Randomly rescale the intensity of cell pixels.

    This transform selects a subset of cells (based on the change_cell_ratio) and
    applies a random intensity scaling to those cells. The intensity scaling is performed
    using the RandScaleIntensity transform from MONAI.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image",),
        change_cell_ratio: float = 0.4,
        scale_factors: tuple[float, float] | float = (0.0, 0.7),
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys (Sequence(str)): Keys in the input dictionary corresponding to the image.
                                  Default is ("image",).
            change_cell_ratio (float): Ratio of cells to apply the intensity scaling.
                                       For example, 0.4 means 40% of the cells will be transformed.
                                       Default is 0.4.
            scale_factors (tuple(float, float) | float): Factors used for random intensity scaling.
                                             Default is (0.0, 0.7).
            allow_missing_keys (bool): If True, missing keys in the input will be ignored.
                                       Default is False.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.change_cell_ratio: float = change_cell_ratio
        # Compose a random intensity scaling transform with 100% probability.
        self.randscale_intensity = Compose([RandScaleIntensity(prob=1.0, factors=scale_factors)])

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply a cell-wise intensity diversification transform to an input image.

        This function modifies the image by randomly selecting a subset of labeled cell regions 
        (per channel) and applying a random intensity scaling operation exclusively to those regions.
        The transformation is performed independently on each channel of the image.

        The steps are as follows:
        1. Extract the label image for all channels (expected shape: (C, H, W)).
        2. For each channel, determine the unique cell IDs, excluding the background (labeled as 0).
        3. Raise a ValueError if no unique objects are found in the current label channel.
        4. Compute the number of cells to modify based on the provided change_cell_ratio.
        5. Randomly select the corresponding cell IDs for intensity modification.
        6. Create a binary mask that highlights the selected cell regions.
        7. Separate the image channel into two parts: one that remains unchanged and one that is 
            subjected to random intensity scaling.
        8. Apply the random intensity scaling to the selected regions.
        9. Combine the unchanged and modified parts to update the image for that channel.

        Args:
            data (dict(str, np.ndarray)): A dictionary containing:
                - "image": The original image array.
                - "mask": The corresponding cell label image array.

        Returns:
            dict(str, np.ndarray): The updated dictionary with the "image" key modified after applying 
                                the intensity transformation.

        Raises:
            ValueError: If no unique cell objects are found in a label channel.
        """
        # Extract the label information for all channels.
        # The label array has dimensions (C, H, W), where C is the number of channels.
        label = data["mask"]  # shape: (C, H, W)

        # Process each channel independently.
        for c in range(label.shape[0]):
            # Extract the label and corresponding image channel for the current channel.
            channel_label = label[c]
            img_channel = data["image"][c]

            # Retrieve all unique cell IDs in the current channel.
            # Exclude the background (0) from these IDs.
            cell_ids = np.unique(channel_label)
            cell_ids = cell_ids[cell_ids > 0]

            # If there are no unique cell objects in this channel, raise an exception.
            if cell_ids.size == 0:
                logger.warning(f"No unique objects found in the label mask for channel {c}")
                continue

            # Determine the number of cells to modify using the change_cell_ratio.
            change_count = int(len(cell_ids) * self.change_cell_ratio)

            # Randomly select a subset of cell IDs for intensity modification.
            selected = np.random.choice(cell_ids, change_count, replace=False)

            # Create a binary mask for the current channel:
            # - Pixels corresponding to the selected cell IDs are set to 1.
            # - All other pixels are set to 0.
            mask_np = np.isin(channel_label, selected).astype(np.float32)

            # Convert mask to same dtype and device
            mask = torch.from_numpy(mask_np).to(dtype=torch.float32, device=channel_label.device)

            # Separate the image channel into two components:
            # 1. img_orig: The portion of the image that remains unchanged.
            # 2. img_changed: The portion that will have its intensity altered.
            img_orig = (1 - mask) * img_channel
            img_changed = mask * img_channel

            # Add a channel dimension for RandScaleIntensity: (1, H, W)
            img_changed = img_changed.unsqueeze(0)
            # Apply a random intensity scaling transformation to the selected regions.
            img_changed = self.randscale_intensity(img_changed)
            img_changed = img_changed.squeeze(0)  # type: ignore # back to shape (H, W)

            # Combine the unchanged and modified parts to update the image channel.
            data["image"][c] = img_orig + img_changed

        return data
