import torch
import numpy as np
from typing import Hashable, List, Sequence, Optional, Tuple

from monai.utils.misc import fall_back_tuple
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms import Randomizable, RandCropd, Crop  # type: ignore

from core.logger import get_logger

logger = get_logger(__name__)


def _compute_multilabel_bbox(
    mask: np.ndarray
) -> Optional[Tuple[List[int], List[int], List[int], List[int]]]:
    """
    Compute per-channel bounding-box constraints and return lists of limits for each axis.

    Args:
        mask: multi-channel instance mask of shape (C, H, W).

    Returns:
        A tuple of four lists:
            - top_mins: list of r_max for each non-empty channel
            - top_maxs: list of r_min for each non-empty channel
            - left_mins: list of c_max for each non-empty channel
            - left_maxs: list of c_min for each non-empty channel
        Or None if mask contains no positive labels.
    """
    channels, rows, cols = np.nonzero(mask)
    if channels.size == 0:
        return None

    top_mins: List[int] = []
    top_maxs: List[int] = []
    left_mins: List[int] = []
    left_maxs: List[int] = []
    C = mask.shape[0]
    for ch in range(C):
        rs, cs = np.nonzero(mask[ch])
        if rs.size == 0:
            continue
        r_min, r_max = int(rs.min()), int(rs.max())
        c_min, c_max = int(cs.min()), int(cs.max())
        # For each channel, record the row/col extents
        top_mins.append(r_max)
        top_maxs.append(r_min)
        left_mins.append(c_max)
        left_maxs.append(c_min)

    return top_mins, top_maxs, left_mins, left_maxs


class SpatialCropAllClasses(Randomizable, Crop):
    """
    Cropper for multi-label instance masks and images: ensures each label-channel's
    instances lie within the crop if possible.

    Must be called on a mask tensor first to compute the crop, then on the image.

    Args:
        roi_size: desired crop size (height, width).
        num_candidates: fallback samples when no single crop fits all instances.
        lazy: defer actual cropping.
    """
    def __init__(
        self,
        roi_size: Sequence[int],
        num_candidates: int = 10,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self.roi_size = tuple(roi_size)
        self.num_candidates = num_candidates
        self._slices: Optional[Tuple[slice, ...]] = None

    def randomize(self, img_size: Sequence[int]) -> None: # type: ignore
        """
        Choose crop offsets so that each non-empty channel is included if possible.
        """
        height, width = img_size
        crop_h, crop_w = self.roi_size
        max_top = max(0, height - crop_h)
        max_left = max(0, width - crop_w)

        # Compute per-channel bbox constraints
        mask = self._img
        bboxes = _compute_multilabel_bbox(mask)
        if bboxes is None:
            # no labels: random patch using MONAI utils
            logger.warning("No labels found; using random patch.")
            # determine actual patch size (fallback)
            self._size = fall_back_tuple(self.roi_size, img_size)
            # compute valid size for random patch
            valid_size = get_valid_patch_size(img_size, self._size)
            # directly get random patch slices
            self._slices = get_random_patch(img_size, valid_size, self.R)
            return
        else:
            top_mins, top_maxs, left_mins, left_maxs = bboxes
            # Convert to allowable windows
            # top_min_global = max(r_max - crop_h +1 for each channel)
            global_top_min = max(0, max(r_max - crop_h + 1 for r_max in top_mins))
            # top_max_global = min(r_min for each channel)
            global_top_max = min(min(top_maxs), max_top)
            # same for left
            global_left_min = max(0, max(c_max - crop_w + 1 for c_max in left_mins))
            global_left_max = min(min(left_maxs), max_left)

            if global_top_min <= global_top_max and global_left_min <= global_left_max:
                # there is a window covering all channels fully
                top = self.R.randint(global_top_min, global_top_max + 1)
                left = self.R.randint(global_left_min, global_left_max + 1)
            else:
                # fallback: sample candidates to maximize channel coverage
                logger.warning(
                    f"Cannot fit all instances; sampling {self.num_candidates} candidates."
                )
                best_cover = -1
                best_top = best_left = 0
                C = mask.shape[0]
                for _ in range(self.num_candidates):
                    cand_top = self.R.randint(0, max_top + 1)
                    cand_left = self.R.randint(0, max_left + 1)
                    window = mask[:, cand_top : cand_top + crop_h, cand_left : cand_left + crop_w]
                    cover = sum(int(window[ch].any()) for ch in range(C))
                    if cover > best_cover:
                        best_cover = cover
                        best_top, best_left = cand_top, cand_left
                logger.info(f"Selected crop covering {best_cover}/{C} channels.")
                top, left = best_top, best_left

        # store slices for use on both mask and image
        self._slices = (
            slice(None),
            slice(top, top + crop_h),
            slice(left, left + crop_w),
        )

    def __call__(self, img: torch.Tensor, lazy: Optional[bool] = None) -> torch.Tensor: # type: ignore
        """
        On first call (mask), computes crop. On subsequent (image), just applies.
        Raises if mask not provided first.
        """
        # Determine tensor shape
        img_size = (
            img.peek_pending_shape()[1:]
            if isinstance(img, MetaTensor)
            else img.shape[1:]
        )
        # First call must be mask to compute slices
        if self._slices is None:
            if not torch.is_floating_point(img) and img.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
                # assume integer mask
                self._img = img.cpu().numpy()
                self.randomize(img_size)
            else:
                raise RuntimeError(
                    "Mask tensor must be passed first for computing crop bounds."
                )
        # Now apply stored slice
        if self._slices is None:
            raise RuntimeError("Crop slices not computed; call on mask first.")
        lazy_exec = self.lazy if lazy is None else lazy
        return super().__call__(img=img, slices=self._slices, lazy=lazy_exec)


class RandSpatialCropAllClassesd(RandCropd):
    """
    Dict-based wrapper: applies SpatialCropAllClasses to mask then image.
    Requires mask present or raises.
    """
    def __init__(
        self,
        keys: Sequence,
        roi_size: Sequence[int],
        num_candidates: int = 10,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ):
        cropper = SpatialCropAllClasses(
            roi_size=roi_size,
            num_candidates=num_candidates,
            lazy=lazy,
        )
        super().__init__(
            keys=keys,
            cropper=cropper,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
        )
