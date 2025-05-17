import numpy as np
import tifffile as tif
import skimage.io as io
from typing import Final, Sequence, Type

from monai.utils.enums import PostFix
from monai.utils.module import optional_import
from monai.utils.misc import ensure_tuple, ensure_tuple_rep
from monai.data.utils import is_supported_format
from monai.data.image_reader import ImageReader, NumpyReader
from monai.transforms import LoadImage, LoadImaged # type: ignore
from monai.config.type_definitions import DtypeLike, PathLike, KeysCollection


# Default value for metadata postfix
DEFAULT_POST_FIX = PostFix.meta()

# Try to import ITK library; if not available, has_itk will be False
itk, has_itk = optional_import("itk", allow_namespace_pkg=True)


__all__ = [
    "CustomLoadImage",         # Basic image loader
    "CustomLoadImaged",        # Dictionary-based image loader
    "CustomLoadImageD",        # Dictionary-based image loader
    "CustomLoadImageDict",     # Dictionary-based image loader
    "SUPPORTED_IMAGE_FORMATS"
]

SUPPORTED_IMAGE_FORMATS: Final[Sequence[str]] = ["tif", "tiff", "png", "jpg", "bmp", "jpeg"]


class CustomLoadImage(LoadImage):
    """
    Class for loading one or multiple images from a given path.
    
    If a reader is not specified, the appropriate file reading method is automatically chosen
    based on the file extension. Priority:
      - Reader passed by the user at runtime.
      - Reader specified in the constructor.
      - Registered readers (from last to first).
      - Standard readers for different formats (e.g., NibabelReader for nii, PILReader for png/jpg, etc.).
      
    [Note] Here, the original ITKReader is replaced by the universal reader UniversalImageReader.
    """
    def __init__(
        self,
        reader: ImageReader | Type[ImageReader] | str | None = None,
        image_only: bool = False,
        dtype: DtypeLike = np.float32,
        ensure_channel_first: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            reader=reader,
            image_only=image_only,
            dtype=dtype,
            ensure_channel_first=ensure_channel_first,
            *args, **kwargs
        )
        # Clear the list of registered readers
        self.readers = []
        # Register the universal reader that handles TIFF, PNG, JPG, BMP, etc.
        self.register(UniversalImageReader(*args, **kwargs))


class CustomLoadImaged(LoadImaged):
    """
    Dictionary-based image loader.
    
    Wraps image loading with CustomLoadImage and allows processing of data represented as a dictionary,
    where keys point to file paths.
    """
    def __init__(
        self,
        keys: KeysCollection,
        reader: Type[ImageReader] | str | None = None,
        dtype: DtypeLike = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            keys=keys,
            reader=reader,
            dtype=dtype,
            meta_keys=meta_keys,
            meta_key_postfix=meta_key_postfix,
            overwriting=overwriting,
            image_only=image_only,
            ensure_channel_first=ensure_channel_first,
            simple_keys=simple_keys,
            allow_missing_keys=allow_missing_keys,
            *args,
            **kwargs,
        )
        # Assign the custom image loader
        self._loader = CustomLoadImage(
            reader=reader,
            image_only=image_only,
            dtype=dtype,
            ensure_channel_first=ensure_channel_first,
            *args, **kwargs
        )
        # Ensure that meta_key_postfix is a string
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a string, but got {type(meta_key_postfix).__name__}."
            )
        # If meta_keys are not provided, create a tuple of None for each key
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        # Check that the number of meta_keys matches the number of keys
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys must have the same length as keys.")
        # Assign each key its corresponding metadata postfix
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


class UniversalImageReader(NumpyReader):
    """
    Universal image reader for TIFF, PNG, JPG, BMP, etc.
    
    Uses:
      - tifffile for reading TIFF files.
      - ITK (if available) for reading other formats.
      - skimage.io for reading if the previous methods fail.
      
    The image is loaded with its original number of channels (layers) without forced modifications
    (e.g., repeating or cropping channels).
    """
    def __init__(
        self, channel_dim: int | None = None, **kwargs,
    ) -> None:
        super().__init__(channel_dim=channel_dim, **kwargs)
        self.kwargs = kwargs
        self.channel_dim = channel_dim

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Check if the file format is supported for reading.
        
        Supported extensions: tif, tiff, png, jpg, bmp, jpeg.
        """
        return has_itk or is_supported_format(filename, SUPPORTED_IMAGE_FORMATS)

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image(s) from the given path.
        
        Arguments:
          data: A file path or a sequence of file paths.
          kwargs: Additional parameters for reading.
        
        Returns:
          A single image or a list of images depending on the number of paths provided.
        """
        images: list[np.ndarray] = []  # List to store the loaded images

        # Convert data to a tuple to support multiple files
        filenames: Sequence[PathLike] = ensure_tuple(data)
        # Merge parameters provided in the constructor and the read() method
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)

        for name in filenames:
            # Convert file name to string
            name = f"{name}"
            # If the file has a .tif or .tiff extension (case-insensitive), use tifffile for reading
            if name.lower().endswith((".tif", ".tiff")):
                img_array = tif.imread(name)
            else:
                # Attempt to read the image using ITK (if available)
                try:
                    img_itk = itk.imread(name, **kwargs_)
                    img_array = itk.array_view_from_image(img_itk, keep_axes=False)
                except Exception:
                    # If ITK fails, use skimage.io for reading
                    img_array = io.imread(name)

            # Check the number of dimensions (axes) of the loaded image
            if img_array.ndim == 2:
                # If the image is 2D (height, width), add a new axis at the end to represent the channel
                img_array = np.expand_dims(img_array, axis=-1)

            images.append(img_array)

        # Return a single image if only one file was provided, otherwise return a list of images
        return images if len(filenames) > 1 else images[0]



CustomLoadImageD = CustomLoadImageDict = CustomLoadImaged