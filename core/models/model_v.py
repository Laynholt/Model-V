from typing import List, Optional

import torch
import torch.nn as nn

from segmentation_models_pytorch import MAnet
from segmentation_models_pytorch.base.modules import Activation

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelV"]


class ModelVParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    encoder_name: str = "mit_b5"                               # Default encoder
    encoder_weights: Optional[str] = "imagenet"                # Pre-trained weights
    decoder_channels: List[int] = [1024, 512, 256, 128, 64]    # Decoder configuration
    decoder_pab_channels: int = 256                            # Decoder Pyramid Attention Block channels
    in_channels: int = 3                                       # Number of input channels
    out_classes: int = 1                                       # Number of output classes
    
    def asdict(self):
        """
        Returns a dictionary of valid parameters for `nn.ModelV`.

        Returns:
            Dict[str, Any]: Dictionary of parameters for nn.ModelV.
        """
        loss_kwargs = self.model_dump()
        return {k: v for k, v in loss_kwargs.items() if v is not None}  # Remove None values


class ModelV(MAnet):
    """ModelV model"""

    def __init__(self, params: ModelVParams) -> None:
        # Initialize the MAnet model with provided parameters
        super().__init__(**params.asdict())
        
        self.num_classes = params.out_classes

        # Remove the default segmentation head as it's not used in this architecture
        self.segmentation_head = None

        # Modify all activation functions in the encoder and decoder from ReLU to Mish
        _convert_activations(self.encoder, nn.ReLU, nn.Mish(inplace=True))
        _convert_activations(self.decoder, nn.ReLU, nn.Mish(inplace=True))

        # Add custom segmentation heads for different segmentation tasks
        self.cellprob_head = DeepSegmentationHead(
            in_channels=params.decoder_channels[-1], out_channels=params.out_classes
        )
        self.gradflow_head = DeepSegmentationHead(
            in_channels=params.decoder_channels[-1], out_channels=2 * params.out_classes
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure the input shape is correct
        self.check_input_shape(x)

        # Encode the input and then decode it
        features = self.encoder(x)
        decoder_output = self.decoder(features)

        # Generate masks for cell probability and gradient flows
        cellprob_mask = self.cellprob_head(decoder_output)
        gradflow_mask = self.gradflow_head(decoder_output)
        
        # Concatenate the masks for output
        masks = torch.cat((gradflow_mask, cellprob_mask), dim=1)

        return masks


class DeepSegmentationHead(nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[str] = None,
        upsampling: int = 1,
    ) -> None:
        # Define a sequence of layers for the segmentation head
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity(),
            Activation(activation) if activation else nn.Identity(),
        ]
        super().__init__(*layers)


def _convert_activations(module: nn.Module, from_activation: type, to_activation: nn.Module) -> None:
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)
