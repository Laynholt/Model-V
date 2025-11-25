import torch
from torch import nn

from typing import Any, Callable
from segmentation_models_pytorch import MAnet
from segmentation_models_pytorch.base.modules import Activation

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelV"]


class ModelVParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    encoder_name: str = "mit_b5"                               # Default encoder
    encoder_weights: str | None = "imagenet"                   # Pre-trained weights
    decoder_channels: list[int] = [1024, 512, 256, 128, 64]    # Decoder configuration
    decoder_pab_channels: int = 256                            # Decoder Pyramid Attention Block channels
    in_channels: int = 3                                       # Number of input channels
    out_classes: int = 1                                       # Number of output classes
    
    prefer_gn: bool = True              # Prefer GroupNorm for small batches; falls back to BatchNorm if prefer_gn=False
    upsample: int = 1                   # Upsampling factor for heads (keep 1 if decoder outputs final spatial size)
    zero_init_heads: bool = False       # Apply zero init to final conv in heads (stabilizes early training)
    
    def asdict(self) -> dict[str, Any]:
        """
        Returns a dictionary of valid parameters for `nn.ModelV`.

        Returns:
            dict(str, Any): Dictionary of parameters for nn.ModelV.
        """
        loss_kwargs = self.model_dump()
        return {k: v for k, v in loss_kwargs.items() if v is not None}  # Remove None values


class ModelV(MAnet):
    """ModelV model"""

    def __init__(self, params: ModelVParams) -> None:
        # Initialize the MAnet model with provided parameters
        super().__init__(**params.asdict())
        
        self.num_classes = params.out_classes
        in_ch: int = params.decoder_channels[-1]

        # Remove the default segmentation head as it's not used in this architecture
        self.segmentation_head = None

        # Modify all activation functions in the encoder and decoder from ReLU to Mish
        _convert_activations(self.encoder, nn.ReLU, lambda: nn.Mish(inplace=True))
        _convert_activations(self.decoder, nn.ReLU, lambda: nn.Mish(inplace=True))

        self.prefer_gn: bool = params.prefer_gn
        self.upsample: int = params.upsample
        self.zero_init_heads: bool = params.zero_init_heads

        # Add custom segmentation heads for different segmentation tasks
        # Per-class logits head (C channels)
        self.logits_head = DeepSegmentationHead(
            in_channels=in_ch,
            out_channels=self.num_classes,
            kernel_size=3,
            activation=None,                # keep logits raw; apply loss with logits
            upsampling=self.upsample,
            prefer_gn=self.prefer_gn,
        )
        
        # Flow head per-class (2*C)
        flow_out = 2 * self.num_classes
        self.gradflow_head = DeepSegmentationHead(
            in_channels=in_ch,
            out_channels=flow_out,
            kernel_size=3,
            activation=None,                # we'll apply tanh explicitly (if enabled)
            upsampling=self.upsample,
            prefer_gn=self.prefer_gn,
        )
        
        if self.zero_init_heads:
            zero_init_last_conv(self.logits_head)
            zero_init_last_conv(self.gradflow_head)


    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Dict with:
              - "flow": (B, 2*C, H, W)  per-class flow;
              - "logits": (B, C, H, W)  per-class logits.
        """
        # Ensure the input shape is correct
        self.check_input_shape(x)

        # Encode the input and then decode it
        features = self.encoder(x)
        decoder_output = self.decoder(features)

        # Generate masks for cell probability and gradient flows
        logits = self.logits_head(decoder_output)        # (B, C, H, W)
        flow = self.gradflow_head(decoder_output)        # (B, 2*C, H, W)
        
        return {"flow": flow, "logits": logits}


class DeepSegmentationHead(nn.Sequential):
    """
    A robust segmentation head block:
      Conv(bias=False) -> Norm -> Mish -> Conv -> (Upsample) -> (Activation?)
    Notes:
      * Using bias=False on the first conv since normalization follows it.
      * GroupNorm is preferred for small batch sizes; fall back to BatchNorm2d.
      * The 'mid' width is clamped by a minimal value to avoid too narrow bottlenecks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str | None = None,
        upsampling: int = 1,
        prefer_gn: bool = True,
        min_mid: int = 8,
        reduce_ratio: float = 0.5,
    ) -> None:
        mid = compute_mid(in_channels, r=reduce_ratio, min_mid=min_mid)
        norm_layer = make_norm(mid, prefer_gn=prefer_gn)

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                mid,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            norm_layer,
            nn.Mish(inplace=True),
            nn.Conv2d(
                mid,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,  # final conv may keep bias; can be zero-initialized
            ),
            nn.Upsample(scale_factor=upsampling, mode="bilinear", align_corners=False)
            if upsampling > 1 else nn.Identity(),
            Activation(activation) if activation else nn.Identity(),
        ]
        super().__init__(*layers)


def _convert_activations(module: nn.Module, from_activation: type, to_activation: Callable) -> None:
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation())
        else:
            _convert_activations(child, from_activation, to_activation)


def make_norm(num_channels: int, prefer_gn: bool = True) -> nn.Module:
    """
    Return a normalization layer resilient to small batch sizes.
    GroupNorm is independent of batch dimension and thus stable when B is small.
    """
    if prefer_gn:
        for g in (32, 16, 8, 4, 2, 1):
            if num_channels % g == 0:
                return nn.GroupNorm(g, num_channels)
        # Fallback: 1 group ~ LayerNorm across channels (per-spatial)
        return nn.GroupNorm(1, num_channels)
    else:
        return nn.BatchNorm2d(num_channels)


def compute_mid(
    in_ch: int,
    r: float = 0.5,
    min_mid: int = 8,
    groups_hint: tuple[int, ...] = (32, 16, 8, 4, 2)
) -> int:
    """
    Compute intermediate channel width for the head.
    Ensures a minimum width and (optionally) tries to align to a group size.
    """
    raw = max(min_mid, int(round(in_ch * r)))
    for g in groups_hint:
        if raw % g == 0:
            return raw
    return raw


def zero_init_last_conv(module: nn.Module) -> None:
    """
    Zero-initialize the last Conv2d in a head to make its initial contribution neutral.
    This often stabilizes early training for multi-task heads.
    """
    last_conv: nn.Conv2d | None = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)