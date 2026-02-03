"""UNet architecture with ResNet-50 encoder for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connection.

    Upsample -> Concat with skip -> ConvBlock
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # Upsampling via transposed convolution
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        # Convolution block after concatenation
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Align sizes if needed (due to odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Channel-wise concatenation
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    """
    ResNet-50 encoder for UNet.

    Extracts feature maps at multiple scales.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Split into blocks for feature extraction
        # conv1: 3 -> 64, stride 2, 7x7 conv + BN + ReLU
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        # maxpool: stride 2
        self.maxpool = resnet.maxpool

        # Residual blocks
        self.layer1 = resnet.layer1  # 64 -> 256
        self.layer2 = resnet.layer2  # 256 -> 512
        self.layer3 = resnet.layer3  # 512 -> 1024
        self.layer4 = resnet.layer4  # 1024 -> 2048

        # Output channels for skip connections
        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass returning feature maps at each scale.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            List of feature maps [c1, c2, c3, c4, c5]
            c1: (B, 64, H/2, W/2)
            c2: (B, 256, H/4, W/4)
            c3: (B, 512, H/8, W/8)
            c4: (B, 1024, H/16, W/16)
            c5: (B, 2048, H/32, W/32)
        """
        # Initial convolution: H/2
        c1 = self.conv1(x)

        # Maxpool + layer1: H/4
        c2 = self.layer1(self.maxpool(c1))

        # Layer2: H/8
        c3 = self.layer2(c2)

        # Layer3: H/16
        c4 = self.layer3(c3)

        # Layer4: H/32
        c5 = self.layer4(c4)

        return [c1, c2, c3, c4, c5]


class UNet(nn.Module):
    """
    UNet with ResNet-50 encoder for binary segmentation.

    Architecture:
        Input (B, 3, H, W)
        -> ResNet Encoder (5 scales)
        -> Decoder with skip connections
        -> Output (B, 1, H, W) with Sigmoid
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_encoder: bool = True,
    ) -> None:
        """
        Initialize UNet.

        Args:
            num_classes: Number of output classes (1 for binary).
            pretrained_encoder: Whether to use pretrained ResNet weights.
        """
        super().__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained_encoder)
        enc_channels = self.encoder.out_channels  # [64, 256, 512, 1024, 2048]

        # Decoder blocks (from deep to shallow)
        # dec4: 2048 -> 1024, skip from c4 (1024)
        self.dec4 = DecoderBlock(enc_channels[4], enc_channels[3], 512)
        # dec3: 512 -> 256, skip from c3 (512)
        self.dec3 = DecoderBlock(512, enc_channels[2], 256)
        # dec2: 256 -> 128, skip from c2 (256)
        self.dec2 = DecoderBlock(256, enc_channels[1], 128)
        # dec1: 128 -> 64, skip from c1 (64)
        self.dec1 = DecoderBlock(128, enc_channels[0], 64)

        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Encoding
        c1, c2, c3, c4, c5 = self.encoder(x)

        # Decoding with skip connections
        d4 = self.dec4(c5, c4)  # H/16
        d3 = self.dec3(d4, c3)  # H/8
        d2 = self.dec2(d3, c2)  # H/4
        d1 = self.dec1(d2, c1)  # H/2

        # Final upsampling to original resolution
        out = self.final_up(d1)  # H
        out = self.final_conv(out)

        return out

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary mask.

        Args:
            x: Input tensor (B, 3, H, W)
            threshold: Threshold for binarization.

        Returns:
            Binary mask (B, 1, H, W)
        """
        with torch.inference_mode():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()
