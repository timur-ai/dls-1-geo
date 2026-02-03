"""GSD estimation model: Regression Tree CNN (Lee & Sull, 2019).

Based on: Lee, J., & Sull, S. (2019). Regression Tree CNN for Estimation of Ground
Sampling Distance Based on Floating-Point Representation. Remote Sensing, 11(19), 2276.

GSD = 20 × mantissa × 2^exponent (cm/pixel)
- exponent ∈ {0,1,2,3,4} — classification (5 classes)
- mantissa ∈ [0.75, 1.5] — regression
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class BinomialTreeLayer(nn.Module):
    """
    Binomial Tree Layer for GSD estimation (Lee & Sull, 2019).

    Architecture:
        - Level 0: 1 node (root) with feature map from CNN
        - Level l: (l+1) nodes with local classification among l classes
        - Node selection by L2 norm (Equation 11)
        - Mantissa head: two pointwise convs (64→1) before GAP
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 5,
        vector_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.vector_dim = vector_dim
        self.depth = num_classes

        # Root convolution: feature map → V-dim vector (level 0, 1 node)
        self.root_conv = nn.Conv2d(in_channels, vector_dim, kernel_size=1, bias=True)

        # Pointwise convolutions for each node at each level (except root)
        self.level_convs = nn.ModuleList()
        for level in range(1, self.depth):
            num_nodes = level + 1
            level_conv = nn.ModuleList(
                [
                    nn.Conv2d(vector_dim, vector_dim, kernel_size=1, bias=True)
                    for _ in range(num_nodes)
                ]
            )
            self.level_convs.append(level_conv)

        # Mantissa regression head: two pointwise convs
        self.mantissa_conv1 = nn.Conv2d(vector_dim, 64, kernel_size=1, bias=True)
        self.mantissa_conv2 = nn.Conv2d(64, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through binomial tree.

        Args:
            features: [B, C, H, W] feature map from backbone

        Returns:
            dict with keys:
                - class_logits: [B, num_classes] — L2 norms of leaf vectors
                - mantissa: [B] — predicted mantissa [0.75, 1.5]
                - selected_class: [B] — selected class (argmax by L2 norm)
        """
        b, _, h, w = features.shape

        # Level 0: root node
        root_feat = self.root_conv(features)
        current_level_feats = [root_feat]

        # Traverse tree levels (levels 1 to depth-1)
        for level_idx, level_conv in enumerate(self.level_convs):
            level = level_idx + 1
            num_nodes = level + 1
            next_level_feats = []

            for node_idx in range(num_nodes):
                parent_left = max(0, node_idx - 1)
                parent_right = min(node_idx, level - 1)

                if parent_left == parent_right:
                    parent_feat = current_level_feats[parent_left]
                else:
                    # Selection by L2 norm (Equation 11)
                    left_feat = current_level_feats[parent_left]
                    right_feat = current_level_feats[parent_right]

                    left_norm = left_feat.mean(dim=(2, 3)).norm(p=2, dim=1)
                    right_norm = right_feat.mean(dim=(2, 3)).norm(p=2, dim=1)

                    select_right = (right_norm > left_norm).float().view(b, 1, 1, 1)
                    parent_feat = (
                        left_feat * (1 - select_right) + right_feat * select_right
                    )

                node_feat = level_conv[node_idx](parent_feat)
                next_level_feats.append(node_feat)

            current_level_feats = next_level_feats

        # Leaf nodes — last level
        leaf_vectors_pooled = [feat.mean(dim=(2, 3)) for feat in current_level_feats]
        leaf_stack = torch.stack(leaf_vectors_pooled, dim=1)
        class_logits = torch.norm(leaf_stack, p=2, dim=2)

        # Select class with maximum L2 norm
        selected_class = torch.argmax(class_logits, dim=1)

        # Select feature map of chosen leaf for mantissa regression
        leaf_feats_stack = torch.stack(current_level_feats, dim=1)
        batch_indices = torch.arange(b, device=features.device)
        selected_feat = leaf_feats_stack[batch_indices, selected_class]

        # Normalize feature map by L2 norm
        feat_norm = torch.norm(selected_feat, p=2, dim=1, keepdim=True) + 1e-8
        normalized_feat = selected_feat / feat_norm

        # Mantissa regression via two pointwise convs, then GAP
        m1 = torch.relu(self.mantissa_conv1(normalized_feat))
        m2 = self.mantissa_conv2(m1)
        mantissa_raw = m2.mean(dim=(2, 3)).squeeze(-1)
        mantissa = torch.sigmoid(mantissa_raw) * 0.75 + 0.75  # [0.75, 1.5]

        return {
            "class_logits": class_logits,
            "mantissa": mantissa,
            "selected_class": selected_class,
        }


class RegressionTreeCNN(nn.Module):
    """
    Regression Tree CNN for GSD estimation (Lee & Sull, 2019).

    Architecture:
        - Backbone: ResNet-101 (pretrained on ImageNet)
        - Binomial Tree Layer: hierarchical classification + mantissa regression
        - Output: GSD = 20 × mantissa × 2^exponent (cm/pixel)
    """

    def __init__(
        self,
        backbone: str = "resnet101",
        num_exponent_classes: int = 5,
        vector_dim: int = 16,
        gsd_min: float = 15.0,
        gsd_max: float = 480.0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_exponent_classes
        self.gsd_min = gsd_min
        self.gsd_max = gsd_max

        # Backbone without avgpool and fc
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=[-1]
        )
        feature_dim = self.backbone.feature_info[-1]["num_chs"]

        # Binomial Tree Layer
        self.tree = BinomialTreeLayer(
            in_channels=feature_dim,
            num_classes=num_exponent_classes,
            vector_dim=vector_dim,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dict with keys:
                - gsd: Predicted GSD [B] (cm/pixel)
                - class_logits: L2 norms of leaf vectors [B, C]
                - mantissa: Predicted mantissa [B] in [0.75, 1.5]
                - selected_class: Selected class indices [B]
        """
        # Backbone: [B, 3, 256, 256] → [B, 2048, 8, 8]
        feat = self.backbone(x)[-1]

        # Binomial Tree Layer
        tree_out = self.tree(feat)

        # GSD = 20 × mantissa × 2^exponent
        gsd = 20.0 * tree_out["mantissa"] * (2.0 ** tree_out["selected_class"].float())

        return {
            "gsd": gsd,
            "class_logits": tree_out["class_logits"],
            "mantissa": tree_out["mantissa"],
            "selected_class": tree_out["selected_class"],
        }
