from torch import nn
from torchvision import models


class DetectionResNet34(nn.Module):
    """使用ResNet34模型进行人数估计，支持配置是否使用预训练模型和自注意力"""

    def __init__(
        self,
        output_size: int = 5,
        use_pretrained: bool = True,
        use_attention: bool = True,
    ) -> None:
        """初始化

        Args:
            output_size: The number of output features.
            use_pretrained: 是否使用预训练模型.
            use_attention: 是否使用自注意力编码器.
        """
        super().__init__()
        self.output_size = output_size
        self.use_pretrained = use_pretrained
        self.use_attention = use_attention

        if self.use_pretrained:
            weights = models.ResNet34_Weights.DEFAULT
        else:
            weights = None
        self.backbone = models.resnet34(weights=weights)

        if not self.use_pretrained:
            self._initialize_resnet_weights()

        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        if self.use_attention:
            self.tse = nn.TransformerEncoderLayer(
                d_model=512, nhead=16, dim_feedforward=2048, batch_first=True
            )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, self.output_size)
        self._initialize_layers()

    def _initialize_resnet_weights(self) -> None:
        """使用凯明初始化ResNet网络权重"""
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_layers(self) -> None:
        """初始化模型层的参数"""
        for m in [self.fc1]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        if self.use_attention:
            for name, param in self.tse.named_parameters():
                if "weight" in name:
                    if "norm" in name:
                        nn.init.ones_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        """前向传播"""
        features = self.features(x)
        features_flat = features.flatten(2).transpose(1, 2)
        if self.use_attention:
            features_flat = self.tse(features_flat)
        features = self.avgpool(features_flat.transpose(1, 2))
        flat_features = features.flatten(1)
        out = self.fc1(flat_features)
        return out
