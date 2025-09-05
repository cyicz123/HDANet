import torch.nn.functional as F
from torch import nn


class LeNet(nn.Module):
    """LeNet模型，用于处理WiFi数据"""

    def __init__(self, use_attention: bool = True) -> None:
        """初始化

        Args:
            use_attention: 是否使用自注意力编码器
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.use_attention = use_attention
        if self.use_attention:
            self.tse = nn.TransformerEncoderLayer(
                d_model=700, nhead=5, dim_feedforward=512, batch_first=True
            )

    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.flatten(2)

        if self.use_attention:
            x = self.tse(x)

        return x


class DetectionLeNet(nn.Module):
    """LeNet模型，增加预测头用于人数估计"""

    def __init__(self, output_size: int = 5, use_attention: bool = True) -> None:
        """初始化

        Args:
            output_size: The number of output features.
            use_attention: 是否使用自注意力模块.
        """
        super().__init__()
        self.output_size = output_size
        self.use_attention = use_attention
        self.lenet = LeNet(use_attention=self.use_attention)

        if self.use_attention:
            self.fc1 = nn.Linear(16 * 700, 1024)
        else:
            self.fc1 = nn.Linear(16 * 20 * 35, 1024)

        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.output_size)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        features = self.lenet(x)
        flat_features = features.flatten(1)
        out = F.relu(self.fc1(flat_features))
        out = F.relu(self.fc2(out))
        predictions = self.fc3(out)
        return predictions
