import os
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import logging

from src.models.components.detection_lenet import DetectionLeNet
from src.models.components.detection_resnet import DetectionResNet34
from src.utils.pylogger import RankedLogger


class FusionNet(nn.Module):
    """多模态融合模型，融合WiFi和视频特征进行人数估计"""

    def __init__(
        self,
        wifi_config: Dict[str, Any],
        video_config: Dict[str, Any],
        fusion_config: Dict[str, Any],
    ):
        """初始化

        Args:
            wifi_config: WiFi模型配置
            video_config: 视频模型配置
            fusion_config: 融合模型配置
        """
        super().__init__()
        self.log = RankedLogger(self.__class__.__name__)

        # 融合参数
        self.fusion_video_scheme = fusion_config.get("fusion_video_scheme", 5)
        self.video_list = fusion_config.get("video_indices", [3, 5, 4, 2, 1])
        self.output_size = fusion_config.get("output_size", 1)  # Defaulting to 1 for single value regression

        # 初始化WiFi处理模型
        self.lenet = DetectionLeNet(use_attention=wifi_config.get("use_attention", True))

        # 加载WiFi预训练模型
        wifi_pretrained = wifi_config.get("pretrained_path")
        if wifi_pretrained and os.path.exists(wifi_pretrained):
            self.log.log(logging.INFO, f"Loading pretrained WiFiViT from: {wifi_pretrained}")
            try:
                self.lenet.load_state_dict(torch.load(wifi_pretrained, map_location="cpu"), strict=False)
                self.log.log(logging.INFO, "WiFiViT pretrained weights loaded successfully.")
            except Exception as e:
                self.log.log(logging.ERROR, f"Error loading WiFiViT pretrained weights: {e}")
                raise e

            # 冻结WiFi特征提取器
            if wifi_config.get("freeze", True):
                for param in self.lenet.parameters():
                    param.requires_grad = False

        # 初始化视频处理模型
        self.resnet_models = nn.ModuleList()
        resnet_models_list = video_config.get("pretrained_list")
        if resnet_models_list and len(resnet_models_list) > 0:
            for i in range(self.fusion_video_scheme):
                resnet = DetectionResNet34(
                    use_attention=video_config.get("use_attention", True),
                    use_pretrained=video_config.get("use_pretrained", True),
                )
                # Find the correct pretrained path for the video index
                video_idx = self.video_list[i]
                pretrained_path = None
                for p in resnet_models_list:
                    if f"resnet_v{video_idx}" in p:
                        pretrained_path = p
                        break
                
                if pretrained_path and os.path.exists(pretrained_path):
                    self.log.log(logging.INFO, f"Loading pretrained video model for video {video_idx} from: {pretrained_path}")
                    try:
                        resnet.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
                        self.log.log(logging.INFO, "Video model pretrained weights loaded successfully.")
                    except Exception as e:
                        self.log.log(logging.ERROR, f"Error loading video model pretrained weights: {e}")
                else:
                    self.log.log(logging.WARNING, f"Pretrained video model not found for video index {video_idx}, using random weights.")
                
                # 冻结视频特征提取器
                if video_config.get("freeze", True):
                    for param in resnet.parameters():
                        param.requires_grad = False
                self.resnet_models.append(resnet)
        else:
            self.log.log(logging.ERROR, "No pretrained video models provided.")
            raise ValueError("No pretrained video models provided.")

        # 初始化融合层
        self.fcvideo = nn.Linear(512, 700)
        self.tsd = nn.TransformerDecoderLayer(d_model=700, nhead=10, dim_feedforward=2048, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Recalculate input features for fc1
        # wifi_f: (B, 16, 700)
        # After concat with one video: (B, 16 + 16 + 1, 700) = (B, 33, 700)
        # After all videos: (B, 16 + fusion_video_scheme * (16 + 1), 700)
        fc1_input_features = (16 + self.fusion_video_scheme * 17) * 700
        self.fc1 = nn.Linear(fc1_input_features, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.output_size)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入元组，包含WiFi数据和视频数据

        Returns:
            模型输出
        """
        wifi_data, video_data = x

        # 获取WiFi特征
        wifi_f = self.lenet.lenet(wifi_data)  # output: [batch_size, 16, 700]

        # 初始化融合特征
        fusion_fs = wifi_f

        # 处理视频特征并融合
        for i in range(self.fusion_video_scheme):
            # 计算当前视频的通道索引
            start_idx = i * 3
            end_idx = start_idx + 3

            resnet = self.resnet_models[i]

            # 处理当前视频帧
            current_video_data = video_data[:, start_idx:end_idx, :, :]
            video_feature = resnet.features(current_video_data)
            video_feature = resnet.tse(video_feature.flatten(2).transpose(1, 2))  # [B, C, H, W] -> [B, H*W, C] -> [B, H*W, 512]
            video_feature = self.fcvideo(video_feature)

            # 使用Transformer解码器处理特征
            decoded_feature = self.tsd(wifi_f, video_feature)
            fusion_fs = torch.cat([fusion_fs, decoded_feature], 1)

            # 添加池化后的特征
            pooled_feature = self.avgpool(video_feature.transpose(1, 2)).transpose(1, 2)
            fusion_fs = torch.cat([fusion_fs, pooled_feature], 1)

        # 最终预测
        fusion_fs = fusion_fs.flatten(1)
        fusion_fs = F.relu(self.fc1(fusion_fs))
        fusion_fs = F.relu(self.fc2(fusion_fs))
        fusion_fs = self.fc3(fusion_fs)

        return fusion_fs
