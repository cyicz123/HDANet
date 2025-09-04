import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.io as scio
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class FusionDataset(Dataset):
    """多模态融合数据集，用于WiFi和视频融合人数估计任务"""

    def __init__(
        self,
        data_dir: str,
        index_file: str,
        wifi_file: str,
        label_file: str,
        img_dir: str,
        fusion_video_scheme: int,
        video_list: List[int],
        video_type_no: int,
        video_save_names: List[str],
        is_training: bool = True,
    ) -> None:
        """初始化
        
        Args:
            data_dir: 数据目录.
            index_file: 索引文件名.
            wifi_file: WiFi数据文件名.
            label_file: 标签文件名.
            img_dir: 图像数据目录.
            fusion_video_scheme: 融合视频的数量.
            video_list: 视频列表.
            video_type_no: 视频类型编号.
            video_save_names: 视频保存名称列表.
            is_training: 是否为训练集.
        """
        self.data_dir = data_dir
        self.index_file = index_file
        self.wifi_file = wifi_file
        self.label_file = label_file
        self.img_dir = img_dir
        self.fusion_video_scheme = fusion_video_scheme
        self.video_list = video_list
        self.video_type_no = video_type_no
        self.video_save_names = video_save_names
        self.is_training = is_training


        self._load_data()

    def _load_data(self) -> None:
        """加载数据"""
        index_path = os.path.join(self.data_dir, self.index_file)
        with h5py.File(index_path, "r") as allidxs:
            trainidx = np.transpose(allidxs["trainidx"])
            trainidx = trainidx.astype(np.int64).squeeze()
            testidx = np.transpose(allidxs["testidx"])
            testidx = testidx.astype(np.int64).squeeze()

        # 注意训练集和测试集默认有交换，此处不动即可
        trainidx, testidx = testidx, trainidx

        wifi_path = os.path.join(self.data_dir, self.wifi_file)
        with h5py.File(wifi_path, "r") as data_file:
            data = np.transpose(data_file["des_maps"])
            data = data.astype(np.float32)
            data = np.expand_dims(data, axis=1)

        idx = trainidx if self.is_training else testidx
        self.wifi_data = data[idx]

        img_data_dir = os.path.join(self.data_dir, self.img_dir)
        self.video_data = None

        for i in range(self.fusion_video_scheme):
            video_idx = self.video_list[i]
            suffix = "_train.pt" if self.is_training else "_test.pt"
            video_path = os.path.join(
                img_data_dir,
                f"{self.video_save_names[self.video_type_no - 1]}_video{video_idx}_80x160{suffix}",
            )
            video = torch.load(video_path, map_location=torch.device('cpu'), weights_only=False)
            if self.video_data is None:
                self.video_data = video
            else:
                self.video_data = np.concatenate((self.video_data, video), axis=1)

        label_path = os.path.join(self.data_dir, self.label_file)
        labels = scio.loadmat(label_path)
        labels = labels["gt_counts"].astype(np.int64)
        self.labels = labels[idx]



    def __len__(self) -> int:
        """获取数据集长度"""
        return len(self.wifi_data)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """获取数据项"""
        wdm = self.wifi_data[idx]
        img = self.video_data[idx]
        target = self.labels[idx]
        return (
            torch.from_numpy(wdm),
            torch.from_numpy(img.copy()),
        ), torch.tensor(target, dtype=torch.float32)


class FusionDataModule(LightningDataModule):
    """`LightningDataModule` for the Fusion dataset."""

    def __init__(
        self,
        data_dir: str = "data/adca",
        index_file: str = "trainidx15rand.mat",
        wifi_file: str = "des_maps_wifi_fz15_sigma2_bs1_sw60.mat",
        label_file: str = "gt_counts.mat",
        img_dir: str = "imgdata",
        fusion_video_scheme: int = 1,
        video_list: List[int] = [3, 5, 4, 2, 1],
        video_type_no: int = 3,
        video_save_names: List[str] = ["orig", "roi", "crop", "fg", "fgm"],
        train_val_split: Tuple[float, float] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `FusionDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Check if data exists."""
        # You can add download logic here if needed
        if not os.path.isdir(self.hparams.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.hparams.data_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        if self.data_train and self.data_val and self.data_test:
            return

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        dataset_kwargs = {
            "data_dir": self.hparams.data_dir,
            "index_file": self.hparams.index_file,
            "wifi_file": self.hparams.wifi_file,
            "label_file": self.hparams.label_file,
            "img_dir": self.hparams.img_dir,
            "fusion_video_scheme": self.hparams.fusion_video_scheme,
            "video_list": self.hparams.video_list,
            "video_type_no": self.hparams.video_type_no,
            "video_save_names": self.hparams.video_save_names,
        }

        self.data_train = FusionDataset(**dataset_kwargs, is_training=True)
        self.data_val = FusionDataset(**dataset_kwargs, is_training=False)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

