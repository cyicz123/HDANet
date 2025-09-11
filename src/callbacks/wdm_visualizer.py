import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.utils import make_grid
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class WDMVisualizer(Callback):
    """
    Callback to visualize WDM and images from a batch on TensorBoard.
    Overlays WDM heatmap on background images specified in the config.
    """

    def __init__(self, background_image_path: str = None, alpha: float = 0.5, images_num: int = 2):
        super().__init__()
        self.images_num = images_num  # Kept for compatibility, might be unused
        self.alpha = alpha
        self.background_image = None

        if background_image_path:
            transform = transforms.Compose([
                transforms.Resize((140, 80)),  # Resize to a common size
                transforms.ToTensor()          # Convert PIL image to tensor and scale to [0, 1]
            ])
            try:
                img = Image.open(background_image_path).convert('RGB')
                self.background_image = transform(img)
            except FileNotFoundError:
                print(f"Warning: Background image not found at {background_image_path}, skipping.")

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        # Only run on the first batch and if background images are loaded
        if batch_idx == 0 and self.background_image is not None:
            logger = trainer.logger
            if isinstance(logger, TensorBoardLogger):
                batch_size = batch[0][0].shape[0]
                rand_idx = random.randint(0, batch_size - 1)
                wdm = batch[0][0][rand_idx].cpu()  # 随机选取一个样本的WDM,并放置到CPU
                images = batch[0][1][rand_idx].cpu()  # 随机选取同一个样本的图片,并放置到CPU

                # Convert WDM to a heatmap
                wdm_np = wdm.squeeze().numpy()
                cmap = plt.get_cmap('jet')
                wdm_heatmap_rgba = cmap(wdm_np)
                wdm_heatmap_rgb = np.delete(wdm_heatmap_rgba, 3, 2)
                wdm_heatmap_tensor = torch.from_numpy(wdm_heatmap_rgb).permute(2, 0, 1)

                # Log the original WDM heatmap
                logger.experiment.add_image(
                    "WDM Heatmap",
                    wdm_heatmap_tensor,
                    global_step=trainer.global_step,
                )

                split_images = torch.split(images, 3, dim=0)
                img_grid = make_grid(list(split_images), nrow=self.images_num)
                logger.experiment.add_image(
                    "Input Images",
                    img_grid,
                    global_step=trainer.global_step
                )

                # Resize heatmap to match background image size
                resized_heatmap = F.interpolate(
                    wdm_heatmap_tensor.unsqueeze(0),
                    size=(140, 80),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

        
                overlay = self.alpha * self.background_image + (1 - self.alpha) * resized_heatmap

                # Create a grid of the overlay images
                logger.experiment.add_image(
                    "WDM Overlay",
                    overlay,
                    global_step=trainer.global_step
                )
