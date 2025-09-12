from typing import Any, Dict, Tuple
from functools import partial
import torch
from lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.metric import Metric


class FusionLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating MAE and MSE
        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        
        # for averaging loss across batches
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()

        # new metrics logic for validation
        self.is_multicam = hasattr(self.net, "video_list") 
        if self.is_multicam:
            self.video_list = self.net.video_list
            self.num_cameras = self.net.output_size
            
            val_metrics = {}
            for i in range(self.num_cameras):
                cam_id = self.video_list[i]
                val_metrics[f"c{cam_id}_mae"] = MeanAbsoluteError()
                val_metrics[f"c{cam_id}_mse"] = MeanSquaredError()
            
            val_metrics["total_mae"] = MeanAbsoluteError()
            val_metrics["total_mse"] = MeanSquaredError()
            
            self.val_metrics = torch.nn.ModuleDict(val_metrics)
        else:
            # single camera case, use existing metrics
            self.val_mae = MeanAbsoluteError()
            self.val_mse = MeanSquaredError()


    def forward(self, x: Any) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """训练开始时调用的 Lightning 钩子函数。"""
        # 默认情况下，Lightning 会在训练开始前执行验证步骤的健全性检查，
        # 因此需要确保验证指标被重置
        self.val_loss.reset()
        if self.is_multicam:
            for metric in self.val_metrics.values():
                metric.reset()
        else:
            self.val_mae.reset()
            self.val_mse.reset()

    def model_step(
        self, batch: Tuple[Any, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        targets = y.contiguous()
        preds = self.forward(x).contiguous()
        loss = self.criterion(preds, targets)
        return loss, preds, targets

    def training_step(
        self, batch: Tuple[Any, Any], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(preds, targets)
        self.train_mae(preds, targets)
        self.train_mse(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.is_multicam:
            # per-camera metrics
            for i in range(self.num_cameras):
                cam_id = self.video_list[i]
                self.val_metrics[f"c{cam_id}_mae"].update(preds[:, i], targets[:, i])
                self.val_metrics[f"c{cam_id}_mse"].update(preds[:, i], targets[:, i])

            # total metrics
            total_preds = torch.sum(preds, dim=1)
            total_targets = torch.sum(targets, dim=1)
            self.val_metrics["total_mae"].update(total_preds, total_targets)
            self.val_metrics["total_mse"].update(total_preds, total_targets)

            # log all metrics
            self.log_dict({f"val/{k}": v for k, v in self.val_metrics.items()}, on_step=False, on_epoch=True)
        else:  # single camera
            self.val_mae.update(preds, targets)
            self.val_mse.update(preds, targets)
            self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if self.is_multicam:
            mae = self.val_metrics["total_mae"].compute()
            self.log("val/total_mae", mae, prog_bar=True)
        else:
            mae = self.val_mae.compute()  # get current val mae
            self.log("val/mae", mae, prog_bar=True)


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good place to setup your model definions, optimizers, schedulers, etc.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used by the
            `Trainer`.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler_partial = self.hparams.scheduler

            # Special handling for SequentialLR which has nested schedulers that need the optimizer
            if (
                isinstance(scheduler_partial, partial)
                and scheduler_partial.func is torch.optim.lr_scheduler.SequentialLR
            ):
                child_partials = scheduler_partial.keywords.get("schedulers", [])
                milestones = scheduler_partial.keywords.get("milestones", [])
                
                instantiated_children = [p(optimizer=optimizer) for p in child_partials]
                
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer=optimizer, schedulers=instantiated_children, milestones=milestones
                )
            else:
                # Standard instantiation for simple schedulers
                scheduler = scheduler_partial(optimizer=optimizer)

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            # For ReduceLROnPlateau, a monitor is required.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "val/loss"
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
            
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FusionLitModule(None, None, None, None)
