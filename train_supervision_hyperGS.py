import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.tuner.tuning import Tuner
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["gt_semantic_seg"]

        prediction = self.net(img)
        loss = self.loss(prediction, mask)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(
                mask[i].cpu().numpy(), pre_mask[i].cpu().numpy()
            )

        return {"loss": loss}

    def on_train_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {"mIoU": mIoU, "F1": F1, "OA": OA}
        # print("train:", eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # print(iou_value)
        self.metrics_train.reset()
        log_dict = {"train_mIoU": mIoU, "train_F1": F1, "train_OA": OA}

        # Add class-specific IoU values to log_dict
        for class_name, iou in iou_value.items():
            log_dict[f"train_IoU_{class_name}"] = iou

        self.log_dict(log_dict, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["gt_semantic_seg"]
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        self.log(
            "val_loss",
            loss_val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {"mIoU": mIoU, "F1": F1, "OA": OA}
        # print("val:", eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # print(iou_value)

        self.metrics_val.reset()
        log_dict = {"val_mIoU": mIoU, "val_F1": F1, "val_OA": OA}

        # Add class-specific IoU values to log_dict
        for class_name, iou in iou_value.items():
            log_dict[f"val_IoU_{class_name}"] = iou

        self.log_dict(log_dict, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    print(config)
    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )

    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode=config.monitor_mode,
    )
    logger = CSVLogger("lightning_logs", name=config.log_name)

    # Define the hyperparameter search space
    lr_range = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    weight_decay_range = [1e-6, 1e-5, 1e-4, 1e-3]
    backbone_lr_range = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    backbone_weight_decay_range = [1e-6, 1e-5, 1e-4, 1e-3]
    batch_size_range = [2, 4, 8, 16]

    best_model = None
    best_score = float("inf") if config.monitor_mode == "min" else float("-inf")

    for lr in lr_range:
        for wd in weight_decay_range:
            for b_lr in backbone_lr_range:
                for b_wd in backbone_weight_decay_range:
                    for bs in batch_size_range:

                        print(f"Current hyperparameters:")
                        print(f"Learning rate: {lr}")
                        print(f"Weight decay: {wd}")
                        print(f"Backbone learning rate: {b_lr}")
                        print(f"Backbone weight decay: {b_wd}")
                        print(f"Batch size: {bs}")

                        # Update config with current hyperparameters
                        config.lr = lr
                        config.weight_decay = wd
                        config.backbone_lr = b_lr
                        config.backbone_weight_decay = b_wd
                        config.train_batch_size = bs
                        config.val_batch_size = bs

                        model = Supervision_Train(config)
                        if config.pretrained_ckpt_path:
                            model = Supervision_Train.load_from_checkpoint(
                                config.pretrained_ckpt_path, config=config
                            )

                        trainer = pl.Trainer(
                            devices=config.gpus,
                            max_epochs=config.max_epoch,
                            accelerator="auto",
                            check_val_every_n_epoch=config.check_val_every_n_epoch,
                            callbacks=[checkpoint_callback, early_stop_callback],
                            strategy="auto",
                            logger=logger,
                        )

                        # Train the model
                        trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)

                        # Check if this is the best model so far
                        current_score = trainer.callback_metrics[config.monitor].item()
                        if (
                            config.monitor_mode == "min" and current_score < best_score
                        ) or (
                            config.monitor_mode == "max" and current_score > best_score
                        ):
                            best_score = current_score
                            best_model = model
                            best_config = config.copy()

    print(f"Best {config.monitor}: {best_score}")
    print("Best hyperparameters:")
    print(f"Learning rate: {best_config.learning_rate}")
    print(f"Weight decay: {best_config.weight_decay}")
    print(f"Backbone learning rate: {best_config.backbone_lr}")
    print(f"Backbone weight decay: {best_config.backbone_weight_decay}")
    print(f"Batch size: {best_config.batch_size}")

    # Save the best hyperparameters
    with open("best_hyperparameters.txt", "w") as f:
        f.write(f"Best {config.monitor}: {best_score}\n")
        f.write("Best hyperparameters:\n")
        f.write(f"Learning rate: {best_config.learning_rate}\n")
        f.write(f"Weight decay: {best_config.weight_decay}\n")
        f.write(f"Backbone learning rate: {best_config.backbone_lr}\n")
        f.write(f"Backbone weight decay: {best_config.backbone_weight_decay}\n")
        f.write(f"Batch size: {best_config.batch_size}\n")

    # Save the best model
    best_model.save_checkpoint("best_model.ckpt")


if __name__ == "__main__":
    main()
