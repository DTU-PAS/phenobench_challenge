
import torch
import lightning.pytorch as pl
import logging
from torchmetrics.classification import MulticlassJaccardIndex  # type: ignore

from utils.helper import collate_fn
from transformers import Mask2FormerForUniversalSegmentation
from transformers import  Mask2FormerConfig, Mask2FormerModel
from utils.helper import  AverageMeter
from torch_ema import ExponentialMovingAverage

class Mask2FormerPhenobenchPlantRefinementLightningModule(pl.LightningModule):
    def __init__(self, config, processor, **kwargs):
        super().__init__()
        self.config = config
        self.processor = processor
        self.train_stats = {
            "loss": AverageMeter(),

        }
        self.val_stats = {
            "loss": AverageMeter(),
            "IoU": MulticlassJaccardIndex(num_classes=3, average=None)
        }
        self.pred_stats = {
            "loss": AverageMeter(),
            "IoU": MulticlassJaccardIndex(num_classes=3, average=None)
        }
        self.model = self.make_model()

        logging.getLogger().setLevel(logging.INFO)
    
    def make_model(self):
        id2label = {0: "background", 1: "crop"}
        config = Mask2FormerConfig.from_pretrained(self.config.model.pretrained)
        config.id2label = id2label
        config.label2id = {v:k for k,v in id2label.items()}

        base_model = Mask2FormerModel.from_pretrained(self.config.model.pretrained)
        model = Mask2FormerForUniversalSegmentation(config)
        model.model = base_model
        if self.config.model.ema_decay < 1.0:
            self.ema = ExponentialMovingAverage(model.model.parameters(), decay=self.config.model.ema_decay)
        return model

    def forward(self, batch):
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        return outputs
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.config.model.ema_decay < 1.0:
            # workaround, because self.device is not yet set in setup()
            if self.ema.shadow_params[0].device != self.device:
                self.ema.shadow_params = [
                    p.to(self.device)
                    for p in self.ema.shadow_params
                ]
            self.ema.update(self.model.model.parameters())

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self.train_stats["loss"].update(outputs.loss.item())
        self.log("train/loss", self.train_stats["loss"].avg, prog_bar=True)
        if (batch_idx + 1) % self.config.training.log_every_n_steps == 0:
            for v in self.train_stats.values():
                v.reset()
        return outputs.loss

    def on_train_epoch_end(self):
        for v in self.train_stats.values():
            v.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        # Evaluation
        if self.config.training.do_val_metrics:
            img_size = (self.config.data.img_size, self.config.data.img_size)
            predictions = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[img_size for _ in range(len(batch["pixel_values"]))])

            for batch, pred in zip(batch["mask_labels"], predictions):
                gt_semantics = batch[1]
                pred_semantics = pred
                self.val_stats["IoU"].update(pred_semantics.cpu(), gt_semantics.cpu())
        # Loss
        if self.config.training.do_val_loss:
            self.val_stats["loss"].update(outputs.loss.item())
        return self.train_stats["loss"].avg

    def on_validation_epoch_end(self):
        if self.config.training.do_val_metrics:
            # Metrics logging
            iou_per_class = self.val_stats["IoU"].compute()
            self.log("val/IoU/mean", round(float(iou_per_class.mean()), 4))
            self.log("val/IoU/soil", round(float(iou_per_class[0]), 4))
            self.log("val/IoU/plant", round(float(iou_per_class[1]), 4))

        if self.config.training.do_val_loss:
            self.log("val/loss", self.val_stats["loss"].avg, prog_bar=True)

        for v in self.val_stats.values():
            v.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        # Evaluation
        img_size = (self.config.data.img_size, self.config.data.img_size)
        predictions = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[img_size for _ in range(len(batch["pixel_values"]))])

        semantic_predictions = []
        for batch, pred in zip(batch["mask_labels"], predictions):
            gt_semantics = batch[1]
            pred_semantics = pred
            semantic_predictions.append(pred_semantics.cpu().numpy())
            self.pred_stats["IoU"].update(pred_semantics.cpu(), gt_semantics.cpu())
            
        self.pred_stats["loss"].update(outputs.loss.item())

        return semantic_predictions, outputs.loss.item()

    def on_predict_epoch_end(self):
        # Metrics logging
        metrics ={}
        iou_per_class = self.pred_stats["IoU"].compute()
        metrics["IoU"] = {"mean": iou_per_class.mean(),
                          "soil": iou_per_class[0],
                          "plant": iou_per_class[1],}

        metrics["loss"] = self.pred_stats["loss"].avg
        logging.info(metrics)
        return metrics

    def configure_optimizers(self):
        model_params = [
            {"params": self.model.model.transformer_module.parameters()},
            {"params": self.model.class_predictor.parameters()},
            {"params": self.model.model.pixel_level_module.decoder.parameters()},
            {"params": self.model.model.pixel_level_module.encoder.parameters(), "lr": self.config.optimizer.lr * self.config.optimizer.encoder_lr_factor},

        ]
        if self.config.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(model_params, 
                                         lr=self.config.optimizer.lr, 
                                         weight_decay=self.config.optimizer.weight_decay)
    
        elif self.config.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(model_params, 
                                         lr=self.config.optimizer.lr, 
                                         weight_decay=self.config.optimizer.weight_decay)
            
        
        schedulers = []
        milestones = []

        if self.config.scheduler.warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=self.config.scheduler.warmup_start_multiplier,
                                                          total_iters=self.config.scheduler.warmup_steps)
            schedulers.append(scheduler)
            milestones.append(self.config.scheduler.warmup_steps)
        
        if self.config.scheduler.name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=self.config.scheduler.milestones, 
                                                             gamma=self.config.scheduler.gamma)
        elif self.config.scheduler.name == "PolynomialLR":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.config.training.max_steps - 1 - self.config.scheduler.warmup_steps)
        
        elif self.config.scheduler.name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training.max_steps - 1 - self.config.scheduler.warmup_steps)
        
        elif self.config.scheduler.name == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.config.scheduler.T_0, T_mult=self.config.scheduler.T_mult)

        schedulers.append(scheduler)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
        return ret