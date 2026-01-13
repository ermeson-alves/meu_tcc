"""
Baseado em:
https://colab.research.google.com/github/qubvel-org/segmentation_models.pytorch/blob/main/examples/camvid_segmentation_multiclass.ipynb
"""
import time
import torch
from dataclasses import dataclass
from typing import Callable, Any
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from src.constants import NUM_REPT, BATCH_SIZE
# from torchmetrics.segmentation import HausdorffDistance


class LOCCAModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, phase, repetition, kfolditer, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        self.number_of_classes = out_classes

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # self.hausdorff_metric = HausdorffDistance(num_classes=out_classes, input_format="index")  # Custo computacional enorme
        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_time = 0
        self.test_time = 0
        self.phase = phase
        self.kfolditer = kfolditer

        assert repetition>0 and repetition<=NUM_REPT
        self.repetition = repetition

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """ Esse é um bloco de código utilizado tanto no treinamento, quanto na
        validação e no teste. """
        image, mask = batch['volume'], batch['mask']
        # Plot para DEBUG
        # img_i = image[3]
        # mask_i = mask[3]
        # plt.subplot(121); plt.imshow(img_i.cpu().permute(2, 1, 0)); 
        # plt.subplot(122); plt.imshow(mask_i.cpu().permute(1,0));
        # plt.show()

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, 1, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        # Plot para DEBUG
        # img_i = image[3]
        # pred_mask_i = pred_mask[3]
        # plt.subplot(121); plt.imshow(img_i.cpu().permute(2, 1, 0)); plt.axis('off');
        # plt.subplot(122); plt.imshow(pred_mask_i.cpu().permute(1,0)); plt.axis('off');
        # plt.show()

        # Calculanto as quantidades de tp,fp,tn,fn que serão usadas como base para as métricas
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )
        metrics = self.compute_metrics(tp, fp, fn, tn, stage)
        self.log_dict(
            # {f"phase{self.phase}_repet{self.repetition}_kfolditer{self.kfolditer}_step_{k}": v for k,v in metrics.items()},
            {f"step_{k}": v for k,v in metrics.items()},
            prog_bar=True,
            batch_size=BATCH_SIZE)
        self.log(
            # f"phase{self.phase}_repet{self.repetition}_kfolditer{self.kfolditer}_step_loss_{stage}",
            f"step_loss_{stage}",
            loss, prog_bar=True,
            batch_size=BATCH_SIZE)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        metrics = self.compute_metrics(tp, fp, fn, tn, stage)

        self.log_dict(
            # {f"phase{self.phase}_repet{self.repetition}_kfolditer{self.kfolditer}_epoch_{k}": v for k,v in metrics.items()},
            {f"epoch_{k}": v for k,v in metrics.items()},
            prog_bar=True,
            batch_size=BATCH_SIZE)

        avg_loss = torch.stack([x["loss"].detach().cpu() for x in outputs]).mean()
        self.log(
            # f"phase{self.phase}_repet{self.repetition}_kfolditer{self.kfolditer}_epoch_avg_loss_{stage}",
            f"epoch_avg_loss_{stage}",
            avg_loss,
            batch_size=BATCH_SIZE)
        
    def on_train_start(self):
        self.training_time = time.time()

    def on_train_end(self):
        training_time = time.time() - self.training_time
        self.logger.experiment.add_scalar(
            # f"phase{self.phase}_repet{self.repetition}_kfolditer{self.kfolditer}_training_time",
            "training_time",
            training_time
        )

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
    
    def on_test_start(self):
        self.test_time = time.time()
    
    def on_test_end(self):
        test_time = time.time() - self.test_time
        self.logger.experiment.add_scalar(
            # f"phase{self.phase}_repet{self.repetition}_kfolditer{self.kfolditer}_test_time",
            "test_time",
            test_time
        )
 
    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        # Sem uso de Scheduler para que não haja mudança na evolução da Lr ao rodar outro modelo
        return {
            "optimizer": optimizer,
        }
    
    def compute_metrics(self, tp, fp, fn, tn, stage):
        # # Housdorff
        # hd = self.hausdorff_metric(pred_mask, mask)
        class_weights = [2., 4., 4.]  # background, lung, lung

        # IoU
        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        per_image_iou_weighted = smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights)
        dataset_iou_weighted = smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted", class_weights=class_weights)

        # sensitivity
        # per_image_sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset_sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro")
        per_image_sensitivity_weighted = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights)
        dataset_sensitivity_weighted = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="weighted", class_weights=class_weights)

        # specificity
        # per_image_specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset_specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro")
        per_image_specificity_weighted = smp.metrics.specificity(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights)
        dataset_specificity_weighted= smp.metrics.specificity(tp, fp, fn, tn, reduction="weighted", class_weights=class_weights)

        # f1-score
        # per_image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        per_image_f1_weighted = smp.metrics.f1_score(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights)
        dataset_f1_weighted = smp.metrics.f1_score(tp, fp, fn, tn, reduction="weighted", class_weights=class_weights)


        return {
            # "hd": hd,

            f"{stage}_per_image_iou": per_image_iou_weighted,
            f"{stage}_dataset_iou": dataset_iou_weighted,

            f"{stage}_per_image_sensitivity": per_image_sensitivity_weighted,
            f"{stage}_dataset_sensitivity": dataset_sensitivity_weighted,
            
            f"{stage}_per_image_specificity": per_image_specificity_weighted,
            f"{stage}_dataset_specificity": dataset_specificity_weighted,

            f"{stage}_per_image_f1": per_image_f1_weighted,
            f"{stage}_dataset_f1": dataset_f1_weighted,
        }



@dataclass
class LOCCAModelFactory:
    arch: str
    encoder_name: str
    in_channels: int
    out_classes: int
    phase: int
    model_kwargs: dict

    def __call__(self, fold: int, repetition: int) -> pl.LightningModule:
        return LOCCAModel(
            arch=self.arch,
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            out_classes=self.out_classes,
            phase=self.phase,
            repetition=repetition,
            kfolditer=fold,
            **self.model_kwargs,
        )
