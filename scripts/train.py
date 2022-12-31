import torch

from argparse import Namespace
from torch.optim import Optimizer
from typing import Any, Dict, Union, Optional
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from .dataset import Batch
from .model import ModelOutput
from .configures import get_model, get_optimizer_and_scheduler, MetricAccuracy


class PlModule(LightningModule):
    def __init__(self, args: Union[None, Dict, Namespace] = None, **kwargs):
        super().__init__()
        self.save_hyperparameters(args)
        self.save_hyperparameters(kwargs)
        self.train_metric = MetricAccuracy(compute_on_step=True, dist_sync_on_step=True)
        self.val_metric = MetricAccuracy(compute_on_step=False, dist_sync_on_step=True)
        self.model = get_model(
                framework_name=self.hparams.framework,
                model_name_or_path=self.hparams.model_name_or_path,
                circle_loss_m=self.hparams.circle_loss_m,
                circle_loss_gamma=self.hparams.circle_loss_gamma,
                dropout=self.hparams.dropout,
                use_gradient_checkpointing=self.hparams.use_gradient_checkpointing
        )

    def forward(self, batch: Batch) -> ModelOutput:
        return self.model(batch)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, preds = self(batch)
        self.train_metric(preds=preds, labels=[table.labels for table in batch.tables])
        self.log(name="acc", value=self.train_metric, prog_bar=True, logger=False, batch_size=1)
        self.log_dict({"train_acc": self.train_metric, "train_loss": loss}, prog_bar=False, logger=True, batch_size=1)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        loss, preds = self(batch)
        self.val_metric(preds=preds, labels=[table.labels for table in batch.tables])
        self.log_dict({"val_acc": self.val_metric, "val_loss": loss}, prog_bar=True, logger=True, batch_size=1)

    def predict_step(self, batch: Batch, batch_idx: int,  dataloader_idx: Optional[int] = None) -> Dict[str, Any]:
        if getattr(self.hparams, "output_embeddings", False):
            return {batch_encode.instance_guid: {"context": batch_encode.context.squeeze(0),
                                                 "gloss": dict(zip(batch_encode.candidates, batch_encode.glosses))}
                    for batch_encode in self.model.encode(batch)}
        else:
            return {table.instance_guid: table.candidates[pred] for table, pred in zip(batch.tables, self(batch).preds)}

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LambdaLR]]:
        return get_optimizer_and_scheduler(
                named_parameters=self.model.named_parameters(),
                optimizer_name=self.hparams.optimizer,
                learning_rate=self.hparams.learning_rate,
                warmup_proportion=self.hparams.warmup_proportion,
                num_training_steps=self.trainer.estimated_stepping_batches,
                weight_decay=self.hparams.weight_decay,
                clip_threshold=self.hparams.gradient_clip_val,
                polynomial_power=self.hparams.polynomial_power,
                epsilon=self.hparams.epsilon
        )
