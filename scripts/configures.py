import json
import torch

from math import ceil
from pathlib import Path
from argparse import Namespace
from torchmetrics import Metric
from collections import defaultdict
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning import Trainer, LightningModule, seed_everything
from typing import Dict, List, Tuple, Union, Iterable, Optional, Sequence
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, Adafactor
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar, BasePredictionWriter

from .model import BEMModel, EBEMModel
from .utils import flatten, merge_seq, all_gather


class PlTqdm(TQDMProgressBar):
    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer=trainer, pl_module=pl_module)
        items.pop("v_num", None)

        return items


class MetricAccuracy(Metric):
    full_state_update = False

    def __init__(self, compute_on_step: bool = True, dist_sync_on_step: bool = False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.add_state(name="correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(name="total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, labels: List[torch.Tensor]) -> None:
        compare = [bool(label[pred]) for pred, label in zip(preds, labels)]
        self.correct += sum(compare)
        self.total += len(compare)

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


class PlPredictionWriter(BasePredictionWriter):
    def __init__(self, root: Union[str, Path] = "./predictions", save_name: Optional[str] = None):
        super().__init__(write_interval="epoch")
        self.root = Path(root)
        self.save_paths = list()
        self.custom_name = save_name

    @property
    def save_name(self) -> str:
        return self.custom_name or ModelCheckpoint.CHECKPOINT_NAME_LAST

    @property
    def last_save_path(self) -> Optional[Path]:
        return self.save_paths[-1] if self.save_paths else None

    @classmethod
    def tolist(cls, dict_obj: Dict) -> Dict:
        for key, value in dict_obj.items():
            if isinstance(value, torch.Tensor):
                dict_obj[key] = value.tolist()
            elif isinstance(value, dict):
                dict_obj[key] = cls.tolist(value)

        return dict_obj

    @rank_zero_only
    def save_predictions(self, predictions: Sequence[Dict], save_path: Union[str, Path]) -> None:
        predictions = tuple(flatten(predictions))
        if isinstance(next(iter(predictions[0].values())), str):
            corpus_preds = defaultdict(dict)
            for batch_preds in predictions:
                for instance_guid, pred in batch_preds.items():
                    name, guid = instance_guid.split(".", 1)
                    corpus_preds[name][guid] = pred
            if list(corpus_preds) == ["ALL"]:
                for guid, pred in corpus_preds["ALL"].items():
                    name, guid = guid.split(".", 1)
                    corpus_preds[name][guid] = pred
            for name in corpus_preds:
                with Path(save_path).joinpath(f"{name}.pred.key.txt").open(mode="w+", encoding="utf-8") as file:
                    for guid, pred in sorted(corpus_preds[name].items(), key=lambda item: (len(item[0]), item[0])):
                        file.write(f"{guid} {pred}\n")
        else:
            with Path(save_path).joinpath(f"embeddings.json").open(mode="w+", encoding="utf-8") as file:
                json.dump(self.tolist(dict(merge_seq(tuple(val.items()) for val in predictions))), file, indent=4)

    def write_on_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           epoch_outputs: Sequence[Dict],
                           batch_indices: Optional[Sequence]) -> None:
        save_path = self.root.joinpath(Path(trainer.ckpt_path).stem if trainer.ckpt_path else self.save_name)
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_predictions(predictions=all_gather(epoch_outputs), save_path=save_path)
        self.save_paths.append(save_path)


def get_model(framework_name: str,
              model_name_or_path: str,
              circle_loss_m: float = 0.5,
              circle_loss_gamma: int = 32,
              dropout: float = 0.0,
              use_gradient_checkpointing: bool = False) -> torch.nn.Module:
    if framework_name == "EBEM":
        return EBEMModel(
                model_name_or_path=model_name_or_path,
                circle_loss_m=circle_loss_m,
                circle_loss_gamma=circle_loss_gamma,
                dropout=dropout,
                use_gradient_checkpointing=use_gradient_checkpointing
        )
    elif framework_name == "BEM":
        return BEMModel(model_name_or_path, dropout=dropout, use_gradient_checkpointing=use_gradient_checkpointing)
    else:
        raise RuntimeError(f"Invalid Model framework '{framework_name}'")


def get_optimizer_and_scheduler(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
                                optimizer_name: str,
                                learning_rate: float,
                                warmup_proportion: float,
                                num_training_steps: int,
                                weight_decay: float = 0.0,
                                clip_threshold: float = 1.0,
                                polynomial_power: float = 1.0,
                                epsilon: Optional[float] = None) -> Dict[str, Union[Optimizer, LambdaLR]]:
    grouped_parameters = [{"params": list(), "weight_decay": weight_decay, "name": "decay"},
                          {"params": list(), "weight_decay": 0.0, "name": "no_decay"}]
    for name, parameter in named_parameters:
        grouped_parameters[any(item in name for item in ("bias", "LayerNorm.weight"))]["params"].append(parameter)
    if optimizer_name == "Adafactor":
        optimizer = Adafactor(
                params=grouped_parameters,
                lr=learning_rate,
                eps=(epsilon or 1e-30, 1e-3),
                clip_threshold=clip_threshold,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False
        )
    elif optimizer_name == "AdamW":
        optimizer = AdamW(params=grouped_parameters, lr=learning_rate, eps=epsilon or 1e-8)
    else:
        raise RuntimeError(f"Invalid optimizer '{optimizer_name}'")
    scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(warmup_proportion * num_training_steps),
            num_training_steps=num_training_steps,
            power=polynomial_power
    )

    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "lr"}}


def get_trainer(args: Namespace,
                logger: Union[bool, LightningLoggerBase, Iterable[LightningLoggerBase]] = False,
                callbacks: Union[None, Callback, List[Callback]] = None,
                **kwargs) -> Trainer:
    seed_everything(args.seed, workers=True)

    return Trainer.from_argparse_args(
            args=args,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            devices=getattr(args, "devices", "auto"),
            benchmark=getattr(args, "benchmark", False),
            enable_model_summary=not args.simple_display,
            accelerator=getattr(args, "accelerator", "auto"),
            deterministic=getattr(args, "deterministic", True),
            amp_level=None if args.amp_backend == "native" else args.amp_level,
            gradient_clip_val=None if args.optimizer == "Adafactor" else args.gradient_clip_val,
            strategy=None if torch.cuda.device_count() <= 1 else "ddp_find_unused_parameters_false",
            **kwargs
    )
