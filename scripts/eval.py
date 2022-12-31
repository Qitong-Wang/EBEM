import re

from pathlib import Path
from argparse import Namespace
from subprocess import getstatusoutput
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import field, asdict, fields, dataclass
from pytorch_lightning.utilities.distributed import rank_zero_only
from typing import Any, Set, List, Dict, Tuple, Union, Callable, Iterable, Optional, Sequence

from .train import PlModule
from .dataset import PlDataModule
from .configures import get_trainer, PlPredictionWriter
from .utils import rank_zero_print, CorpusPath, CorpusReader


@dataclass
class POSSet:
    n: Set[str] = field(default_factory=set)
    v: Set[str] = field(default_factory=set)
    a: Set[str] = field(default_factory=set)
    r: Set[str] = field(default_factory=set)

    def add(self, guid: str, pos: str) -> None:
        if pos in {Field.name for Field in fields(self)}:
            getattr(self, pos).add(guid)
        else:
            raise RuntimeError(f"Unknown part of speech '{pos}'")


@dataclass
class FrequencySet:
    MFS: Set[str] = field(default_factory=set)
    LFS: Set[str] = field(default_factory=set)

    def add(self, guid: str, is_mfs: bool) -> None:
        if is_mfs:
            self.MFS.add(guid)
        else:
            self.LFS.add(guid)


@dataclass
class SpecialSets:
    pos: POSSet
    frequency: FrequencySet

    def __init__(self, corpus_path: Union[str, Path], train_data: str = "Semcor", eval_data: str = "ALL"):
        self.corpus_reader = CorpusReader(corpus_path)
        self.train_data = self.corpus_reader(train_data, read_gold=True, correct=False, add_prefix=False)
        self.eval_data = self.corpus_reader(eval_data, read_gold=True, correct=False, add_prefix=False)
        for Field in fields(self):
            hook_name = f"get_{Field.name}"
            if isinstance(getattr(self, hook_name, None), Callable):
                setattr(self, Field.name, getattr(self, hook_name)())
            else:
                raise RuntimeError(f"Missing '{Field.name}' generation function '{hook_name}()'")

    def items(self) -> Iterable[Tuple[str, Set[str]]]:
        for name_parent, child in asdict(self).items():
            for name_child, values in child.items():
                yield f"{name_parent}.{name_child}", values

    def is_mfs(self, guid: str) -> bool:
        return self.corpus_reader.get_wn30_candidates(guid)[0] in self.corpus_reader.get_gold(guid)

    def get_pos(self) -> POSSet:
        pos = POSSet()
        for instance in self.eval_data.enum_instances(check_guid=True):
            pos.add(guid=instance.guid, pos=instance.pos)

        return pos

    def get_frequency(self) -> FrequencySet:
        frequency = FrequencySet()
        for instance in self.eval_data.enum_instances(check_guid=True):
            frequency.add(guid=instance.guid, is_mfs=self.is_mfs(instance.guid))

        return frequency


class Evaluator(object):
    def __init__(self, args: Union[None, Dict, Namespace] = None, **kwargs):
        self.__model = None
        self.__checkpoint_path = None
        self.__prediction_path = None
        self.hparams = Namespace()
        self.save_hyperparameters(args, kwargs, deterministic=False)
        self.corpus_path = CorpusPath(self.hparams.corpus_path)

    def save_hyperparameters(self, *args, **kwargs) -> None:
        hparams_new = kwargs
        for arg_dict in [arg if isinstance(arg, dict) else vars(arg) for arg in args if arg]:
            hparams_new.update(arg_dict)
        for key, value in hparams_new.items():
            setattr(self.hparams, key, value)

    @property
    def checkpoint_path(self) -> Optional[Path]:
        if not self.__checkpoint_path:
            self.__checkpoint_path = self.parse_checkpoint_path(self.hparams.checkpoint_path)

        return self.__checkpoint_path

    @property
    def prediction_path(self) -> Optional[Dict[str, Path]]:
        if not self.__prediction_path:
            self.__prediction_path = self.parse_prediction_path(self.hparams.prediction_path)

        return self.__prediction_path

    @property
    def model(self) -> Optional[PlModule]:
        if not self.__model:
            self.__model = self.load_model(self.checkpoint_path)

        return self.__model

    @classmethod
    def parse_checkpoint_path(cls, root: Union[str, Path]) -> Path:
        root = Path(root)
        if root.is_file():
            return root.resolve()
        elif root.is_dir():
            paths = [path.resolve() for path in root.iterdir() if path.match("*val_acc=*.ckpt")]
            if paths:
                return max(paths, key=lambda path: re.findall(r"(?<=val_acc=).*?(?=-|$)", path.stem)[0])
            elif root.joinpath(f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}.ckpt").exists():
                return root.joinpath(f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}.ckpt").resolve()
        raise RuntimeError("No available checkpoint file found")

    @classmethod
    def parse_prediction_path(cls, root: Union[str, Path]) -> Dict[str, Path]:
        root = Path(root)
        if root.is_file():
            return {root.name.split(".")[0]: root.resolve()}
        else:
            paths = {path.name.split(".")[0]: path.resolve() for path in root.iterdir() if path.match("*.pred.key.txt")}
            if paths:
                return paths
        raise RuntimeError("No available prediction file found")

    def load_model(self, checkpoint_path: Union[None, str, Path] = None) -> PlModule:
        rank_zero_print(f"Load model checkpoint from {checkpoint_path or self.checkpoint_path}...")
        model = PlModule.load_from_checkpoint(checkpoint_path or self.checkpoint_path)
        model.save_hyperparameters(self.hparams)

        return model

    @classmethod
    def calc(cls, gold: Dict[str, Sequence[str]], pred: Dict[str, Sequence[str]]) -> Dict[str, str]:
        ok = notok = 0.0
        for key in pred:
            if key in gold:
                local_ok = local_notok = 0
                for answer in pred[key]:
                    if answer in gold[key]:
                        local_ok += 1
                    else:
                        local_notok += 1
                ok += local_ok / len(pred[key])
                notok += local_notok / len(pred[key])
        p = ok / (ok + notok)
        r = ok / len(gold)

        return {"P": f"{p * 100: .1f}", "R": f"{r * 100: .1f}", "F1": f"{(2 * p * r) / (p + r) * 100: .1f}"}

    @rank_zero_only
    def score_all(self, prediction_path: Union[None, str, Path] = None) -> Dict[str, Dict[str, str]]:
        scores = dict()
        prediction_path = self.parse_prediction_path(prediction_path) if prediction_path else self.prediction_path
        for name, pred_path in prediction_path.items():
            gold_path = self.corpus_path.get_corpus(name).gold_path.resolve()
            status, output = getstatusoutput(f"java {self.corpus_path.scorer.resolve()} {gold_path} {pred_path}")
            if status == 0:
                scores[name] = dict(zip(("P", "R", "F1"), re.findall(r"(?<==\t).*(?=%)", output)))
            else:
                scores[name] = self.calc(gold=CorpusReader.read_gold(gold_path), pred=CorpusReader.read_gold(pred_path))

        return scores

    @rank_zero_only
    def score_limit(self, guids: Iterable[str], prediction_path: Union[None, str, Path] = None) -> Dict[str, str]:
        limit_pred = dict()
        limit_gold = {guid: None for guid in guids}
        prediction_path = self.parse_prediction_path(prediction_path) if prediction_path else self.prediction_path
        for name, pred_path in prediction_path.items():
            pred = CorpusReader.read_gold(pred_path)
            gold = CorpusReader.read_gold(self.corpus_path.get_corpus(name).gold_path.resolve())
            limit_pred.update({guid: pred[guid] for guid in limit_gold.keys() & pred.keys()})
            limit_gold.update({guid: gold[guid] for guid in limit_gold.keys() & gold.keys()})

        return self.calc(gold=limit_gold, pred=limit_pred)

    @rank_zero_only
    def score_special(self,
                      train_data: str = "Semcor",
                      eval_data: str = "ALL",
                      prediction_path: Union[None, str, Path] = None) -> Dict[str, Dict[str, str]]:
        special = SpecialSets(corpus_path=self.corpus_path.root, train_data=train_data, eval_data=eval_data)

        return {name: self.score_limit(guids=guids, prediction_path=prediction_path) for name, guids in special.items()}

    def run(self,
            checkpoint_path: Union[None, str, Path] = None,
            dataloaders: Union[None, DataLoader, List[DataLoader], PlDataModule] = None) -> Dict[str, Any]:
        model = self.load_model(checkpoint_path) if checkpoint_path else self.model
        save_name = Path(checkpoint_path).stem if checkpoint_path else self.checkpoint_path.stem
        prediction_writer = PlPredictionWriter(root=self.hparams.prediction_path, save_name=save_name)
        trainer = get_trainer(args=model.hparams, max_epochs=1, logger=False, callbacks=[prediction_writer])
        trainer.predict(model=model, dataloaders=dataloaders or PlDataModule(args=model.hparams, eval_data="ALL"))
        self.save_hyperparameters(prediction_path=prediction_writer.last_save_path)

        return {"Evaluation Sets": self.score_all(), "Special Sets": self.score_special()}
