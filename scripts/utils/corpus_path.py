from pathlib import Path
from typing import Union, Optional
from dataclasses import field, dataclass


@dataclass
class WSDPath:
    data_path: Path
    gold_path: Optional[Path] = field(default=None)

    def __init__(self, root: Union[str, Path]):
        for path in Path(root).iterdir():
            if path.match(f"*.xml"):
                self.data_path = path
            elif path.match(f"*.gold.key.txt"):
                self.gold_path = path


class CorpusPath(object):
    def __init__(self, root: Union[str, Path] = "./WSD_Evaluation_Framework"):
        self.root = Path(root)
        self.wn30 = self.root.joinpath("./Data_Validation/candidatesWN30.txt")
        self.scorer = self.root.joinpath("./Evaluation_Datasets/Scorer.java")
        self.train = {path.name.lower(): WSDPath(root=path)
                      for path in self.root.joinpath("./Training_Corpora").iterdir() if path.is_dir()}
        self.eval = {path.name.lower(): WSDPath(root=path)
                     for path in self.root.joinpath("./Evaluation_Datasets").iterdir() if path.is_dir()}

    def get_corpus(self, name: str) -> Optional[WSDPath]:
        if name.lower() in self.train:
            return self.train[name.lower()]
        elif name.lower() in self.eval:
            return self.eval[name.lower()]
        else:
            raise RuntimeError(f"Invalid corpus name '{name}'")
