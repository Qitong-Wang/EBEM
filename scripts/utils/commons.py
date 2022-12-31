from operator import add
from tqdm.auto import tqdm
from functools import reduce
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from torch.distributed import get_world_size, all_gather_object
from typing import Any, List, Union, Generator, Iterable, Optional, Sequence
from pytorch_lightning.utilities.distributed import rank_zero_only, distributed_available


def flatten(seq_obj: Any) -> Iterable:
    for val in seq_obj:
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            yield from flatten(val)
        else:
            yield val


def merge_seq(seq_obj: Union[Generator, Sequence]) -> Sequence:
    if isinstance(seq_obj, Generator):
        seq_obj = tuple(seq_obj)

    return reduce(add, seq_obj) if seq_obj else list()


def get_synset(sense: str) -> Synset:
    return wordnet.lemma_from_key(sense).synset()


def get_synsets(lemma: str, pos: Optional[str] = None) -> List[Synset]:
    return wordnet.synsets(lemma=lemma, pos=pos)


def get_rank() -> int:
    return rank_zero_only.rank


def all_gather(val: Any) -> Any:
    output = val
    if distributed_available():
        output = [None] * get_world_size()
        all_gather_object(output, val)

    return output


def rank_zero_tqdm(val: Iterable, desc: Optional[str] = None, disable: bool = False) -> Any:
    if get_rank() == 0:
        return tqdm(val, desc=desc, disable=disable)
    else:
        return val


def rank_zero_print(*val: Any) -> None:
    if get_rank() == 0:
        print(*val)
