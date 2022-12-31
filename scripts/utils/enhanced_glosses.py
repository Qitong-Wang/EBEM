from typing import List
from collections import defaultdict
from dataclasses import field, dataclass

from .commons import get_synset


@dataclass
class Gloss:
    string_a: str
    string_b: str = field(default="")


def get_glosses(senses: List[str]) -> List[Gloss]:
    return [Gloss(string_a=get_synset(sense).definition()) for sense in senses]


def get_enhance_glosses(senses: List[str]) -> List[Gloss]:
    enhance_glosses = list()
    upper_bound_map = defaultdict(list)
    lower_bound = [get_synset(sense) for sense in senses]
    hypernym_paths = [synset.hypernym_paths()[0] for synset in lower_bound]
    for i in range(len(hypernym_paths)):
        upper_bound_map[hypernym_paths[i][0]].append(i)
    while upper_bound_map:
        for upper_synset in list(upper_bound_map.keys()):
            hypernym_path_indices = upper_bound_map.pop(upper_synset)
            if len(hypernym_path_indices) > 1:
                for index in hypernym_path_indices:
                    if len(hypernym_paths[index]) > 2:
                        hypernym_paths[index].pop(0)
                        upper_bound_map[hypernym_paths[index][0]].append(index)
    upper_bound = [hypernym_path[0] for hypernym_path in hypernym_paths]
    for lower, upper in zip(lower_bound, upper_bound):
        enhance_glosses.append(Gloss(
                string_a=lower.definition() if lower == upper else "; ".join([lower.definition(), upper.definition()]),
                string_b="; ".join(lower.examples())
        ))

    return enhance_glosses
