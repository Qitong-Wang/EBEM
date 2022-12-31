import re

from pathlib import Path
from random import sample
from dataclasses import field, dataclass
from xml.etree.ElementTree import ElementTree
from typing import Dict, List, Tuple, Union, Callable, Iterable, Optional

from .corpus_path import CorpusPath
from .enhanced_glosses import Gloss


@dataclass
class CorpusInstance:
    guid: Optional[str]
    lemma: str
    pos: str
    word: str


@dataclass
class CorpusSentence:
    guid: str
    instances: List[CorpusInstance] = field(default_factory=list)

    @property
    def words(self) -> List[str]:
        return [instance.word for instance in self.instances]


@dataclass
class CorpusDocument:
    guid: str
    sentences: List[CorpusSentence] = field(default_factory=list)


@dataclass
class CorpusData:
    name: str
    documents: List[CorpusDocument] = field(default_factory=list)

    def enum_instances(self, check_guid: bool = True) -> Iterable[CorpusInstance]:
        for document in self.documents:
            for sentence in document.sentences:
                for instance in sentence.instances:
                    if not check_guid or instance.guid:
                        yield instance


@dataclass
class CorpusExtraData:
    gold: Dict[str, Tuple[str]] = field(default_factory=dict)
    wn30: Dict[Tuple[str, str], Tuple[str]] = field(default_factory=dict)
    gloss: Dict[Tuple[str], Dict[str, Gloss]] = field(default_factory=dict)
    element: Dict[str, CorpusInstance] = field(default_factory=dict)


class CorpusReader(object):
    POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
    CORRECT_MAP = {"``": '"', "''": '"', " 's": "'s", "s ' ": "s' ", "&amp;": "&", "&apos;": "'"}

    def __init__(self, root: Union[str, Path]):
        self.extra_data = CorpusExtraData()
        self.corpus_path = CorpusPath(root=root)
        with self.corpus_path.wn30.open(mode="r", encoding="utf-8") as file:
            for line in file.readlines():
                lemma, pos, candidates = line.strip().split("\t", maxsplit=2)
                self.extra_data.wn30[(lemma, pos)] = tuple(candidates.split("\t"))

    @classmethod
    def read_gold(cls, path: Union[str, Path]) -> Dict[str, Tuple[str]]:
        golddata = dict()
        with Path(path).open(mode="r", encoding="utf-8") as file:
            for line in file.readlines():
                guid, gold = line.strip().split(maxsplit=1)
                golddata[guid] = tuple(gold.split())

        return golddata

    def __call__(self, name: str, read_gold: bool = True, correct: bool = True, add_prefix: bool = True) -> CorpusData:
        data = CorpusData(name=name)
        path = self.corpus_path.get_corpus(name)
        prefix = f"{name}." if add_prefix else ""
        xml_root = ElementTree(file=path.data_path).getroot()
        if name.lower() == "wngt":
            pos_map = {"N": "n", "V": "v", "J": "a", "R": "r"}
            for i, sent in enumerate(xml_root[0][0]):
                instance_id = 1
                document_guid = f"{prefix}d{i + 1:0>6d}"
                sentence = CorpusSentence(guid=f"{document_guid}.s001")
                for ele in sent:
                    instance = CorpusInstance(
                            guid=f"{sentence.guid}.t{instance_id:0>3d}" if ele.get("wn30_key", None) else None,
                            lemma=ele.get("lemma", ele.get("surface_form")),
                            pos=pos_map.get(ele.get("pos")[0], ele.get("pos")),
                            word=ele.get("surface_form")
                    )
                    if instance.guid:
                        if read_gold:
                            self.extra_data.gold[instance.guid] = ele.get("wn30_key"),
                        instance_id += 1
                        self.extra_data.element[instance.guid] = instance
                    sentence.instances.append(instance)
                data.documents.append(CorpusDocument(guid=document_guid, sentences=[sentence]))
        else:
            for doc in xml_root:
                document = CorpusDocument(guid=prefix + doc.get("id"))
                for sent in doc:
                    sentence = CorpusSentence(guid=prefix + sent.get("id"))
                    for ele in sent:
                        word = ele.text
                        if correct:
                            if word == "n't" and sentence.instances[-1].word in ("won", "can"):
                                sentence.instances[-1].lemma += "'t"
                                sentence.instances[-1].word += "'t"
                                continue
                            word = re.sub("(?<=[^s]) ' ", "'", self.CORRECT_MAP.get(word, word))
                        instance = CorpusInstance(
                                guid=prefix + ele.get("id") if ele.tag == "instance" else None,
                                lemma=ele.get("lemma"),
                                pos=self.POS_MAP.get(ele.get("pos"), ele.get("pos")),
                                word=word
                        )
                        if instance.guid:
                            self.extra_data.element[instance.guid] = instance
                        sentence.instances.append(instance)
                    document.sentences.append(sentence)
                data.documents.append(document)
            if read_gold:
                for guid, gold in self.read_gold(path.gold_path).items():
                    self.extra_data.gold[prefix + guid] = gold

        return data

    def get_gold(self, guid: str) -> Tuple[str]:
        return self.extra_data.gold.get(guid, list())

    def get_gloss(self, keys: Iterable[str]) -> Dict[str, Gloss]:
        return self.extra_data.gloss[tuple(sorted(keys))]

    def get_wn30_candidates(self, guid: str) -> Tuple[str]:
        return self.extra_data.wn30[(self.extra_data.element[guid].lemma, self.extra_data.element[guid].pos)]

    def get_candidates(self, guid: str, gloss_collate_fn: Callable, shuffle: bool = True) -> Tuple[str]:
        candidates = self.get_wn30_candidates(guid)
        gloss_key = tuple(sorted(candidates))
        if gloss_key not in self.extra_data.gloss:
            self.extra_data.gloss[gloss_key] = dict(zip(candidates, gloss_collate_fn(candidates)))

        return tuple(sample(candidates, k=len(candidates))) if shuffle else candidates
