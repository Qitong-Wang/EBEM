from dataclasses import field, dataclass
from typing import List, Iterable, Optional

from .corpus_reader import CorpusReader
from .corpus_tokenizer import Tokenizer
from .commons import merge_seq, rank_zero_tqdm
from .related_sentences import ContextExtractor
from .enhanced_glosses import get_glosses, get_enhance_glosses


@dataclass
class RecordInstance:
    guid: Optional[str]
    token_ids: List[int]
    candidates: List[str] = field(default_factory=list)
    gold: List[str] = field(default_factory=list)

    def copy_as_invalid(self) -> "RecordInstance":
        return RecordInstance(guid=None, token_ids=self.token_ids)


@dataclass
class RecordSentence:
    guid: str
    above: List["RecordSentence"] = field(default_factory=list)
    below: List["RecordSentence"] = field(default_factory=list)
    instances: List[RecordInstance] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def is_valid(self) -> bool:
        return any(instance.guid for instance in self.instances)

    @property
    def token_ids(self) -> List[int]:
        return merge_seq(instance.token_ids for instance in self.instances)

    def get_above_token_ids(self) -> List[int]:
        return merge_seq(sentence.token_ids for sentence in self.above)

    def get_below_token_ids(self) -> List[int]:
        return merge_seq(sentence.token_ids for sentence in self.below)


@dataclass
class RecordDocument:
    guid: str
    sentences: List[RecordSentence] = field(default_factory=list)


@dataclass
class RecordData:
    name: str
    documents: List[RecordDocument] = field(default_factory=list)

    @classmethod
    def stack(cls, datas: List["RecordData"]) -> "RecordData":
        return RecordData("_".join([data.name for data in datas]), merge_seq(data.documents for data in datas))

    def enum_instances(self, check_guid: bool = True) -> Iterable[RecordInstance]:
        for document in self.documents:
            for sentence in document.sentences:
                for instance in sentence.instances:
                    if not check_guid or instance.guid:
                        yield instance


class DataRecorder(object):
    def __init__(self,
                 corpus_reader: CorpusReader,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 sbert_name_or_path: str,
                 sbert_batch_size: int,
                 sbert_device: str = "cuda",
                 correct: bool = True,
                 add_extra_gloss: bool = True,
                 add_extra_context: bool = True,
                 min_semantic_sim: float = 0.0):
        self.corpus_reader = corpus_reader
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.correct = correct
        self.add_extra_gloss = add_extra_gloss
        self.add_extra_context = add_extra_context
        self.context_extractor = ContextExtractor(
                model_name_or_path=sbert_name_or_path,
                batch_size=sbert_batch_size,
                max_seq_len=max_seq_len,
                min_semantic_sim=min_semantic_sim,
                device=sbert_device
        )

    def __call__(self,
                 corpus_name: str,
                 read_gold: bool = True,
                 add_multi_gold: bool = True,
                 shuffle_candidates: bool = False,
                 tqdm_desc: str = "creat_record_data",
                 tqdm_disable: bool = False):
        record_data = RecordData(name=corpus_name)
        corpus_data = self.corpus_reader(name=corpus_name, read_gold=read_gold, correct=self.correct)
        for document in rank_zero_tqdm(corpus_data.documents, desc=tqdm_desc, disable=tqdm_disable):
            sentence_extra = list()
            record_document = RecordDocument(guid=document.guid)
            for sentence in document.sentences:
                record_sentence = RecordSentence(guid=sentence.guid)
                token_ids, sentence_str, offsets_map = self.tokenizer.encode_words(sentence.words, correct=self.correct)
                instances_mask = [-1] * len(token_ids)
                for i, instance in enumerate(sentence.instances):
                    if instance.guid:
                        for j in offsets_map[i]:
                            if instances_mask[j] != -1:
                                raise RuntimeError(f"Overlap target tokens on position '{i}'")
                            instances_mask[j] = i
                if len(token_ids) > self.max_seq_len - 2:
                    drop = instances_mask[self.max_seq_len - 2]
                    instances_mask = [mask if mask != drop else -1 for mask in instances_mask[:self.max_seq_len - 2]]
                for i, mask in enumerate(instances_mask):
                    curr_guid = sentence.instances[mask].guid if mask != -1 else None
                    if record_sentence.instances and record_sentence.instances[-1].guid == curr_guid:
                        record_sentence.instances[-1].token_ids.append(token_ids[i])
                    elif curr_guid:
                        record_sentence.instances.append(RecordInstance(
                                guid=curr_guid,
                                token_ids=[token_ids[i]],
                                candidates=self.corpus_reader.get_candidates(
                                        guid=curr_guid,
                                        shuffle=shuffle_candidates,
                                        gloss_collate_fn=get_enhance_glosses if self.add_extra_gloss else get_glosses
                                ),
                                gold=self.corpus_reader.get_gold(curr_guid)[:None if add_multi_gold else 1]
                        ))
                    else:
                        record_sentence.instances.append(RecordInstance(guid=curr_guid, token_ids=[token_ids[i]]))
                sentence_extra.append((sentence_str, len(record_sentence)))
                record_document.sentences.append(record_sentence)
            if self.add_extra_context:
                for sentence, indices in zip(record_document.sentences, self.context_extractor(*zip(*sentence_extra))):
                    sentence.above = [record_document.sentences[index] for index in indices.above]
                    sentence.below = [record_document.sentences[index] for index in indices.below]
            record_data.documents.append(record_document)

        return record_data
