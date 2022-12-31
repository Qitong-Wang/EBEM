import torch

from random import sample
from argparse import Namespace
from collections import defaultdict
from torch.utils.data import DataLoader
from dataclasses import field, dataclass
from typing import Dict, List, Union, Optional
from pytorch_lightning import LightningDataModule

from .utils import flatten, rank_zero_print, MLMInput, Tokenizer, RecordData, CorpusReader, DataRecorder, RecordSentence


@dataclass
class Table:
    instance_guid: str
    context_index: torch.Tensor
    target_indices: torch.Tensor
    gloss_indices: torch.Tensor
    labels: torch.Tensor
    candidates: List[str]


@dataclass
class Batch:
    context: MLMInput
    gloss: MLMInput
    tables: List[Table]


@dataclass
class Data:
    train_data: Optional[List[List[RecordSentence]]] = field(default=None)
    val_data: Optional[List[List[RecordSentence]]] = field(default=None)
    predict_data: Optional[List[List[RecordSentence]]] = field(default=None)


class PlDataModule(LightningDataModule):
    def __init__(self, args: Union[None, Dict, Namespace] = None, **kwargs):
        super().__init__()
        self.data = Data()
        self.save_hyperparameters(args)
        self.save_hyperparameters(kwargs)
        self.tokenizer = Tokenizer(self.hparams.model_name_or_path)
        self.corpus_reader = CorpusReader(self.hparams.corpus_path)
        self.data_recorder = DataRecorder(
                corpus_reader=self.corpus_reader,
                tokenizer=self.tokenizer,
                max_seq_len=self.hparams.max_context_seq_len,
                sbert_name_or_path=self.hparams.sbert_name_or_path,
                sbert_batch_size=self.hparams.sbert_batch_size or self.hparams.batch_size,
                sbert_device=self.hparams.sbert_device,
                correct=self.hparams.correct_text,
                add_extra_gloss=self.hparams.add_extra_gloss,
                add_extra_context=self.hparams.add_extra_context,
                min_semantic_sim=self.hparams.min_semantic_sim
        )

    def collate_batch(self, sentences: List[RecordSentence]) -> Batch:
        context = list()
        gloss = list()
        tables = list()
        for sentence in flatten(sentences):
            curr_len = 1 + len(sentence.get_above_token_ids())
            for i, instance in enumerate(sentence.instances):
                if instance.guid:
                    gloss_query = self.corpus_reader.get_gloss(instance.candidates)
                    gloss.extend([self.tokenizer.get_mlm_input(
                            max_seq_len=self.hparams.max_gloss_seq_len,
                            tokens_a=self.tokenizer.encode_string(gloss_query[candidate].string_a),
                            tokens_b=self.tokenizer.encode_string(gloss_query[candidate].string_b)
                    ) for candidate in instance.candidates])
                    tables.append(Table(
                            instance_guid=instance.guid,
                            context_index=torch.tensor(len(context)),
                            target_indices=torch.arange(curr_len, curr_len + len(instance.token_ids)),
                            gloss_indices=torch.arange(len(gloss) - len(instance.candidates), len(gloss)),
                            labels=torch.tensor([int(candidate in instance.gold) for candidate in instance.candidates]),
                            candidates=instance.candidates
                    ))
                curr_len += len(instance.token_ids)
            context.append(self.tokenizer.get_mlm_input(
                    max_seq_len=self.hparams.max_context_seq_len,
                    tokens_a=sentence.get_above_token_ids() + sentence.token_ids + sentence.get_below_token_ids(),
            ))

        return Batch(context=MLMInput.stack(context), gloss=MLMInput.stack(gloss), tables=tables)

    def split_data(self, corpus: RecordData) -> List[List[RecordSentence]]:
        data = list()
        batch = list()
        sent_count = 0
        for document in corpus.documents:
            for i, sentence in enumerate(document.sentences):
                data_sent = RecordSentence(guid=sentence.guid, above=sentence.above, below=sentence.below)
                for j, instance in enumerate(sentence.instances):
                    if instance.guid:
                        if sent_count + len(instance.candidates) + len(batch) + 1 > self.hparams.batch_size:
                            for below in sentence.instances[j:]:
                                data_sent.instances.append(below.copy_as_invalid())
                            if data_sent.is_valid:
                                batch.append(data_sent)
                            if batch:
                                data.append(batch)
                            data_sent = RecordSentence(guid=sentence.guid, above=sentence.above, below=sentence.below)
                            for above in sentence.instances[:j]:
                                data_sent.instances.append(above.copy_as_invalid())
                            batch = list()
                            sent_count = 0
                        sent_count += len(instance.candidates)
                    data_sent.instances.append(instance)
                if data_sent.is_valid:
                    batch.append(data_sent)
        if batch:
            data.append(batch)

        return data

    def preprocess(self, category: str, corpus_names: str) -> List[List[RecordSentence]]:
        record_datas = list()
        rank_zero_print(f"Generate {category} data from corpus: {corpus_names}...")
        for corpus_name in corpus_names.split("-"):
            record_datas.append(self.data_recorder(
                    corpus_name=corpus_name,
                    read_gold=category != "eval",
                    add_multi_gold=self.hparams.add_multi_gold or category != "train",
                    shuffle_candidates=self.hparams.shuffle_data and category == "train",
                    tqdm_disable=self.hparams.simple_display
            ))
        data = RecordData.stack(record_datas)
        if category == "train" and self.hparams.kshot is not None:
            element = defaultdict(list)
            for instance in data.enum_instances(check_guid=True):
                for sense in self.corpus_reader.get_gold(instance.guid):
                    element[sense].append(instance)
            for sense, instances in element.items():
                valid_instances = [instance for instance in instances if instance.guid]
                if len(valid_instances) > self.hparams.kshot:
                    for index in sample(range(len(valid_instances)), len(valid_instances) - self.hparams.kshot):
                        valid_instances[index].guid = None

        return self.split_data(data)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in [None, "fit"]:
            self.data.train_data = self.preprocess(category="train", corpus_names=self.hparams.train_data)
        if stage in [None, "fit", "validate"]:
            self.data.val_data = self.preprocess(category="val", corpus_names=self.hparams.val_data)
        if stage in [None, "predict"]:
            self.data.predict_data = self.preprocess(category="eval", corpus_names=self.hparams.eval_data)

    def get_dataloader(self, dataset: List[List[RecordSentence]], shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, shuffle=shuffle, collate_fn=self.collate_batch, num_workers=self.hparams.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.data.train_data, shuffle=self.hparams.shuffle_data)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.data.val_data, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.data.predict_data, shuffle=False)
