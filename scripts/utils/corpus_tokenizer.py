import re
import torch

from string import punctuation
from transformers import AutoTokenizer
from sacremoses import MosesDetokenizer
from dataclasses import astuple, dataclass
from typing import List, Tuple, Union, Optional

from .commons import merge_seq


@dataclass
class MLMInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    @classmethod
    def stack(cls, mlm_inputs: List["MLMInput"]) -> "MLMInput":
        return MLMInput(*[torch.stack(tensor) for tensor in zip(*[astuple(mlm_input) for mlm_input in mlm_inputs])])


class Tokenizer(object):
    def __init__(self, model_name_or_path: str):
        self.detokenizer = MosesDetokenizer(lang="en")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @property
    def type(self) -> str:
        return self.tokenizer.__class__.__name__.split("Tokenizer")[0]

    def detokenize(self, words: List[str], correct: bool = True) -> Union[str, List[List[int]]]:
        sentence = self.detokenizer.detokenize(tokens=words, return_str=True)
        if correct:
            sentence = re.sub(r"\* \* f", "**f", sentence)
            sentence = re.sub(r" ?\* \* h", "...", sentence)
            sentence = re.sub(f" (?=(st|nd|rd|th|n't|²)[ {punctuation}])", "", sentence)
            sentence = sentence.replace(" ^ ", "^").replace("i. e.", "i.e.").replace(" & & &", "...")
        if self.type == "Deberta":
            char_map = [index for index, char in enumerate(sentence) if char != " "]
        else:
            char_map = list(range(len(sentence.replace(" ", ""))))
        left = 0
        offsets = list()
        for word in words:
            right = left + len(word.replace(" ", ""))
            offsets.append((char_map[left], char_map[right - 1] + 1))
            left = right

        return sentence, offsets

    def encode_words(self, words: List[str], correct: bool = True) -> Tuple[List[int], str, List[int]]:
        sentence, char_offsets = self.detokenize(words, correct=correct)
        encoding = self.tokenizer(text=sentence, add_special_tokens=False).encodings[0]
        token_offsets = merge_seq([i] * (end - start) for i, (start, end) in enumerate(encoding.offsets))

        return encoding.ids, sentence, [{token_offsets[i] for i in range(start, end)} for start, end in char_offsets]

    def encode_string(self, sentence: str) -> List[int]:
        return self.tokenizer(text=sentence, add_special_tokens=False).encodings[0].ids

    def get_mlm_input(self, max_seq_len: int, tokens_a: List[int], tokens_b: Optional[List[int]] = None) -> MLMInput:
        tokens = [self.cls_token_id] + tokens_a + [self.sep_token_id]
        if tokens_b:
            tokens += tokens_b + [self.sep_token_id]
        del tokens[max_seq_len - 1:-1]
        input_ids = tokens + [self.pad_token_id] * (max_seq_len - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (max_seq_len - len(tokens))
        if self.type == "Roberta":
            token_type_ids = [0] * max_seq_len
        else:
            token_type_ids = ([0] * (len(tokens_a) + 2) + attention_mask[len(tokens_a) + 2:])[:max_seq_len]

        return MLMInput(
                input_ids=torch.tensor(input_ids),
                attention_mask=torch.tensor(attention_mask),
                token_type_ids=torch.tensor(token_type_ids)
        )
