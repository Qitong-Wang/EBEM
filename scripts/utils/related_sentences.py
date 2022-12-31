import torch

from typing import List, Optional
from itertools import zip_longest
from dataclasses import field, dataclass
from sentence_transformers import util, SentenceTransformer


@dataclass
class RelatedSentIndices:
    center: int
    above: List[int] = field(default_factory=list)
    below: List[int] = field(default_factory=list)

    def append(self, val: int) -> None:
        if val < self.center:
            self.above.append(val)
        elif val > self.center:
            self.below.append(val)


class ContextExtractor(object):
    def __init__(self,
                 model_name_or_path: str,
                 max_seq_len: int,
                 batch_size: int,
                 min_semantic_sim: float = 0.0,
                 device: Optional[str] = None):
        super().__init__()
        self.encoder = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.min_semantic_sim = min_semantic_sim

    @classmethod
    def get_around_indices(cls, center: int, n: int, exclude: Optional[List[int]] = None) -> List[int]:
        return [index for indices in zip_longest(range(center - 1, -1, -1), range(center + 1, n)) for index in indices
                if index and (not exclude or index not in exclude)]

    def __call__(self, sentences: List[str], sentences_len: List[int]) -> List[RelatedSentIndices]:
        result = list()
        if len(sentences) == 1:
            result.append(RelatedSentIndices(center=0))
        else:
            embeddings = self.encoder.encode(sentences=sentences, batch_size=self.batch_size, convert_to_tensor=True)
            for i, cosine_scores in enumerate(util.cos_sim(embeddings, embeddings)):
                curr_len = sentences_len[i] + 2
                sent_indices = RelatedSentIndices(center=i)
                scores, indices = torch.sort(cosine_scores, descending=True)
                indices = indices[scores >= self.min_semantic_sim][1:].tolist()
                for j in indices + self.get_around_indices(center=i, n=len(sentences), exclude=indices):
                    if curr_len + sentences_len[j] <= self.max_seq_len:
                        sent_indices.append(j)
                        curr_len += sentences_len[j]
                    else:
                        break
                sent_indices.above.sort()
                sent_indices.below.sort()
                result.append(sent_indices)

        return result
