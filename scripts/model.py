import torch

from transformers import AutoModel
from collections import OrderedDict
from dataclasses import field, dataclass
from typing import List, Iterable, Optional
from pytorch_metric_learning.losses import CircleLoss

from .dataset import Batch
from .utils import MLMInput


@dataclass
class EncodeOutput:
    instance_guid: str
    context: torch.Tensor
    glosses: torch.Tensor
    labels: torch.Tensor
    candidates: List[str]


@dataclass
class ModelOutput:
    __loss: List[torch.Tensor] = field(default_factory=list)
    __preds: List[torch.Tensor] = field(default_factory=list)

    def __getitem__(self, key) -> torch.Tensor:
        inner_dict = OrderedDict(loss=self.loss, preds=self.preds)
        if isinstance(key, str):
            return inner_dict[key]
        else:
            return tuple(inner_dict.values())[key]

    @property
    def loss(self) -> torch.Tensor:
        return torch.stack(self.__loss).mean()

    @property
    def preds(self) -> torch.Tensor:
        return torch.stack(self.__preds)

    def append(self, loss: Optional[torch.Tensor], pred: torch.Tensor) -> None:
        self.__loss.append(loss)
        self.__preds.append(pred)


class Encoder(torch.nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.0, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(dropout)
        if use_gradient_checkpointing:
            self.deberta.gradient_checkpointing_enable()

    def forward(self, batch: MLMInput) -> torch.Tensor:
        return self.dropout(torch.stack(self.deberta(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids
        ).hidden_states[-4:]).mean(dim=0))


class BEMModel(torch.nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.0, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.context_encoder = Encoder(model_name_or_path, dropout, use_gradient_checkpointing)
        self.gloss_encoder = Encoder(model_name_or_path, dropout, use_gradient_checkpointing)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def encode(self, batch: Batch) -> Iterable[EncodeOutput]:
        context_embeddings = self.context_encoder(batch.context)
        gloss_embeddings = self.gloss_encoder(batch.gloss)
        for table in batch.tables:
            context_target = context_embeddings[table.context_index, table.target_indices].mean(dim=0, keepdim=True)
            gloss_targets = gloss_embeddings[table.gloss_indices, 0]
            yield EncodeOutput(
                    instance_guid=table.instance_guid,
                    context=context_target,
                    glosses=gloss_targets,
                    labels=table.labels,
                    candidates=table.candidates
            )

    def forward(self, batch: Batch) -> ModelOutput:
        model_output = ModelOutput()
        for batch_encode in self.encode(batch):
            logits = (batch_encode.context * batch_encode.glosses).sum(dim=-1, keepdim=True)
            model_output.append(
                    loss=self.loss_fn(input=logits.view(1, -1, 1), target=batch_encode.labels.argmax().view(1, 1)),
                    pred=logits.argmax()
            )

        return model_output


class EBEMModel(BEMModel):
    def __init__(self,
                 model_name_or_path: str,
                 circle_loss_m: float = 0.5,
                 circle_loss_gamma: int = 32,
                 dropout: float = 0.0,
                 use_gradient_checkpointing: bool = False):
        super().__init__(model_name_or_path, dropout=dropout, use_gradient_checkpointing=use_gradient_checkpointing)
        self.loss_fn = CircleLoss(m=circle_loss_m, gamma=circle_loss_gamma)

    def forward(self, batch: Batch) -> ModelOutput:
        model_output = ModelOutput()
        for batch_encode in self.encode(batch):
            logits = self.loss_fn.distance(batch_encode.context, batch_encode.glosses)
            model_output.append(
                    loss=self.loss_fn(
                            embeddings=batch_encode.context,
                            labels=batch_encode.labels.new_ones(1),
                            ref_emb=batch_encode.glosses,
                            ref_labels=batch_encode.labels),
                    pred=logits.argmax()
            )

        return model_output
