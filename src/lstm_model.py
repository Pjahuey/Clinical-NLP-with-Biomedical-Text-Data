"""
lstm_model.py
-------------
Responsible: Riley Bendure
Handles:
  - LSTM-based multiple-choice classification model
  - Compatible with existing MedMCQADataset and mc_data_collator
  - Mirrors the transformer model interface for drop-in comparison
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer

LSTM_TOKENIZER = "bert-base-uncased"
NUM_LABELS = 4


def get_lstm_tokenizer():
    """Return the shared BERT tokenizer for the LSTM model."""
    print(f"Loading LSTM tokenizer: {LSTM_TOKENIZER}")
    return AutoTokenizer.from_pretrained(LSTM_TOKENIZER)


class LSTMMultipleChoice(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_choices: int = NUM_LABELS,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.num_choices = num_choices
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def _encode_one_choice(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        T = out.size(1)
        mask = attention_mask[:, :T].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        choice_logits = []
        for c in range(self.num_choices):
            pooled = self._encode_one_choice(
                input_ids[:, c, :], attention_mask[:, c, :]
            )
            pooled = self.dropout(pooled)
            logit = self.classifier(pooled)
            choice_logits.append(logit)
        logits = torch.cat(choice_logits, dim=1)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return _LSTMOutput(loss=loss, logits=logits)


class _LSTMOutput:
    """Lightweight output container matching the HuggingFace interface."""
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits
        self._values = (loss, logits)

    def __getitem__(self, idx):
        return
    def __iter__(self):
        return iter(self._values)
    