import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DNABERT2_MODEL, DROPOUT, EMBED_DIM


# ---------------------------------------------------------------------------
# Clinical / annotation encoder
# ---------------------------------------------------------------------------

class ClinicalEncoder(nn.Module):
    """MLP encoder for the 21-dim ClinVar annotation feature vector.

    Input:  [batch, clin_input_dim]
    Output: [batch, EMBED_DIM]
    """

    def __init__(self, input_dim: int, embed_dim: int = EMBED_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Sequence encoder (DNABERT-2 backbone + projection head)
# ---------------------------------------------------------------------------

class SeqEncoder(nn.Module):
    """DNABERT-2 backbone with a linear projection head.

    Tokenizes raw DNA strings, runs them through DNABERT-2, takes the [CLS]
    token embedding (768-dim), and projects down to EMBED_DIM (128-dim).

    Input:  list of DNA strings, length batch_size
    Output: [batch, EMBED_DIM]
    """

    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT,
                 freeze_base_layers: bool = True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            DNABERT2_MODEL, trust_remote_code=True
        )
        self.backbone = AutoModel.from_pretrained(
            DNABERT2_MODEL, trust_remote_code=True
        )

        if freeze_base_layers:
            # Freeze all layers except the last 2 transformer blocks
            self._freeze_base(keep_last_n=2)

        hidden_size = self.backbone.config.hidden_size  # 768 for DNABERT-2-117M
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def _freeze_base(self, keep_last_n: int = 2) -> None:
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N encoder layers
        encoder_layers = self.backbone.encoder.layer
        for layer in encoder_layers[-keep_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always keep the pooler unfrozen if it exists
        if hasattr(self.backbone, "pooler") and self.backbone.pooler is not None:
            for param in self.backbone.pooler.parameters():
                param.requires_grad = True

    def forward(self, sequences: list[str],
                device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.backbone(**tokens)
        # DNABERT-2 returns a tuple; standard HF models return a BaseModelOutput
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls_emb = hidden[:, 0, :]  # [batch, 768]
        return self.projection(cls_emb)               # [batch, EMBED_DIM]
