import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DROPOUT, EMBED_DIM, NUM_ATTN_HEADS


class CrossAttentionFusion(nn.Module):
    """Fuses sequence and clinical embeddings via self-attention over 2 tokens.

    Both encoders produce EMBED_DIM-dimensional vectors. We stack them as a
    2-token sequence and apply one round of multi-head self-attention, allowing
    each modality to attend to the other. The attended tokens are mean-pooled
    back to a single EMBED_DIM vector for classification.

    Input:  seq_emb   [batch, EMBED_DIM]
            clin_emb  [batch, EMBED_DIM]
    Output: fused_emb [batch, EMBED_DIM]
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_ATTN_HEADS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emb: torch.Tensor,
                clin_emb: torch.Tensor) -> torch.Tensor:
        # Stack into 2-token sequence: [batch, 2, embed_dim]
        tokens = torch.stack([seq_emb, clin_emb], dim=1)

        # Self-attention: each token attends to both
        attended, _ = self.attn(tokens, tokens, tokens)

        # Residual + norm
        fused = self.norm(tokens + self.dropout(attended))

        # Mean pool over the 2 tokens → [batch, embed_dim]
        return fused.mean(dim=1)


class ClassificationHead(nn.Module):
    """Two-layer MLP that maps the fused embedding to a single logit.

    Returns raw logits (not sigmoid) — use BCEWithLogitsLoss during training.

    Input:  [batch, embed_dim]
    Output: [batch, 1]
    """

    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ABBIModel(nn.Module):
    """Full ABBI model: SeqEncoder + ClinicalEncoder → CrossAttentionFusion → ClassificationHead.

    Usage:
        model = ABBIModel(clin_input_dim=21)
        logits = model(sequences, X_clin)   # sequences: list[str], X_clin: [batch, 21]
        probs  = torch.sigmoid(logits)
    """

    def __init__(self, clin_input_dim: int,
                 embed_dim: int = EMBED_DIM,
                 freeze_base_layers: bool = True):
        super().__init__()
        from src.models.encoders import ClinicalEncoder, SeqEncoder

        self.seq_encoder = SeqEncoder(
            embed_dim=embed_dim, freeze_base_layers=freeze_base_layers
        )
        self.clin_encoder = ClinicalEncoder(
            input_dim=clin_input_dim, embed_dim=embed_dim
        )
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)
        self.head = ClassificationHead(embed_dim=embed_dim)

    def forward(self, sequences: list[str],
                x_clin: torch.Tensor) -> torch.Tensor:
        seq_emb  = self.seq_encoder(sequences)            # [batch, embed_dim]
        clin_emb = self.clin_encoder(x_clin)              # [batch, embed_dim]
        fused    = self.fusion(seq_emb, clin_emb)         # [batch, embed_dim]
        return self.head(fused).squeeze(-1)               # [batch]

    def get_attention_weights(self, sequences: list[str],
                              x_clin: torch.Tensor) -> torch.Tensor:
        """Return attention weights from the fusion layer for interpretability.

        Output shape: [batch, num_heads, 2, 2]
        Entry [b, h, i, j] = how much token i attends to token j for sample b.
        Token 0 = sequence, Token 1 = clinical.
        """
        seq_emb  = self.seq_encoder(sequences)
        clin_emb = self.clin_encoder(x_clin)
        tokens   = torch.stack([seq_emb, clin_emb], dim=1)
        _, weights = self.fusion.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        return weights  # [batch, num_heads, 2, 2]
