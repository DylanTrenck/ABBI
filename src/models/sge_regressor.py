"""
SGE functional score regressor.

Architecture:
  DNABERT-2 (frozen or last-block lightly tuned) → CLS embedding (768-d)
  + tabular features (50-d):
      ref_aa one-hot (20)  + alt_aa one-hot (20)
      physicochemical deltas (3): hydrophobicity, charge, volume
      consequence one-hot (4): missense | synonymous | stop | splice/other
      aa_pos normalised (1): position in protein / 1863
      phyloP mammalian (1): nucleotide conservation [-4.4, 2.9] / 5
      aGVGD.diff normalised (1): Grantham deviation / 400 (0 for non-missense)
  → concat (818-d) → MLP head → scalar SGE score
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.config import DNABERT2_MODEL, DROPOUT

# ---------------------------------------------------------------------------
# Amino-acid physicochemical tables
# ---------------------------------------------------------------------------

_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_AA_IDX   = {aa: i for i, aa in enumerate(_AA_ORDER)}

_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}
_H_MIN, _H_MAX = -4.5, 4.5

_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C":  0,
    "Q": 0, "E": -1, "G": 0, "H":  0, "I":  0,
    "L": 0, "K": 1, "M":  0, "F":  0, "P":  0,
    "S": 0, "T": 0, "W":  0, "Y":  0, "V":  0,
}

_VOLUME = {
    "G": 48,  "A": 67,  "S": 73,  "C": 86,  "T": 93,
    "V": 105, "I": 124, "L": 124, "P": 90,  "F": 135,
    "Y": 141, "W": 163, "D": 91,  "E": 109, "N": 96,
    "Q": 114, "K": 135, "R": 148, "H": 118, "M": 124,
}
_V_MIN, _V_MAX = 48, 227

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

TABULAR_DIM = 20 + 20 + 3 + 4 + 1 + 1 + 1  # = 50
SEQ_DIM     = 768

_BRCA1_AA_LEN = 1863.0
_PHYLOP_SCALE = 5.0    # divide phyloP to get [-1, ~0.6] range
_AGVGD_SCALE  = 400.0  # divide aGVGD.diff to get rough [0, 1] range

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

CONSEQUENCE_MAP = {
    "missense_variant":       0,
    "synonymous_variant":     1,
    "stop_gained":            2,
    "splice_region_variant":  3,
    "splice_donor_variant":   3,
    "splice_acceptor_variant":3,
    "5_prime_UTR_variant":    3,
    "3_prime_UTR_variant":    3,
    "intron_variant":         3,
}


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if (f != f) else f  # NaN check
    except (ValueError, TypeError):
        return default


def aa_onehot(aa: str) -> list[float]:
    vec = [0.0] * 20
    idx = _AA_IDX.get((aa or "").upper(), -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def physchem_delta(ref: str, alt: str) -> list[float]:
    r, a = (ref or "").upper(), (alt or "").upper()
    dh = (_HYDRO.get(a, 0) - _HYDRO.get(r, 0)) / (_H_MAX - _H_MIN)
    dc = float(_CHARGE.get(a, 0) - _CHARGE.get(r, 0))
    dv = (_VOLUME.get(a, 0) - _VOLUME.get(r, 0)) / (_V_MAX - _V_MIN)
    return [dh, dc, dv]


def consequence_onehot(csq: str) -> list[float]:
    idx = CONSEQUENCE_MAP.get(str(csq or "").split("&")[0].strip(), 3)
    vec = [0.0] * 4
    vec[idx] = 1.0
    return vec


def build_tabular(row: dict, aa_len: float = _BRCA1_AA_LEN) -> list[float]:
    """Build the 50-d tabular feature vector for one variant row."""
    ref_aa = str(row.get("aa_ref",  "") or "")
    alt_aa = str(row.get("aa_alt",  "") or "")
    csq    = str(row.get("consequence", "") or "")

    aa_pos_norm  = _safe_float(row.get("aa_pos"))   / aa_len
    phylop_norm  = _safe_float(row.get("phyloP (mammalian)"), 0.0) / _PHYLOP_SCALE
    agvgd_norm   = _safe_float(row.get("aGVGD.diff"), 0.0) / _AGVGD_SCALE

    return (
        aa_onehot(ref_aa)
        + aa_onehot(alt_aa)
        + physchem_delta(ref_aa, alt_aa)
        + consequence_onehot(csq)
        + [aa_pos_norm, phylop_norm, agvgd_norm]
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SGERegressor(nn.Module):
    def __init__(self, unfreeze_last_block: bool = False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            DNABERT2_MODEL, trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            DNABERT2_MODEL, trust_remote_code=True
        )

        # Freeze all DNABERT-2 parameters by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Optionally re-enable the last transformer block only
        if unfreeze_last_block:
            for param in self.encoder.encoder.layer[-1].parameters():
                param.requires_grad = True

        fused_dim = SEQ_DIM + TABULAR_DIM  # 768 + 50 = 818
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def encode_seq(self, sequences: list[str], device: torch.device) -> torch.Tensor:
        tok = self.tokenizer(
            sequences, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        # No-grad only when encoder is fully frozen; grad flows when last block unfrozen
        ctx = torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) \
              else torch.enable_grad()
        with ctx:
            out = self.encoder(**tok)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        return hidden[:, 0, :]  # CLS token

    def forward(self, sequences: list[str], tabular: torch.Tensor,
                device: torch.device) -> torch.Tensor:
        seq_emb = self.encode_seq(sequences, device)
        fused   = torch.cat([seq_emb, tabular.to(device)], dim=1)
        return self.head(fused).squeeze(-1)
