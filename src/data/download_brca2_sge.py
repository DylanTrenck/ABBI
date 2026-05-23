"""
Download and annotate BRCA2 SGE data from MAVE-DB.

Source: urn:mavedb:00001225-a-1
  ~6,959 BRCA2 SNVs (exons 15-26, HAP1 cells, Starita/Findlay lab)

Steps:
  1. Download scores CSV from MAVE-DB API
  2. Parse aa_ref, aa_alt, aa_pos from hgvs_pro (3-letter to 1-letter)
  3. Compute aGVGD.diff (Grantham 1974 distance between ref/alt amino acids)
  4. Batch Ensembl VEP annotation: consequence, genomic coordinates, exon
  5. Query UCSC REST API for phyloP100way scores (range-batched by exon)
  6. Derive function_class (LOF/INT/FUNC) from score vs. synonymous distribution
  7. Save to data/raw/brca2_sge.csv

Usage:
  python src/data/download_brca2_sge.py
"""

import io
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import SGE_BRCA2_RAW

MAVEDB_URN   = "urn:mavedb:00001225-a-1"
MAVEDB_API   = "https://api.mavedb.org/api/v1"
VEP_URL      = "https://rest.ensembl.org/vep/human/hgvs"
UCSC_URL     = "https://api.genome.ucsc.edu/getData/track"
PHYLOP_TRACK = "phyloP100way"

BRCA2_CHR    = "13"
BRCA2_AA_LEN = 3418

# ---------------------------------------------------------------------------
# Amino acid tables (3-letter to 1-letter conversion)
# ---------------------------------------------------------------------------

_AA3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'Xaa': 'X',
}

# Grantham (1974) physicochemical parameters: composition, polarity, volume
_GRAM_C = {'A': 0,    'R': 0.65, 'N': 1.33, 'D': 1.38, 'C': 2.75,
           'Q': 1.24, 'E': 1.27, 'G': 0.74, 'H': 0.58, 'I': 0,
           'L': 0,    'K': 0.33, 'M': 0,    'F': 0,    'P': 0.39,
           'S': 1.42, 'T': 0.71, 'W': 0.13, 'Y': 0.20, 'V': 0}
_GRAM_P = {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
           'Q': 10.5,'E': 12.3, 'G': 9.0,  'H': 10.4, 'I': 5.2,
           'L': 4.9, 'K': 11.3, 'M': 5.7,  'F': 5.2,  'P': 8.0,
           'S': 9.2, 'T': 8.6,  'W': 5.4,  'Y': 6.2,  'V': 5.9}
_GRAM_V = {'A': 31,  'R': 124,  'N': 56,   'D': 54,   'C': 55,
           'Q': 85,  'E': 83,   'G': 3,    'H': 96,   'I': 111,
           'L': 111, 'K': 119,  'M': 105,  'F': 132,  'P': 32.5,
           'S': 32,  'T': 61,   'W': 170,  'Y': 136,  'V': 84}


def grantham_distance(a: str, b: str) -> float:
    """Grantham (1974) physicochemical distance between two amino acids."""
    if not a or not b or a == b:
        return 0.0
    a, b = a.upper(), b.upper()
    if a not in _GRAM_C or b not in _GRAM_C:
        return 0.0
    dc = _GRAM_C[a] - _GRAM_C[b]
    dp = _GRAM_P[a] - _GRAM_P[b]
    dv = _GRAM_V[a] - _GRAM_V[b]
    return 50.723 * (1.833 * dc**2 + 0.1018 * dp**2 + 0.000399 * dv**2) ** 0.5


# ---------------------------------------------------------------------------
# HGVS protein parsing (3-letter amino acid codes)
# ---------------------------------------------------------------------------

_HGVS_PRO_RE = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)', re.ASCII)


def parse_hgvs_pro(hgvs_pro: str) -> tuple:
    """
    Parse p.Arg2336Cys -> (ref='R', alt='C', pos=2336).
    Returns ('', '', None) for synonymous/non-missense/unparseable.
    """
    if not isinstance(hgvs_pro, str):
        return '', '', None
    if ':' in hgvs_pro:
        hgvs_pro = hgvs_pro.split(':', 1)[1]
    if hgvs_pro in ('p.=', 'p.0', 'p.?', ''):
        return '', '', None
    m = _HGVS_PRO_RE.search(hgvs_pro)
    if not m:
        return '', '', None
    ref3, pos_str, alt3 = m.group(1), m.group(2), m.group(3)
    ref1 = _AA3TO1.get(ref3, '')
    alt1 = '*' if alt3 == '*' else _AA3TO1.get(alt3, '')
    try:
        pos = int(pos_str)
    except ValueError:
        pos = None
    return ref1, alt1, pos


# ---------------------------------------------------------------------------
# Ensembl VEP batch annotation
# ---------------------------------------------------------------------------

_VEP_HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json",
}


def _parse_exon_from_tc(tc: dict) -> str:
    """
    Extract simplified exon label from VEP transcript consequence.
    - Exonic: exon field "15/27" → "E15"
    - Intronic: intron field "14/26" → "E15" (assign to next exon, splice-acceptor convention)
    """
    exon_str   = tc.get("exon",   "") or ""
    intron_str = tc.get("intron", "") or ""
    if exon_str:
        m = re.match(r"(\d+)", str(exon_str))
        return f"E{m.group(1)}" if m else ""
    if intron_str:
        m = re.match(r"(\d+)", str(intron_str))
        if m:
            return f"E{int(m.group(1)) + 1}"  # intron N → adjacent to exon N+1
    return ""


def vep_batch(hgvs_list: list, batch_size: int = 200) -> dict:
    """
    Annotate HGVS identifiers via Ensembl VEP REST API.
    Returns dict mapping hgvs_id -> annotation fields.
    """
    results = {}
    total = len(hgvs_list)
    for i in range(0, total, batch_size):
        batch = hgvs_list[i: i + batch_size]
        payload = {"hgvs_notations": batch}
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{VEP_URL}?canonical=1&numbers=1",
                    json=payload,
                    headers=_VEP_HEADERS,
                    timeout=90,
                )
                if resp.status_code == 200:
                    for entry in resp.json():
                        hgvs_id = entry.get("id", "")
                        tcs = entry.get("transcript_consequences", [])
                        tc = next((t for t in tcs if t.get("canonical") == 1), None)
                        if tc is None and tcs:
                            tc = tcs[0]
                        aa_str = tc.get("amino_acids", "/") if tc else "/"
                        aa_parts = aa_str.split("/") if "/" in aa_str else [aa_str, ""]
                        results[hgvs_id] = {
                            "consequence": (tc.get("consequence_terms", [""])[0] if tc else ""),
                            "aa_ref":      aa_parts[0].strip() if aa_parts else "",
                            "aa_alt":      aa_parts[1].strip() if len(aa_parts) > 1 else "",
                            "aa_pos":      tc.get("protein_start") if tc else None,
                            "exon_label":  _parse_exon_from_tc(tc) if tc else "",
                            "chrom":       str(entry.get("seq_region_name", "")),
                            "genomic_pos": entry.get("start"),
                        }
                    break
                elif resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    print(f"    VEP rate-limited: sleeping {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"    VEP HTTP {resp.status_code} (batch {i//batch_size + 1})")
                    time.sleep(5)
                    break
            except requests.RequestException as e:
                print(f"    VEP error (attempt {attempt+1}): {e}")
                time.sleep(10)
        time.sleep(0.5)
        done = min(i + batch_size, total)
        print(f"  VEP: {done:,}/{total:,} annotated", end="\r")
    print()
    return results


# ---------------------------------------------------------------------------
# UCSC REST API — range-batched phyloP100way queries
# ---------------------------------------------------------------------------

def fetch_phylop(positions: list) -> dict:
    """
    Fetch phyloP100way for a list of (chrom_str, pos_1based) tuples.

    Groups nearby positions into range queries to minimise API calls.
    Returns dict mapping (chrom, pos_1based) -> float.
    """
    # Group by chromosome
    by_chrom: dict = defaultdict(list)
    for chrom, pos in positions:
        if chrom and pos is not None:
            by_chrom[str(chrom)].append(int(pos))

    cache: dict = {}
    RANGE_PAD = 5   # extra bp each side

    for chrom, pos_list in by_chrom.items():
        pos_list = sorted(set(pos_list))
        chrom_str = f"chr{chrom}" if not chrom.startswith("chr") else chrom

        # Build non-overlapping query windows that cover runs of nearby positions
        windows: list = []
        if not pos_list:
            continue
        w_start = pos_list[0]
        w_end   = pos_list[0]
        for p in pos_list[1:]:
            if p - w_end <= 500:     # merge positions within 500bp into one query
                w_end = p
            else:
                windows.append((w_start - RANGE_PAD, w_end + RANGE_PAD))
                w_start = p
                w_end   = p
        windows.append((w_start - RANGE_PAD, w_end + RANGE_PAD))

        for win_start, win_end in windows:
            # UCSC uses 0-based half-open coordinates
            start0 = max(0, win_start - 1)
            end0   = win_end
            for attempt in range(3):
                try:
                    resp = requests.get(
                        UCSC_URL,
                        params={
                            "genome": "hg38",
                            "track":  PHYLOP_TRACK,
                            "chrom":  chrom_str,
                            "start":  start0,
                            "end":    end0,
                        },
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        vals = data.get(PHYLOP_TRACK, [])
                        for entry in vals:
                            if isinstance(entry, dict):
                                pos0_entry = int(entry["start"])
                                val = float(entry["value"])
                                cache[(chrom, pos0_entry + 1)] = val
                            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                                pos0_entry = int(entry[0])
                                val = float(entry[2])
                                cache[(chrom, pos0_entry + 1)] = val
                        break
                    elif resp.status_code == 429:
                        time.sleep(30)
                    else:
                        break
                except Exception:
                    time.sleep(5)
            time.sleep(0.15)

    hit_count = sum(1 for (c, p) in [(ch, po) for ch, po in positions
                                     if (ch, po) in cache])
    print(f"  phyloP: {len(cache):,} unique positions cached  "
          f"(coverage: {hit_count}/{len(positions)})")
    return cache


# ---------------------------------------------------------------------------
# Function class assignment (relative to synonymous variant distribution)
# ---------------------------------------------------------------------------

def assign_function_class(scores: pd.Series, syn_mask: pd.Series,
                          lof_nsigma: float = 2.5,
                          func_nsigma: float = 0.0) -> pd.Series:
    """
    Assign LOF / INT / FUNC relative to synonymous variant distribution.

    Thresholds (relative to synonymous mean/std):
      LOF:  score <= syn_mean - lof_nsigma  * syn_std   (default: -2.5σ, conservative LOF)
      FUNC: score >= syn_mean - func_nsigma * syn_std   (default:  0.0σ, at synonymous mean)
      INT:  everything between the two thresholds
    """
    syn_scores = scores[syn_mask].dropna()
    if len(syn_scores) >= 5:
        syn_mean = syn_scores.mean()
        syn_std  = syn_scores.std()
        lof_thresh  = syn_mean - lof_nsigma  * syn_std
        func_thresh = syn_mean - func_nsigma * syn_std
        print(f"  Synonymous: n={len(syn_scores)}  mean={syn_mean:.3f}  std={syn_std:.3f}")
    else:
        lof_thresh  = float(scores.quantile(0.15))
        func_thresh = float(scores.quantile(0.70))
        print(f"  Fallback percentile thresholds (too few synonymous variants)")

    print(f"  LOF threshold  (score <= {lof_thresh:.3f})")
    print(f"  FUNC threshold (score >= {func_thresh:.3f})")

    def classify(s):
        if pd.isna(s):
            return np.nan
        if s <= lof_thresh:
            return "LOF"
        if s >= func_thresh:
            return "FUNC"
        return "INT"

    return scores.apply(classify)


# ---------------------------------------------------------------------------
# MAVE-DB download
# ---------------------------------------------------------------------------

def download_mavedb() -> pd.DataFrame:
    url = f"{MAVEDB_API}/score-sets/{MAVEDB_URN}/scores"
    print(f"Downloading MAVE-DB: {MAVEDB_URN}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  Downloaded: {len(df):,} rows  cols={list(df.columns)}")
    # Drop MAVE-DB special sentinel rows
    if "hgvs_nt" in df.columns:
        df = df[~df["hgvs_nt"].isin(["_wt", "_sy", "_stop"])].copy()
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SGE_BRCA2_RAW.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download ---
    df = download_mavedb()
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    print(f"After dropping NA scores: {len(df):,} rows")

    # --- Step 2: Parse hgvs_pro -> aa features ---
    parsed = [parse_hgvs_pro(v) for v in df.get("hgvs_pro", pd.Series(dtype=str))]
    df["aa_ref"] = [p[0] for p in parsed]
    df["aa_alt"] = [p[1] for p in parsed]
    df["aa_pos"]  = [p[2] for p in parsed]

    # --- Step 3: Grantham distance ---
    df["aGVGD.diff"] = [
        grantham_distance(r, a)
        for r, a in zip(df["aa_ref"], df["aa_alt"])
    ]

    # --- Step 4: VEP batch annotation ---
    hgvs_col = "hgvs_nt"
    if hgvs_col not in df.columns:
        print("ERROR: 'hgvs_nt' column not found in MAVE-DB data.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    hgvs_list = df[hgvs_col].dropna().unique().tolist()
    print(f"\nAnnotating {len(hgvs_list):,} unique variants with Ensembl VEP...")
    vep_results = vep_batch(hgvs_list)

    def vep_get(hgvs, field):
        if not isinstance(hgvs, str):
            return "" if field not in ("genomic_pos", "aa_pos") else None
        return vep_results.get(hgvs, {}).get(field, "" if field not in ("genomic_pos", "aa_pos") else None)

    df["consequence"] = df[hgvs_col].map(lambda h: vep_get(h, "consequence"))
    df["chrom"]       = df[hgvs_col].map(lambda h: vep_get(h, "chrom"))
    df["genomic_pos"] = df[hgvs_col].map(lambda h: vep_get(h, "genomic_pos"))
    df["experiment"]  = df[hgvs_col].map(lambda h: vep_get(h, "exon_label"))

    # Fill aa_ref/aa_alt from VEP where hgvs_pro parsing failed
    vep_aa_ref = df[hgvs_col].map(lambda h: vep_get(h, "aa_ref"))
    vep_aa_alt = df[hgvs_col].map(lambda h: vep_get(h, "aa_alt"))
    vep_aa_pos = df[hgvs_col].map(lambda h: vep_get(h, "aa_pos"))

    df.loc[df["aa_ref"] == "", "aa_ref"] = vep_aa_ref[df["aa_ref"] == ""]
    df.loc[df["aa_alt"] == "", "aa_alt"] = vep_aa_alt[df["aa_alt"] == ""]
    df.loc[df["aa_pos"].isna(), "aa_pos"] = vep_aa_pos[df["aa_pos"].isna()]

    print(f"  VEP annotation rate:   {(df['consequence'] != '').mean():.1%}")
    print(f"  Exon coverage:         {(df['experiment'] != '').mean():.1%}")
    print(f"  Unique exons found:    {sorted(df['experiment'].unique())}")

    # --- Step 5: phyloP100way ---
    valid_mask = df["genomic_pos"].notna() & (df["chrom"] != "")
    pos_tuples = list(zip(
        df.loc[valid_mask, "chrom"].tolist(),
        df.loc[valid_mask, "genomic_pos"].astype(float).astype(int).tolist(),
    ))
    print(f"\nFetching phyloP for {len(set(pos_tuples)):,} unique genomic positions...")
    phylop_cache = fetch_phylop(pos_tuples)

    def get_phylop(row):
        chrom = row.get("chrom", "")
        pos   = row.get("genomic_pos")
        if not chrom or pd.isna(pos):
            return np.nan
        return phylop_cache.get((str(chrom), int(pos)), np.nan)

    df["phyloP (mammalian)"] = df.apply(get_phylop, axis=1)
    print(f"  phyloP coverage: {df['phyloP (mammalian)'].notna().mean():.1%}")

    # --- Step 6: Function class ---
    syn_mask = (
        (df["consequence"] == "synonymous_variant")
        | (df.get("hgvs_pro", pd.Series("", index=df.index)).fillna("").str.contains(r"p\.=", regex=False))
    )
    print(f"\nAssigning function_class (synonymous variants: {syn_mask.sum():,})...")
    df["function_class"] = assign_function_class(df["score"], syn_mask)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  BRCA2 SGE dataset summary:")
    print(f"  Total variants: {len(df):,}")
    exon_counts = df["experiment"].value_counts().sort_index()
    print(f"  Exon distribution:\n{exon_counts.to_string()}")
    print(f"  Score: {df['score'].min():.3f} to {df['score'].max():.3f}  "
          f"mean={df['score'].mean():.3f}")
    print(f"  Classes: {df['function_class'].value_counts().to_dict()}")
    print(f"{'='*55}")

    # --- Save ---
    keep_cols = [c for c in [
        "hgvs_nt", "hgvs_pro", "score",
        "aa_ref", "aa_alt", "aa_pos",
        "aGVGD.diff", "consequence", "experiment",
        "phyloP (mammalian)", "function_class",
        "chrom", "genomic_pos",
    ] if c in df.columns]
    df[keep_cols].to_csv(SGE_BRCA2_RAW, index=False)
    print(f"\nSaved -> {SGE_BRCA2_RAW}  ({len(df):,} rows)")


def reclass_only() -> None:
    """
    Reload the existing brca2_sge.csv and recompute function_class with
    wider thresholds (LOF=-2.5σ, FUNC=0.0σ relative to synonymous mean).
    Skips VEP/UCSC annotation entirely — use when only thresholds change.
    """
    if not SGE_BRCA2_RAW.exists():
        print(f"brca2_sge.csv not found: {SGE_BRCA2_RAW}")
        print("Run without --reclass-only first.")
        sys.exit(1)

    df = pd.read_csv(SGE_BRCA2_RAW)
    print(f"Loaded: {len(df):,} rows from {SGE_BRCA2_RAW}")

    syn_mask = (
        (df["consequence"] == "synonymous_variant")
        | (df.get("hgvs_pro", pd.Series("", index=df.index)).fillna("").str.contains(r"p\.=", regex=False))
    )
    print(f"Synonymous variants: {syn_mask.sum():,}")
    df["function_class"] = assign_function_class(df["score"], syn_mask)

    print(f"New class distribution: {df['function_class'].value_counts().to_dict()}")
    df.to_csv(SGE_BRCA2_RAW, index=False)
    print(f"Saved -> {SGE_BRCA2_RAW}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reclass-only", action="store_true",
                   help="Skip VEP/UCSC; just recompute function_class from existing CSV.")
    args = p.parse_args()
    if args.reclass_only:
        reclass_only()
    else:
        main()
