"""
Baseline comparison: ABBI SGE regressor vs standard in silico predictors.
Computes ROC AUC (LOF vs FUNC) on held-out test exons for both BRCA1 and BRCA2.

Baselines: CADD, PolyPhen-2, SIFT, phyloP (mammalian), Grantham (aGVGD.diff)

BRCA1: all scores are pre-loaded from Findlay 2018 supplementary (no API calls needed).
BRCA2: SIFT/PolyPhen fetched via Ensembl VEP REST API; CADD via CADD REST API.
       Scores are cached to brca2_baselines_cache.csv on first run.

Usage:
  python src/baselines.py               # fetch BRCA2 annotations then compute
  python src/baselines.py --cached      # skip API calls, use existing cache file
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BRCA2_SGE_SPLITS_DIR, RESULTS_DIR, SGE_SPLITS_DIR

VEP_URL     = "https://rest.ensembl.org/vep/human/hgvs"
CADD_URL    = "https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_PATH  = BRCA2_SGE_SPLITS_DIR / "brca2_baselines_cache.csv"

# ---------------------------------------------------------------------------
# Ensembl VEP  (SIFT + PolyPhen)
# ---------------------------------------------------------------------------

def _canonical_tc(tcs: list) -> dict:
    if not tcs:
        return {}
    for tc in tcs:
        if tc.get("canonical"):
            return tc
    return tcs[0]


def vep_batch_scores(hgvs_list: list[str], batch_size: int = 200) -> dict:
    """
    POST to VEP with sift=1&polyphen=1.
    Returns {hgvs: {"sift": float|None, "polyphen": float|None,
                    "allele_string": str, "pos": int, "chrom": str}}
    """
    results = {}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    url     = f"{VEP_URL}?canonical=1&numbers=1&sift=1&polyphen=1"
    total_batches = (len(hgvs_list) - 1) // batch_size + 1

    for i in range(0, len(hgvs_list), batch_size):
        batch = hgvs_list[i : i + batch_size]
        print(f"  VEP batch {i // batch_size + 1}/{total_batches}  ({len(batch)} variants)...")
        data = []
        for attempt in range(3):
            try:
                resp = requests.post(url, json={"hgvs_notations": batch},
                                     headers=headers, timeout=90)
                if resp.status_code == 429:
                    print("    Rate-limited; waiting 30 s...")
                    time.sleep(30)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"    VEP failed after 3 attempts: {exc}")
                else:
                    time.sleep(5)

        for entry in data:
            hgvs_in = entry.get("input", "")
            tc      = _canonical_tc(entry.get("transcript_consequences", []))
            results[hgvs_in] = {
                "sift":         tc.get("sift_score"),
                "polyphen":     tc.get("polyphen_score"),
                "allele_string": entry.get("allele_string", ""),
                "pos":          entry.get("start"),
                "chrom":        str(entry.get("seq_region_name", "")),
            }
        time.sleep(1.5)

    return results


# ---------------------------------------------------------------------------
# CADD REST API
# ---------------------------------------------------------------------------

def _fetch_cadd_single(chrom: str, pos: int, ref: str, alt: str) -> float:
    chrom_clean = str(chrom).replace("chr", "").replace("Chr", "")
    url = f"{CADD_URL}/{chrom_clean}:{int(pos)}_{ref}_{alt}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            entries = data if isinstance(data, list) else [data]
            if entries:
                entry = entries[0]
                for key in ("PHRED", "phred", "CADD_PHRED"):
                    if key in entry:
                        return float(entry[key])
    except Exception:
        pass
    return np.nan


def fetch_cadd_batch(records: list[tuple], rate_delay: float = 0.3) -> dict:
    """
    records: list of (hgvs, chrom, pos, ref, alt)
    Returns {hgvs: cadd_phred}
    """
    results = {}
    n = len(records)
    for idx, (hgvs, chrom, pos, ref, alt) in enumerate(records):
        if (idx + 1) % 50 == 0:
            print(f"  CADD {idx + 1}/{n}...")
        if ref and alt and pos and chrom:
            results[hgvs] = _fetch_cadd_single(chrom, pos, ref, alt)
        else:
            results[hgvs] = np.nan
        time.sleep(rate_delay)
    return results


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def compute_auc(labels: pd.Series, scores: pd.Series, negate: bool = False) -> float:
    mask = pd.notna(scores)
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return np.nan
    y = labels[mask].values.astype(int)
    s = scores[mask].values.astype(float)
    auc = roc_auc_score(y, -s if negate else s)
    # Always report in LOF-positive orientation (AUC >= 0.5)
    return max(auc, 1.0 - auc)


def bootstrap_auc_ci(
    labels: pd.Series, scores: pd.Series,
    negate: bool = False, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    mask = pd.notna(scores)
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return (np.nan, np.nan)
    y = labels[mask].values.astype(int)
    s = (-scores[mask].values if negate else scores[mask].values).astype(float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        a = roc_auc_score(yb, sb)
        boot.append(max(a, 1.0 - a))
    boot = np.array(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _auc_with_ci(labels: pd.Series, scores: pd.Series, negate: bool = False) -> tuple:
    auc = compute_auc(labels, scores, negate=negate)
    lo, hi = bootstrap_auc_ci(labels, scores, negate=negate)
    return (auc, lo, hi)


# ---------------------------------------------------------------------------
# BRCA1
# ---------------------------------------------------------------------------

def brca1_aucs() -> dict:
    test_df = pd.read_csv(SGE_SPLITS_DIR / "sge_test.csv")
    pred_df = pd.read_csv(RESULTS_DIR / "sge_regressor_predictions.csv")[
        ["hgvs_nt", "pred_score"]
    ]
    df  = test_df.merge(pred_df, on="hgvs_nt", how="left")
    clf = df[df["function_class"].isin(["LOF", "FUNC"])].copy()
    clf["label"] = (clf["function_class"] == "LOF").astype(int)

    print(f"  BRCA1 test (LOF+FUNC): {len(clf):,}  "
          f"(LOF={( clf['label']==1).sum()}, FUNC={(clf['label']==0).sum()})")

    return {
        "ABBI":       _auc_with_ci(clf["label"], clf["pred_score"],          negate=True),
        "CADD":       _auc_with_ci(clf["label"], clf["CADD.score"]),
        "PolyPhen-2": _auc_with_ci(clf["label"], clf["polyphen2"]),
        "SIFT":       _auc_with_ci(clf["label"], clf["sift"],                negate=True),
        "phyloP":     _auc_with_ci(clf["label"], clf["phyloP (mammalian)"]),
        "Grantham":   _auc_with_ci(clf["label"], clf["aGVGD.diff"]),
    }


# ---------------------------------------------------------------------------
# BRCA2
# ---------------------------------------------------------------------------

def brca2_aucs(use_cache: bool = False) -> dict:
    test_df = pd.read_csv(BRCA2_SGE_SPLITS_DIR / "brca2_sge_test.csv")
    pred_df = pd.read_csv(RESULTS_DIR / "sge_regressor_predictions_brca2.csv")[
        ["hgvs_nt", "pred_score"]
    ]
    df = test_df.merge(pred_df, on="hgvs_nt", how="left")

    if use_cache and CACHE_PATH.exists():
        print(f"  Loading BRCA2 baseline cache: {CACHE_PATH}")
        cache = pd.read_csv(CACHE_PATH)[["hgvs_nt", "sift", "polyphen", "cadd_phred"]]
        df = df.merge(cache, on="hgvs_nt", how="left")
    else:
        print("  Fetching SIFT/PolyPhen via Ensembl VEP...")
        vep_res = vep_batch_scores(df["hgvs_nt"].tolist())

        df["sift"]     = df["hgvs_nt"].map(lambda h: vep_res.get(h, {}).get("sift"))
        df["polyphen"] = df["hgvs_nt"].map(lambda h: vep_res.get(h, {}).get("polyphen"))
        df["_allele"]  = df["hgvs_nt"].map(lambda h: vep_res.get(h, {}).get("allele_string", ""))
        df["_pos"]     = df["hgvs_nt"].map(lambda h: vep_res.get(h, {}).get("pos"))
        df["_chrom"]   = df["hgvs_nt"].map(lambda h: vep_res.get(h, {}).get("chrom", ""))

        n_sift = df["sift"].notna().sum()
        n_poly = df["polyphen"].notna().sum()
        print(f"  VEP coverage: SIFT {n_sift}/{len(df)}, PolyPhen {n_poly}/{len(df)}")

        print("  Fetching CADD scores (individual REST calls)...")
        records = []
        for _, row in df.iterrows():
            allele = str(row["_allele"])
            parts  = allele.split("/")
            ref, alt = (parts[0], parts[1]) if len(parts) == 2 else ("", "")
            records.append((row["hgvs_nt"], row["_chrom"], row["_pos"], ref, alt))

        cadd_map     = fetch_cadd_batch(records)
        df["cadd_phred"] = df["hgvs_nt"].map(cadd_map)

        n_cadd = df["cadd_phred"].notna().sum()
        print(f"  CADD coverage: {n_cadd}/{len(df)}")

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df[["hgvs_nt", "sift", "polyphen", "cadd_phred"]].to_csv(CACHE_PATH, index=False)
        print(f"  Cache saved -> {CACHE_PATH}")

        df.drop(columns=["_allele", "_pos", "_chrom"], inplace=True, errors="ignore")

    clf = df[df["function_class"].isin(["LOF", "FUNC"])].copy()
    clf["label"] = (clf["function_class"] == "LOF").astype(int)

    print(f"  BRCA2 test (LOF+FUNC): {len(clf):,}  "
          f"(LOF={(clf['label']==1).sum()}, FUNC={(clf['label']==0).sum()})")

    return {
        "ABBI":       _auc_with_ci(clf["label"], clf["pred_score"],          negate=True),
        "CADD":       _auc_with_ci(clf["label"], clf["cadd_phred"]),
        "PolyPhen-2": _auc_with_ci(clf["label"], clf["polyphen"]),
        "SIFT":       _auc_with_ci(clf["label"], clf["sift"],                negate=True),
        "phyloP":     _auc_with_ci(clf["label"], clf["phyloP (mammalian)"]),
        "Grantham":   _auc_with_ci(clf["label"], clf["aGVGD.diff"]),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_comparison(b1: dict, b2: dict) -> None:
    methods = ["ABBI", "CADD", "PolyPhen-2", "SIFT", "phyloP", "Grantham"]
    b1_vals = [b1.get(m, (np.nan,))[0] for m in methods]
    b2_vals = [b2.get(m, (np.nan,))[0] for m in methods]

    x, w = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w / 2, b1_vals, w, label="BRCA1", color="#2E86AB", alpha=0.88)
    bars2 = ax.bar(x + w / 2, b2_vals, w, label="BRCA2", color="#E76F51", alpha=0.88)

    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("ROC AUC  (LOF vs FUNC)", fontsize=10)
    ax.set_title("ABBI vs In Silico Predictors — Held-out Test Exons", fontsize=11)
    ax.legend(fontsize=10)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "baseline_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = ["ABBI", "CADD", "PolyPhen-2", "SIFT", "phyloP", "Grantham"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ABBI SGE regressor to standard in silico predictors."
    )
    parser.add_argument("--cached", action="store_true",
                        help="Use cached BRCA2 VEP/CADD annotations (skip API calls).")
    args = parser.parse_args()

    print("=" * 55)
    print("BRCA1 baselines")
    print("=" * 55)
    b1 = brca1_aucs()
    for m in METHODS:
        auc, lo, hi = b1.get(m, (np.nan, np.nan, np.nan))
        if np.isnan(auc):
            print(f"  {m:12s}: N/A (insufficient data)")
        else:
            print(f"  {m:12s}: {auc:.4f}  (95% CI {lo:.4f}--{hi:.4f})")

    print()
    print("=" * 55)
    print("BRCA2 baselines")
    print("=" * 55)
    b2 = brca2_aucs(use_cache=args.cached)
    for m in METHODS:
        auc, lo, hi = b2.get(m, (np.nan, np.nan, np.nan))
        if np.isnan(auc):
            print(f"  {m:12s}: N/A (insufficient data)")
        else:
            print(f"  {m:12s}: {auc:.4f}  (95% CI {lo:.4f}--{hi:.4f})")

    # --- Summary table ---
    rows = []
    for m in METHODS:
        auc1, lo1, hi1 = b1.get(m, (np.nan, np.nan, np.nan))
        auc2, lo2, hi2 = b2.get(m, (np.nan, np.nan, np.nan))
        rows.append({"Method": m,
                     "BRCA1_AUC": auc1, "BRCA1_CI_lo": lo1, "BRCA1_CI_hi": hi1,
                     "BRCA2_AUC": auc2, "BRCA2_CI_lo": lo2, "BRCA2_CI_hi": hi2})
    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "baseline_comparison.csv"
    table.to_csv(out_csv, index=False)

    print()
    print(f"{'Method':12s}  {'BRCA1 AUC (95% CI)':>26}  {'BRCA2 AUC (95% CI)':>26}")
    print("-" * 70)
    for _, row in table.iterrows():
        def fmt(auc, lo, hi):
            if np.isnan(auc):
                return "N/A"
            return f"{auc:.3f} ({lo:.3f}--{hi:.3f})"
        print(f"{row['Method']:12s}  {fmt(row['BRCA1_AUC'], row['BRCA1_CI_lo'], row['BRCA1_CI_hi']):>26}"
              f"  {fmt(row['BRCA2_AUC'], row['BRCA2_CI_lo'], row['BRCA2_CI_hi']):>26}")
    print(f"\nTable -> {out_csv}")

    print("\nGenerating comparison figure...")
    plot_comparison(b1, b2)


if __name__ == "__main__":
    main()
