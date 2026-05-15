"""
Download Findlay et al. 2018 BRCA1 saturation genome editing (SGE) data.

Findlay GM et al. (2018) "Accurate classification of BRCA1 variants with saturation
genome editing." Nature 562, 217-222. DOI: 10.1038/s41586-018-0461-z

The dataset contains functional scores for 3,893 BRCA1 SNVs covering the RING domain
(exons 2-5) and BRCT domain (exons 15-23).

Score sign convention:
  LOW  score = loss of function (pathogenic-like)
  HIGH score = functional      (benign-like)

Primary source: NCBI GEO accession GSE117159 (complete dataset, all 13 exons)
Fallback:       MAVE-DB (Exon 2 only)
Saved to:       data/raw/findlay_brca1_sge.csv
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import SGE_BRCA1_RAW

# GEO accession for Findlay 2018 (complete dataset)
GEO_ACCESSION = "GSE117159"
GEO_FILE      = "GSE117159_Supplementary_Table_1.xlsx"
GEO_FTP_BASE  = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/GSE117159/suppl"

# MAVE-DB Exon 2 fallback (only covers 1 of 13 exons)
MAVEDB_API   = "https://api.mavedb.org/api/v1"
FINDLAY_URNS = [
    "urn:mavedb:00000097-b-2",   # Exon 2 Rep 2 (most recent)
    "urn:mavedb:00000097-a-2",   # Exon 2 Rep 1 (most recent)
]


# ---------------------------------------------------------------------------
# Primary: GEO download
# ---------------------------------------------------------------------------

def try_geo_download() -> pd.DataFrame | None:
    """Download the complete Findlay 2018 dataset from NCBI GEO."""
    urls = [
        f"{GEO_FTP_BASE}/{GEO_FILE}.gz",
        f"{GEO_FTP_BASE}/{GEO_FILE}",
        f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={GEO_ACCESSION}&format=file&file={GEO_FILE}.gz",
        f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={GEO_ACCESSION}&format=file&file={GEO_FILE}",
    ]

    for url in urls:
        print(f"  Trying GEO: {url}")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code != 200:
                print(f"    -> HTTP {resp.status_code}")
                continue

            content = b"".join(resp.iter_content(chunk_size=65536))
            print(f"    -> Downloaded {len(content):,} bytes")

            # Try to parse — may be gzipped Excel
            for compression in (None, "gzip"):
                try:
                    buf = io.BytesIO(content)
                    if compression == "gzip":
                        import gzip
                        buf = io.BytesIO(gzip.decompress(content))
                    # The GEO Excel has 4 header rows; actual column names are on row 3 (0-indexed)
                    df = pd.read_excel(buf, header=3, engine="openpyxl")
                    print(f"    -> Parsed Excel: {df.shape}  cols={list(df.columns[:8])}")
                    return df
                except Exception:
                    continue

        except requests.RequestException as e:
            print(f"    -> Request failed: {e}")

    return None


# ---------------------------------------------------------------------------
# Fallback: MAVE-DB (Exon 2 only)
# ---------------------------------------------------------------------------

def try_mavedb_fallback() -> pd.DataFrame | None:
    """Download Exon 2 data from MAVE-DB as a partial fallback."""
    frames = []
    for urn in FINDLAY_URNS:
        url = f"{MAVEDB_API}/score-sets/{urn}/scores"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                # Fetch target sequence for coordinate mapping
                meta_resp = requests.get(
                    f"{MAVEDB_API}/score-sets/{urn}", timeout=30
                )
                if meta_resp.ok:
                    meta = meta_resp.json()
                    tg = meta.get("targetGenes", [{}])
                    if tg:
                        df["target_seq"] = tg[0].get("targetSequence", {}).get("sequence", "")
                        df["exon_title"] = meta.get("title", "")
                df["source_urn"] = urn
                frames.append(df)
                print(f"  MAVE-DB {urn}: {len(df)} variants")
        except Exception as e:
            print(f"  MAVE-DB {urn}: {e}")
        time.sleep(0.4)

    return pd.concat(frames, ignore_index=True) if frames else None


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from either the GEO Excel or MAVE-DB format."""
    aliases = {
        # HGVS nucleotide identifier
        "hgvs_nt": ["transcript_variant", "hgvs_nt", "hgvs_base", "nucleotide", "variant"],
        # Functional score
        "score": ["function.score.mean", "score", "function_score", "func_score"],
        # Functional class (LOF / FUNC / INTERM)
        "function_class": ["func.class", "function_class", "class", "bin",
                           "call", "functional_class", "classification"],
        # ClinVar status column from GEO data
        "clinvar_simple": ["clinvar_simple", "clinvar"],
        # Transcript position (integer cDNA coordinate)
        "transcript_position": ["transcript_position"],
        # Alleles on transcript
        "transcript_ref": ["transcript_ref"],
        "transcript_alt": ["transcript_alt"],
    }
    rename = {}
    for canonical, options in aliases.items():
        for opt in options:
            if opt in df.columns and canonical not in df.columns:
                rename[opt] = canonical
                break
    if rename:
        df = df.rename(columns=rename)

    # Drop MAVE-DB special rows
    if "hgvs_nt" in df.columns:
        df = df[~df["hgvs_nt"].isin(["_wt", "_sy", "_stop"])].copy()

    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nDataset summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    if "hgvs_nt" in df.columns:
        print(f"  hgvs_nt sample: {df['hgvs_nt'].iloc[0]}")
    if "score" in df.columns:
        print(f"  Score range: {df['score'].min():.3f} to {df['score'].max():.3f}")
        print(f"  Score mean:  {df['score'].mean():.3f}  std={df['score'].std():.3f}")
    if "function_class" in df.columns:
        print(f"  Classes:\n{df['function_class'].value_counts().to_string()}")


def print_manual_instructions() -> None:
    print(f"""
  ----------------------------------------------------------------
  MANUAL DOWNLOAD INSTRUCTIONS
  ----------------------------------------------------------------
  The complete Findlay 2018 dataset is available from NCBI GEO:

    Accession: {GEO_ACCESSION}
    URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_ACCESSION}

  Download the file: {GEO_FILE}
  Save it to: data/raw/{GEO_FILE}
  Then re-run this script — it will detect and convert the Excel file.

  Alternatively, the Nature supplementary data (Excel) is at:
    https://www.nature.com/articles/s41586-018-0461-z#Sec22
  ----------------------------------------------------------------
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Check if manually downloaded Excel already exists
    excel_path = SGE_BRCA1_RAW.parent / GEO_FILE
    if excel_path.exists():
        print(f"Found manually downloaded Excel: {excel_path}")
        df = pd.read_excel(excel_path, header=3, engine="openpyxl")
        print(f"Loaded: {df.shape}  cols={list(df.columns[:8])}")
        df = normalise(df)
        print_summary(df)
        df.to_csv(SGE_BRCA1_RAW, index=False)
        print(f"\nSaved -> {SGE_BRCA1_RAW}")
        return

    print("Downloading Findlay 2018 BRCA1 SGE data...\n")

    # 1. Try GEO (complete dataset)
    print("[1/2] Attempting GEO download (complete, 3,893 variants)...")
    df = try_geo_download()

    # 2. Fall back to MAVE-DB (Exon 2 only, 312 variants)
    if df is None:
        print("\n[2/2] GEO download failed. Falling back to MAVE-DB (Exon 2 only)...")
        df = try_mavedb_fallback()

    if df is None:
        print("\nAll automatic downloads failed.")
        print_manual_instructions()
        sys.exit(1)

    df = normalise(df)
    df = df.drop_duplicates(subset=["hgvs_nt"]) if "hgvs_nt" in df.columns else df
    df = df.dropna(subset=["score"]) if "score" in df.columns else df

    print_summary(df)

    SGE_BRCA1_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SGE_BRCA1_RAW, index=False)
    print(f"\nSaved -> {SGE_BRCA1_RAW}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
