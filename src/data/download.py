import argparse
import sys
import time
from pathlib import Path

from Bio import Entrez, SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import BRCA1_ACCESSION, BRCA2_ACCESSION, BRCA1_REF, BRCA2_REF

ENTREZ_EMAIL = "dylan.trenck@gmail.com"


def fetch_reference(accession: str, dest: Path, retries: int = 3) -> None:
    if dest.exists():
        print(f"  {dest.name} already exists, skipping.")
        return

    for attempt in range(1, retries + 1):
        try:
            print(f"  Fetching {accession} (attempt {attempt})...")
            handle = Entrez.efetch(
                db="nucleotide",
                id=accession,
                rettype="gb",
                retmode="text",
            )
            record = SeqIO.read(handle, "genbank")
            handle.close()
            with open(dest, "w", encoding="utf-8") as fh:
                SeqIO.write(record, fh, "genbank")
            print(f"  Saved -> {dest}  ({len(record.seq):,} bp)")
            return
        except Exception as exc:
            print(f"  Attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(5)

    raise RuntimeError(f"Failed to fetch {accession} after {retries} attempts.")


def main(args: argparse.Namespace) -> None:
    Entrez.email = ENTREZ_EMAIL

    targets = [
        (BRCA1_ACCESSION, BRCA1_REF),
        (BRCA2_ACCESSION, BRCA2_REF),
    ]

    for accession, dest in targets:
        print(f"\n[{accession}]")
        if args.force and dest.exists():
            dest.unlink()
            print(f"  Removed existing {dest.name} (--force)")
        fetch_reference(accession, dest)

    print("\nDone. Reference files:")
    for _, dest in targets:
        size_kb = dest.stat().st_size / 1024
        print(f"  {dest}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch BRCA1/2 GenBank reference sequences from NCBI.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    main(parser.parse_args())
