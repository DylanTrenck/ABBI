from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CLINVAR_BRCA1 = RAW_DIR / "clinvar_BRCA1.txt"
CLINVAR_BRCA2 = RAW_DIR / "clinvar_BRCA2.txt"
BRCA1_REF = RAW_DIR / "brca1_reference.gb"
BRCA2_REF = RAW_DIR / "brca2_reference.gb"

CLINVAR_CLEAN_PATH = PROCESSED_DIR / "clinvar_clean.csv"
SEQUENCES_PATH = PROCESSED_DIR / "sequences.npy"
X_CLIN_PATH = PROCESSED_DIR / "X_clin.npy"          # full 21-feature matrix
X_CLIN_SEQ_PATH = PROCESSED_DIR / "X_clin_seq.npy"  # 12-feature matrix (mol_csq dropped)
X_KMER_PATH = PROCESSED_DIR / "X_kmer.npy"
Y_PATH = PROCESSED_DIR / "y.npy"

# --- NCBI reference accessions ---
BRCA1_ACCESSION = "NM_007294"
BRCA2_ACCESSION = "NM_000059"

# --- Sequence features ---
SEQ_WINDOW = 100        # bp context window around each variant position
KMER_SIZES = [3, 4, 5]

# --- Model architecture ---
DNABERT2_MODEL = "zhihan1996/DNABERT-2-117M"  # DNABERT-2 model for sequence embeddings
EMBED_DIM = 128
DROPOUT = 0.2
NUM_ATTN_HEADS = 1

# --- Training ---
SEED = 42
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
PATIENCE = 10           # early stopping on val AUC-ROC

# --- Data splits ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
