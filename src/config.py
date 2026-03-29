"""Central configuration for the workplace injury claim modeling project."""

from pathlib import Path

# ----------------------------
# Reproducibility settings
# ----------------------------
RANDOM_STATE = 42
TF_SEED = 42

# ----------------------------
# Data paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_PATH = PROJECT_ROOT / "data" / "private" / "pre_model_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# ----------------------------
# Business-specific columns
# ----------------------------
TARGET_COLUMN = "CLAIM_AGG"

# Keep this list empty if you want all non-target columns used automatically.
FEATURE_COLUMNS = []

# ----------------------------
# Modeling parameters
# ----------------------------
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
BATCH_SIZE = 2048
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.50
HIDDEN_UNITS = 16

# Thresholds to compare from a business perspective
EVALUATION_THRESHOLDS = [0.50, 0.10, 0.01]

# ----------------------------
# Output filenames
# ----------------------------
SUMMARY_TABLE_FILE = RESULTS_DIR / "method_comparison.csv"
METRICS_JSON_FILE = RESULTS_DIR / "method_metrics.json"
