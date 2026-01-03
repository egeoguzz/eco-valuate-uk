from pathlib import Path

# CONTROL PANEL
# "LONDON" or "LEEDS"
CURRENT_CITY = "LONDON"

ROOT_DIR = Path(__file__).resolve().parent.parent

# PATH DEFINITIONS

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

if CURRENT_CITY == "LEEDS":
    RAW_PRICE_FILE = DATA_DIR / "price_paid_leeds.parquet"
    RAW_EPC_FILE = DATA_DIR / "epc_leeds.parquet"
    MERGED_FILE = DATA_DIR / "merged_leeds.parquet"
    MODEL_READY_FILE = DATA_DIR / "final_model_ready_leeds.parquet"
    MODEL_PATH = MODEL_DIR / "catboost_leeds_model.cbm"
    FIGURE_PATH_CURVE = FIGURES_DIR / "green_premium_curve_leeds.png"
    FIGURE_PATH_SUMMARY = FIGURES_DIR / "shap_summary_leeds.png"
else:
    RAW_PRICE_FILE = DATA_DIR / "price_paid_london.parquet"
    RAW_EPC_FILE = DATA_DIR / "epc_london.parquet"
    MERGED_FILE = DATA_DIR / "merged_london.parquet"
    MODEL_READY_FILE = DATA_DIR / "final_model_ready_london.parquet"
    MODEL_PATH = MODEL_DIR / "catboost_london_model.cbm"
    FIGURE_PATH_CURVE = FIGURES_DIR / "green_premium_curve_london.png"
    FIGURE_PATH_SUMMARY = FIGURES_DIR / "shap_summary_london.png"

RANDOM_SEED = 42
TEST_SIZE = 0.2