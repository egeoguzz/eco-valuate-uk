from pathlib import Path

# CONTROL PANEL
# "LONDON" or "LEEDS"
CURRENT_CITY = "LONDON"

# PATH DEFINITIONS
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# DYNAMIC FILE MAPPING
if CURRENT_CITY == "LEEDS":
    print(f"Operating Mode: LEEDS (Control Group)")

    # 1. Inputs
    RAW_PRICE_FILE = DATA_DIR / "price_paid_leeds.parquet"
    RAW_EPC_FILE = DATA_DIR / "epc_leeds.parquet"

    # 2. Intermediates
    MERGED_FILE = DATA_DIR / "merged_leeds.parquet"
    MODEL_READY_FILE = DATA_DIR / "final_model_ready_leeds.parquet"

    # 3. Outputs (Model & FIGURES)
    MODEL_PATH = MODEL_DIR / "catboost_leeds_model.cbm"

    # PNG Names
    FIGURE_PATH_CURVE = FIGURES_DIR / "green_premium_curve_leeds.png"
    FIGURE_PATH_SUMMARY = FIGURES_DIR / "shap_summary_leeds.png"

else:
    # DEFAULT: LONDON
    print(f"Operating Mode: LONDON (Main Analysis)")

    # 1. Inputs
    RAW_PRICE_FILE = DATA_DIR / "price_paid_london.parquet"
    RAW_EPC_FILE = DATA_DIR / "epc_london.parquet"

    # 2. Intermediates
    MERGED_FILE = DATA_DIR / "merged_london.parquet"
    MODEL_READY_FILE = DATA_DIR / "final_model_ready_london.parquet"

    # 3. Outputs
    MODEL_PATH = MODEL_DIR / "catboost_london_model.cbm"

    # PNG Names
    FIGURE_PATH_CURVE = FIGURES_DIR / "green_premium_curve_london.png"
    FIGURE_PATH_SUMMARY = FIGURES_DIR / "shap_summary_london.png"

# ⚙SETTINGS
RANDOM_SEED = 42