import sys
import shap
import config as cfg
import polars as pl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from pathlib import Path

# CONFIGURATION
sys.path.append(str(Path(__file__).parent))
INPUT_FILE = cfg.MODEL_READY_FILE
MODEL_PATH = cfg.MODEL_PATH
FIGURES_DIR = cfg.FIGURES_DIR

def explain_model_predictions():
    """
    Generates SHAP (SHapley Additive exPlanations) values to interpret model decisions.

    This pipeline produces two key outputs:
    1. Summary Plot: Global feature importance overview.
    2. Dependence Plot: Isolates the marginal contribution of Energy Ratings to price.
    """
    print("Initializing SHAP explanation pipeline...")

    # 1. Validation and Setup
    if not INPUT_FILE.exists() or not MODEL_PATH.exists():
        print(f"Required files not found.\nInput: {INPUT_FILE}\nModel: {MODEL_PATH}")
        return

    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir()

    # 2. Data Loading & Alignment
    # We must replicate the exact feature engineering steps used during training
    # to ensure the feature space matches the model's expectation.
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)

    # Extract Postcode District (e.g., 'SW1A' from 'SW1A 1AA')
    df = df.with_columns(
        pl.col("postcode").str.split(" ").list.first().alias("postcode_district")
    )

    # Define the exact feature set used in the CatBoost model
    feature_cols = [
        "TOTAL_FLOOR_AREA",
        "energy_rating_rank",
        "transaction_year",
        "property_type",
        "old_new",
        "town",
        "postcode_district"
    ]

    # 3. Sampling
    # SHAP TreeExplainer is computationally intensive (`O(TLD^2)` complexity).
    # A random sample of 5,000 instances provides a statistically significant
    # approximation of the global distribution without excessive runtime.
    print("Sampling data (N=5000) for efficient computation...")
    X_sample = df.select(feature_cols).sample(5000, seed=42).to_pandas()

    # 4. Model Loading
    print("Loading trained CatBoost Regressor...")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    # 5. SHAP Value Calculation
    print("Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 6. Visualization: Global Feature Importance
    print("Generating SHAP Summary Plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x / 1000)}k' if x != 0 else '0'))

    plt.title(f"Feature Impact on House Prices - {cfg.CURRENT_CITY} (SHAP Summary)", fontsize=14)
    plt.xlabel("SHAP Value (Impact on Price in GBP)", fontsize=12)
    plt.tight_layout()

    summary_plot_path = cfg.FIGURE_PATH_SUMMARY
    plt.savefig(summary_plot_path, dpi=300)
    print(f"[SUCCESS] Cleaned summary plot saved to: {summary_plot_path}")
    plt.close()

    # 7. Visualization: Green Premium Analysis (Dependence Plot)
    # This plot isolates the effect of 'energy_rating_rank' on the predicted price.
    # We disable interaction_index to view the clean, marginal effect of the rating.
    print("Generating Green Premium Dependence Plot...")

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        "energy_rating_rank",
        shap_values,
        X_sample,
        interaction_index=None,
        show=False,
        alpha=0.5
    )

    # Customizing the plot for report readability
    plt.title("Marginal Effect of Energy Rating on Property Value", fontsize=12)
    plt.ylabel("SHAP Value (Impact on Price in GBP)", fontsize=10)
    plt.xlabel("EPC Rating Rank (1=G, 7=A)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    premium_plot_path = cfg.FIGURE_PATH_CURVE
    plt.savefig(premium_plot_path, dpi=300)
    print(f"Green Premium curve saved to: {premium_plot_path}")
    plt.close()

    print("\nPipeline complete. Visualizations are available in the 'figures' directory.")


if __name__ == "__main__":
    explain_model_predictions()