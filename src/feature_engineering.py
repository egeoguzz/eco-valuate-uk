import polars as pl
from pathlib import Path

# CONFIGURATION
DATA_DIR = Path("data")
#INPUT_FILE = DATA_DIR / "merged_dataset.parquet"
#OUTPUT_FILE = DATA_DIR / "final_model_ready.parquet"

INPUT_FILE = DATA_DIR / "merged_leeds.parquet"
OUTPUT_FILE = DATA_DIR / "model_ready_leeds.parquet"


def perform_feature_engineering():
    """
    Transforms the raw merged dataset into a model-ready format.
    1. Temporal Feature Extraction: Allows the model to account for inflation/HPI.
    2. Unit Price Calculation: Price per sqm is a more comparable metric.
    3. Ordinal Encoding: Converts Energy Ratings (A-G) to numeric ranks (7-1).
    4. Outlier Removal: Filters extreme values to ensure model stability.
    """
    print("Starting feature engineering pipeline...")

    if not INPUT_FILE.exists():
        print(f"CRITICAL ERROR: {INPUT_FILE} not found. Please run the merge pipeline first.")
        return

    try:
        # Use LazyFrame for memory efficiency
        q = (
            pl.scan_parquet(INPUT_FILE)

            # 1. TEMPORAL FEATURES (Handling Inflation)
            # By providing Year and Month, the model can learn the 'Time Trend'
            # (i.e., inflation/HPI) independently from the 'Green Premium'.
            .with_columns([
                pl.col("date").dt.year().alias("transaction_year"),
                pl.col("date").dt.month().alias("transaction_month"),
                pl.col("date").dt.quarter().alias("transaction_quarter")
            ])

            # 2. FEATURE CREATION
            # Price per square meter calculation
            .with_columns(
                (pl.col("price") / pl.col("TOTAL_FLOOR_AREA")).alias("price_per_sqm")
            )

            # Encode Energy Rating: A (Best) -> 7, G (Worst) -> 1
            .with_columns(
                pl.col("CURRENT_ENERGY_RATING").replace({
                    "A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1
                }, default=None).cast(pl.Int32).alias("energy_rating_rank")
            )

            # 3. OUTLIER REMOVAL & CLEANING
            # Filtering unrealistic properties to prevent model skew.
            .filter(
                (pl.col("TOTAL_FLOOR_AREA") > 20) &  # Exclude tiny units (<20m2)
                (pl.col("TOTAL_FLOOR_AREA") < 500) &  # Exclude mega-mansions (>500m2)
                (pl.col("price") > 50_000) &  # Exclude derelict/auction properties
                (pl.col("price") < 5_000_000) &  # Exclude ultra-luxury segment
                (pl.col("energy_rating_rank").is_not_null())  # Ensure valid energy data
            )

            # Handle Missing Categorical Values
            .with_columns([
                pl.col("property_type").fill_null("Other"),
                pl.col("BUILT_FORM").fill_null("Unknown")
            ])
        )

        # Execute Pipeline
        df_final = q.collect()

        # 4. REPORTING
        print("Feature engineering complete.")
        print(f"Final Dataset Size: {df_final.shape[0]:,} rows")

        # Print a small sample to verify columns
        print("\nData Snippet (Temporal & Energy Features):")
        print(df_final.select([
            "transaction_year", "price", "CURRENT_ENERGY_RATING", "energy_rating_rank"
        ]).head(5))

        df_final.write_parquet(OUTPUT_FILE)
        print(f"\nModel-Ready Data Saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error during feature engineering: {e}")


if __name__ == "__main__":
    perform_feature_engineering()