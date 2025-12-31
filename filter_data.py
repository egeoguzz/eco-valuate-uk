import polars as pl
import os
from pathlib import Path

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# Define paths relative to the project root
DATA_DIR = Path("data")
PRICE_RAW_PATH = DATA_DIR / "pp-complete.csv"
EPC_RAW_PATH = DATA_DIR / "certificates.csv"

# Output paths for optimized Parquet files
PRICE_PROCESSED_PATH = DATA_DIR / "price_paid_london_filtered.parquet"
EPC_PROCESSED_PATH = DATA_DIR / "epc_london_filtered.parquet"


def process_price_paid_data():
    """
    Ingests raw HM Land Registry Price Paid Data, applies filtering for
    London-based transactions from 2018 onwards, and saves to Parquet.
    """
    print(f"Starting ingestion: {PRICE_RAW_PATH}")

    # Define schema manually as the raw file lacks a header
    price_cols = [
        "id", "price", "date", "postcode", "property_type",
        "old_new", "duration", "paon", "saon", "street",
        "locality", "town", "district", "county", "ppd_cat", "status"
    ]

    try:
        # Use LazyFrame for memory-efficient processing
        q = (
            pl.scan_csv(PRICE_RAW_PATH, has_header=False, new_columns=price_cols)
            .with_columns(pl.col("date").str.to_datetime("%Y-%m-%d %H:%M"))

            # Filter 1: Temporal scope (2018 - Present)
            .filter(pl.col("date").dt.year() >= 2018)

            # Filter 2: Geospatial scope (Greater London area)
            .filter(
                (pl.col("town").str.contains("LONDON")) |
                (pl.col("district").str.contains("LONDON")) |
                (pl.col("county").str.contains("LONDON"))
            )

            # Select only relevant features for the valuation model
            .select([
                "price", "date", "postcode", "property_type",
                "old_new", "paon", "saon", "street", "town", "district"
            ])
        )

        # Execute the query
        df_price = q.collect()
        print(f"Price Paid Data processed. Rows: {df_price.shape[0]}")

        df_price.write_parquet(PRICE_PROCESSED_PATH)
        print(f"Saved to: {PRICE_PROCESSED_PATH}")

    except Exception as e:
        print(f"Failed to process Price Paid Data: {e}")


def process_epc_data():
    """
    Ingests raw EPC certificates, selects key energy efficiency metrics,
    and saves to Parquet format for downstream merging.
    """
    print(f"\nStarting ingestion: {EPC_RAW_PATH}")

    # Features required for Green Premium analysis
    target_cols = [
        "LMK_KEY", "ADDRESS1", "ADDRESS2", "POSTCODE",
        "CURRENT_ENERGY_RATING", "POTENTIAL_ENERGY_RATING",
        "TOTAL_FLOOR_AREA", "PROPERTY_TYPE", "BUILT_FORM",
        "CONSTRUCTION_AGE_BAND", "NUMBER_HABITABLE_ROOMS"
    ]

    try:
        q = (
            pl.scan_csv(EPC_RAW_PATH, ignore_errors=True)
            .select(target_cols)
            # Ensure valid geospatial identifiers
            .filter(pl.col("POSTCODE").is_not_null())
        )

        df_epc = q.collect()
        print(f"EPC Data processed. Rows: {df_epc.shape[0]}")

        df_epc.write_parquet(EPC_PROCESSED_PATH)
        print(f"Saved to: {EPC_PROCESSED_PATH}")

    except Exception as e:
        print(f"Failed to process EPC Data: {e}")


if __name__ == "__main__":
    # Ensure dependencies are met before execution
    # Requirement: pip install "polars[rtcompat]" for legacy CPU support if needed
    process_price_paid_data()
    process_epc_data()