import polars as pl
import sys
import config as cfg
from pathlib import Path

# CONFIGURATION

sys.path.append(str(Path(__file__).resolve().parent))

RAW_PRICE_FILE = cfg.DATA_DIR / "pp-complete.csv"
RAW_EPC_FILE = cfg.DATA_DIR / "certificates.csv"

OUTPUT_PRICE = cfg.RAW_PRICE_FILE
OUTPUT_EPC = cfg.RAW_EPC_FILE

TARGET_CITY = cfg.CURRENT_CITY


def filter_comparison_city():
    """
    Extracts transaction and EPC data for a specific control city (e.g., Leeds)
    from the massive raw UK datasets.
    """
    print(f"Starting data extraction for target city: {TARGET_CITY}")

    # 1. PRICE PAID DATA EXTRACTION
    if not RAW_PRICE_FILE.exists():
        print(f"Raw price file not found: {RAW_PRICE_FILE}")
        return

    print("Scanning raw Price Paid Data...")

    # Define column names based on HM Land Registry documentation
    # The raw CSV typically lacks headers.
    ppd_cols = [
        "id", "price", "date", "postcode", "property_type", "old_new",
        "duration", "paon", "saon", "street", "locality", "town",
        "district", "county", "ppd_category", "record_status"
    ]

    try:
        # Use LazyFrame for memory efficiency
        q_price = (
            pl.scan_csv(RAW_PRICE_FILE, has_header=False, new_columns=ppd_cols)
            .filter(pl.col("town") == TARGET_CITY)
            .with_columns(pl.col("date").str.to_datetime())
            # Filter for recent years to ensure relevance and reduce file size
            .filter(pl.col("date").dt.year() >= 2018)
        )

        df_price = q_price.collect()

        if df_price.height == 0:
            print(f"No price records found for {TARGET_CITY}. Check spelling.")
        else:
            print(f"Found {df_price.height:,} transaction records for {TARGET_CITY}.")
            df_price.write_parquet(OUTPUT_PRICE)
            print(f"Saved price data to: {OUTPUT_PRICE}")

    except Exception as e:
        print(f"Failed to process Price Data: {e}")

    # 2. EPC DATA EXTRACTION
    if not RAW_EPC_FILE.exists():
        print(f"Raw EPC file not found: {RAW_EPC_FILE}")
        return

    print("Scanning raw EPC Data...")

    try:
        # Scan EPC data. Note: DLUHC data usually uses 'POSTTOWN' for the city column.
        q_epc = (
            pl.scan_csv(RAW_EPC_FILE, ignore_errors=True)
            .filter(pl.col("POSTTOWN").str.to_uppercase() == TARGET_CITY)
        )

        df_epc = q_epc.collect()

        if df_epc.height == 0:
            print(f"No EPC records found for {TARGET_CITY}. Check column names (e.g. POSTTOWN).")
        else:
            print(f"Found {df_epc.height:,} EPC records for {TARGET_CITY}.")
            df_epc.write_parquet(OUTPUT_EPC)
            print(f"Saved EPC data to: {OUTPUT_EPC}")

    except Exception as e:
        print(f"[ERROR] Failed to process EPC Data: {e}")


if __name__ == "__main__":
    filter_comparison_city()