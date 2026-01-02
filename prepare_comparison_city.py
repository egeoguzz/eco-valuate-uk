import polars as pl
from pathlib import Path

# CONFIGURATION
DATA_DIR = Path("data")

# Input Files (Raw Data)
# Ensure these match your actual raw file names in the data directory
RAW_PRICE_FILE = DATA_DIR / "pp-complete.csv"
RAW_EPC_FILE = DATA_DIR / "certificates.csv"

# Target City for Control Group Analysis
# Must be uppercase to match Land Registry standards (e.g., 'LEEDS', 'MANCHESTER', 'LIVERPOOL')
TARGET_CITY = "LEEDS"

# Output Files (Distinct filenames to prevent overwriting London data)
OUTPUT_PRICE = DATA_DIR / f"price_paid_{TARGET_CITY.lower()}.parquet"
OUTPUT_EPC = DATA_DIR / f"epc_{TARGET_CITY.lower()}.parquet"


def filter_comparison_city():
    """
    Extracts transaction and EPC data for a specific control city (e.g., Leeds)
    from the massive raw UK datasets.
    """
    print(f"[INFO] Starting data extraction for target city: {TARGET_CITY}")

    # 1. PRICE PAID DATA EXTRACTION
    if not RAW_PRICE_FILE.exists():
        print(f"[ERROR] Raw price file not found: {RAW_PRICE_FILE}")
        return

    print("[INFO] Scanning raw Price Paid Data...")

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
            print(f"[WARNING] No price records found for {TARGET_CITY}. Check spelling.")
        else:
            print(f"[INFO] Found {df_price.height:,} transaction records for {TARGET_CITY}.")
            df_price.write_parquet(OUTPUT_PRICE)
            print(f"[SUCCESS] Saved price data to: {OUTPUT_PRICE}")

    except Exception as e:
        print(f"[ERROR] Failed to process Price Data: {e}")

    # 2. EPC DATA EXTRACTION
    if not RAW_EPC_FILE.exists():
        print(f"[ERROR] Raw EPC file not found: {RAW_EPC_FILE}")
        return

    print("[INFO] Scanning raw EPC Data...")

    try:
        # Scan EPC data. Note: DLUHC data usually uses 'POSTTOWN' for the city column.
        q_epc = (
            pl.scan_csv(RAW_EPC_FILE, ignore_errors=True)
            .filter(pl.col("POSTTOWN").str.to_uppercase() == TARGET_CITY)
        )

        df_epc = q_epc.collect()

        if df_epc.height == 0:
            print(f"[WARNING] No EPC records found for {TARGET_CITY}. Check column names (e.g. POSTTOWN).")
        else:
            print(f"[INFO] Found {df_epc.height:,} EPC records for {TARGET_CITY}.")
            df_epc.write_parquet(OUTPUT_EPC)
            print(f"[SUCCESS] Saved EPC data to: {OUTPUT_EPC}")

    except Exception as e:
        print(f"[ERROR] Failed to process EPC Data: {e}")


if __name__ == "__main__":
    filter_comparison_city()