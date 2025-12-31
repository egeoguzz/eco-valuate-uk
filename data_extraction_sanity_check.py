import polars as pl
from pathlib import Path

# CONFIGURATION
DATA_DIR = Path("data")
PRICE_FILE = DATA_DIR / "price_paid_london_filtered.parquet"
EPC_FILE = DATA_DIR / "epc_london_filtered.parquet"


def run_sanity_check():
    print("STARTING SANITY CHECK...")
    print("-" * 30)

    # ---------------------------------------------------------
    # 1. PRICE PAID DATA CHECK
    # ---------------------------------------------------------
    if not PRICE_FILE.exists():
        print(f"CRITICAL ERROR: {PRICE_FILE} not found!")
        return

    print(f"Checking: {PRICE_FILE.name}")
    try:
        df_price = pl.read_parquet(PRICE_FILE)

        # Metric 1: Volume
        row_count = df_price.shape[0]
        print(f"Total Rows: {row_count:,}")

        # Metric 2: Temporal Validity
        if row_count > 0:
            min_date = df_price["date"].min()
            max_date = df_price["date"].max()
            print(f"Date Range: {min_date} to {max_date}")

            # Metric 3: Geospatial Validity (Sample Towns)
            sample_towns = df_price["town"].unique().head(5).to_list()
            print(f"Sample Locations: {sample_towns}")

            # Verdict
            if row_count < 10000:
                print("WARNING: Row count is suspiciously low for London (2018+). Check filters.")
            elif min_date.year < 2018:
                print("WARNING: Found data from before 2018. Date filter might be broken.")
            else:
                print("STATUS: PRICE DATA LOOKS GOOD.")
        else:
            print("WARNING: Price Paid Data is empty.")

    except Exception as e:
        print(f"Error reading price data: {e}")

    print("\n" + "=" * 50 + "\n")

    # ---------------------------------------------------------
    # 2. EPC DATA CHECK
    # ---------------------------------------------------------
    if not EPC_FILE.exists():
        print(f"CRITICAL ERROR: {EPC_FILE} not found!")
        return

    print(f"Checking: {EPC_FILE.name}")
    try:
        df_epc = pl.read_parquet(EPC_FILE)

        # Metric 1: Volume
        epc_count = df_epc.shape[0]
        print(f"Total Rows: {epc_count:,}")

        # Metric 2: Critical Columns
        required_cols = ["CURRENT_ENERGY_RATING", "TOTAL_FLOOR_AREA", "POSTCODE"]
        missing_cols = [col for col in required_cols if col not in df_epc.columns]

        if missing_cols:
            print(f"ERROR: Missing critical columns: {missing_cols}")
        else:
            # Check distribution of Energy Ratings
            print("Energy Rating Distribution (Top 5):")
            print(df_epc["CURRENT_ENERGY_RATING"].value_counts().sort("count", descending=True).head(5))
            print("STATUS: EPC DATA LOOKS GOOD.")

    except Exception as e:
        print(f"Error reading EPC data: {e}")


if __name__ == "__main__":
    run_sanity_check()