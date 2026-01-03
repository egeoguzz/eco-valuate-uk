import polars as pl
import sys
import config as cfg
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

MERGED_FILE = cfg.MERGED_FILE


def inspect_merged_data():
    print("LOADING MERGED DATASET...")

    if not MERGED_FILE.exists():
        print(f"ERROR: {MERGED_FILE} not found.")
        return

    df = pl.read_parquet(MERGED_FILE)

    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    print("-" * 50)

    # 1. COLUMN CHECK
    print("AVAILABLE COLUMNS:")
    print(df.columns)
    print("-" * 50)

    # 2. REALITY CHECK (Sample Data)
    # We select key columns to make it readable in the terminal
    # Seeing 'Price' next to 'Energy Rating' is the core of this project.
    sample_cols = [
        "price",
        "date",
        "postcode",
        "CURRENT_ENERGY_RATING",
        "TOTAL_FLOOR_AREA",
        "PROPERTY_TYPE",
        "clean_addr_price"  # The address we used for matching
    ]

    print("SAMPLE DATA (Random 10 Rows):")
    # set_tbl_formatting helps to display full columns without truncation if needed
    with pl.Config(tbl_rows=10, fmt_str_lengths=30):
        print(df.select(sample_cols).sample(10))


if __name__ == "__main__":
    inspect_merged_data()