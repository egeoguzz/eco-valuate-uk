import polars as pl
from pathlib import Path

# CONFIGURATION
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "final_model_ready.parquet"


def perform_sanity_check():
    print("[INFO] Starting Sanity Check on Raw Data...")

    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE} not found.")
        return

    # Load the actual data used for modeling
    df = pl.read_parquet(INPUT_FILE)

    print(f"[INFO] Dataset loaded. Total Rows: {df.shape[0]:,}\n")

    # CHECK 1: The "Heritage Paradox" Proof
    # We look at the Median Price for each Energy Rating.
    # Hypothesis: If 'Brown Premium' exists, low ratings (F/G) should have high prices.
    print("--- CHECK 1: Median Price by Energy Rating ---")
    print("(Expectation: G and F should not be the cheapest if Heritage Paradox exists)")

    stats = (
        df.group_by("CURRENT_ENERGY_RATING")
        .agg([
            pl.col("price").median().alias("median_price"),
            pl.col("price").mean().alias("mean_price"),
            pl.len().alias("count")
        ])
        .sort("CURRENT_ENERGY_RATING")
    )

    # Formatting for readable output
    with pl.Config(tbl_rows=10, fmt_str_lengths=50):
        print(stats)
    print("-" * 50 + "\n")

    # CHECK 2: Who are the "Energy Poor" but "Cash Rich"?
    # We filter for very expensive houses (> £1 Million) with bad energy ratings (F or G).
    # We want to see WHERE they are.
    print("--- CHECK 2: High Value (>£1M) but Low Efficiency (F/G) ---")
    print("(Expectation: Looking for prime locations like Kensington, Chelsea, Westminster)")

    rich_but_drafty = (
        df.filter(
            (pl.col("price") > 1_000_000) &
            (pl.col("CURRENT_ENERGY_RATING").is_in(["F", "G"]))
        )
        .group_by("town")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )

    print(rich_but_drafty)
    print("-" * 50 + "\n")

    # CHECK 3: Who are the "High Efficiency" (A/B) homes?
    # We filter for top efficiency. Are they in cheap or expensive areas?
    print("--- CHECK 3: Top Efficiency (A/B) Locations ---")
    print("(Expectation: Likely newer, less central/historic areas)")

    efficient_homes = (
        df.filter(pl.col("CURRENT_ENERGY_RATING").is_in(["A", "B"]))
        .group_by("town")
        .agg([
            pl.len().alias("count"),
            pl.col("price").median().alias("median_price")
        ])
        .sort("count", descending=True)
        .head(10)
    )

    print(efficient_homes)
    print("-" * 50)


if __name__ == "__main__":
    perform_sanity_check()