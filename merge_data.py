import polars as pl
from pathlib import Path

# CONFIGURATION
DATA_DIR = Path("data")
PRICE_INPUT = DATA_DIR / "price_paid_london_filtered.parquet"
EPC_INPUT = DATA_DIR / "epc_london_filtered.parquet"
OUTPUT_FILE = DATA_DIR / "merged_dataset.parquet"


def normalize_address_string(df: pl.LazyFrame, col_name: str, alias: str) -> pl.LazyFrame:
    """
    Standardizes address strings to improve matching rates.
    - Lowercase
    - Remove punctuation
    - Remove common prefixes like 'flat', 'apartment'
    """
    return df.with_columns(
        pl.col(col_name)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r"[^\w\s]", "")
        .str.replace_all(r"\b(flat|apartment|unit|suite)\b", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias(alias)
    )


def run_merge_pipeline():
    print("STARTING MERGE PIPELINE...")

    # 1. PREPARE EPC DATA
    print("Loading and cleaning EPC Data...")

    # Strategy: Sort by LMK_KEY (or date if available) and keep the most recent one per address.
    # For MVP, we group by POSTCODE + ADDRESS1 and keep the first one.

    q_epc = (
        pl.scan_parquet(EPC_INPUT)
        .with_columns([
            # Create a clean join key combining Postcode + Address
            (pl.col("POSTCODE").str.replace(" ", "")).alias("join_pcode"),
            pl.col("ADDRESS1").alias("raw_addr")
        ])
        # Apply normalization to address
        .pipe(normalize_address_string, "raw_addr", "clean_addr_epc")
        # Deduplicate: Keep one EPC per address (can be improved with date logic later)
        .unique(subset=["join_pcode", "clean_addr_epc"], keep="first")
    )

    # 2. PREPARE PRICE DATA
    print("Loading and cleaning Price Data...")

    q_price = (
        pl.scan_parquet(PRICE_INPUT)
        .with_columns([
            (pl.col("postcode").str.replace(" ", "")).alias("join_pcode"),

            # Construct full address from PAON and SAON
            pl.concat_str([
                pl.col("saon").fill_null(""),
                pl.col("paon").fill_null(""),
                pl.col("street").fill_null("")
            ], separator=" ").alias("full_address_raw")
        ])
        # Apply normalization
        .pipe(normalize_address_string, "full_address_raw", "clean_addr_price")
    )

    # 3. PERFORM JOIN
    print("Executing Merge (Left Join Price -> EPC)...")

    # We join on Postcode (exact) AND the Cleaned Address (exact)
    # Note: This is a strict match. We might lose some data, but the matches will be high quality.

    merged_df = (
        q_price.join(
            q_epc,
            left_on=["join_pcode", "clean_addr_price"],
            right_on=["join_pcode", "clean_addr_epc"],
            how="inner"  # Use 'inner' to keep only sold houses with EPC data
        )
        .collect()
    )

    row_count = merged_df.shape[0]
    print(f"MERGE COMPLETE. Final Dataset Rows: {row_count:,}")

    if row_count > 0:
        merged_df.write_parquet(OUTPUT_FILE)
        print(f"Saved to: {OUTPUT_FILE}")

        # Calculate Match Rate
        # We need total price rows to calc rate, but LazyFrame needs separate collection.
        # Assuming approx 847k from previous step.
        print(f"Approximate Match Rate: {(row_count / 846995) * 100:.2f}%")
    else:
        print("CRITICAL: Zero matches found. Check address normalization logic.")


if __name__ == "__main__":
    run_merge_pipeline()