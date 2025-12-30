import polars as pl
import os

# CONFIGURATION
# ------------------------------------------------------------------------------
# Update these paths according to your local environment
PRICE_PAID_DATA_PATH = "pp-complete.csv"  # HM Land Registry Data
EPC_DATA_PATH = "certificates.csv"  # EPC Data


def inspect_data():
    """
    Loads the first few rows of the datasets to understand the structure,
    column names, and data types without loading the entire 11GB into RAM.
    """

    print("--- INSPECTING PRICE PAID DATA (HM Land Registry) ---")
    try:
        # The Price Paid data usually lacks a header row.
        # We load it without a header to check the raw columns.
        df_price = pl.read_csv(PRICE_PAID_DATA_PATH, has_header=False, n_rows=5)
        print(f"Successfully loaded snippet. Shape: {df_price.shape}")
        print(df_price)
    except Exception as e:
        print(f"Error loading Price Paid Data: {e}")

    print("\n" + "=" * 80 + "\n")

    print("--- INSPECTING EPC DATA (Energy Performance Certificates) ---")
    try:
        # EPC data typically contains headers.
        # We ignore errors to bypass potential bad lines in the first few rows.
        df_epc = pl.read_csv(EPC_DATA_PATH, n_rows=5, ignore_errors=True)
        print(f"Successfully loaded snippet. Shape: {df_epc.shape}")
        print("Column Names:")
        print(df_epc.columns)
        print("\nSample Data:")
        print(df_epc)
    except Exception as e:
        print(f"Error loading EPC Data: {e}")


if __name__ == "__main__":
    inspect_data()