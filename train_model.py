import polars as pl
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

# CONFIGURATION
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "final_model_ready.parquet"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "catboost_price_model.cbm"


def train_price_model():
    print("[INFO] Starting model training pipeline...")

    # Load Data
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print(f"[INFO] Loading dataset from {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)

    # Define Features (X) and Target (y)
    # Target: Price of the property
    y = df["price"].to_numpy()

    # Feature Engineering on the fly: Extract Postcode District (e.g., 'SW1A' from 'SW1A 1AA')
    # This helps the model generalize better than using the full unique postcode.
    # We take the part before the space.
    df = df.with_columns(
        pl.col("postcode").str.split(" ").list.first().alias("postcode_district")
    )

    # Selected features for the valuation model
    feature_cols = [
        "TOTAL_FLOOR_AREA",
        "energy_rating_rank",
        "transaction_year",
        "property_type",
        "old_new",
        "town",
        "postcode_district"
    ]

    # Convert to pandas/numpy for Scikit-Learn/CatBoost compatibility
    X = df.select(feature_cols).to_pandas()

    # Identify Categorical Features for CatBoost
    # CatBoost requires specific indices for categorical columns (text data)
    cat_features_indices = np.where(X.dtypes == object)[0]

    print(f"[INFO] Features Selected: {feature_cols}")
    print(f"[INFO] Categorical Feature Indices: {cat_features_indices}")

    # Train/Test Split
    print("[INFO] Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 4. Initialize and Train CatBoost Regressor
    print("[INFO] Initializing CatBoost Regressor...")
    print("[INFO] Training started. This may take a few minutes...")

    model = CatBoostRegressor(
        iterations=1000,  # Total number of trees
        learning_rate=0.1,  # Step size shrinkage used in update to prevents overfitting
        depth=8,  # Depth of the tree (6-10 is standard)
        loss_function='RMSE',  # Root Mean Squared Error optimization
        eval_metric='R2',  # We track R2 score during training
        random_seed=42,
        verbose=100,  # Log progress every 100 iterations
        allow_writing_files=False
    )

    # Fit the model
    model.fit(
        X_train, y_train,
        cat_features=cat_features_indices,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    # Model Evaluation
    print("\n[INFO] --- MODEL EVALUATION RESULTS ---")
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"[RESULT] R2 Score: {r2:.4f}")
    print(f"[RESULT] Mean Absolute Error (MAE): GBP {mae:,.0f}")
    print(f"[RESULT] Root Mean Squared Error (RMSE): GBP {rmse:,.0f}")

    # Save Model
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()

    model.save_model(str(MODEL_PATH))
    print(f"[INFO] Model saved successfully to: {MODEL_PATH}")

    # 7. Feature Importance Analysis
    print("\n[INFO] Top 3 Most Influential Features:")
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)[::-1]

    for i in sorted_idx[:3]:
        print(f"   - {feature_cols[i]}: {importance[i]:.2f}%")


if __name__ == "__main__":
    train_price_model()