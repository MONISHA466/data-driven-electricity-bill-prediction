import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data_sample.csv"
MODEL_PATH = BASE_DIR / "bill_model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def preprocess(df: pd.DataFrame):
    # For this simple example we treat all numeric columns as features except target
    feature_cols = [
        "Units_Consumed",
        "Washing_Machine_Usage",
        "AC_Usage",
        "Heater_Usage",
        "Temperature",
    ]
    target_col = "Total_Bill"

    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, r2


def save_model(model, feature_cols):
    MODEL_PATH.write_bytes(b"")  # ensure file exists/overwrite
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
        },
        MODEL_PATH,
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_data(DATA_PATH)
    X, y, feature_cols = preprocess(df)
    model, mae, r2 = train_and_evaluate(X, y)
    save_model(model, feature_cols)

    print("Training completed.")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

