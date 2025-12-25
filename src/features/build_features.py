import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


INPUT_DATA_PATH = Path("data/processed/cleaned_churn.csv")
OUTPUT_DATA_PATH = Path("data/processed/features.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}")
    return pd.read_csv(path)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = categorical_cols.drop("customerID", errors="ignore")

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["customerID"], errors="ignore")


def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Features saved to {path}")


def main():
    df = load_data(INPUT_DATA_PATH)
    df = encode_target(df)
    df = encode_categoricals(df)
    df = drop_unused_columns(df)
    save_features(df, OUTPUT_DATA_PATH)


if __name__ == "__main__":
    main()
