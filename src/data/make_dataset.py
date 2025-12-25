import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_churn.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load raw churn dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> None:
    """Basic sanity checks."""
    required_columns = {"customerID", "Churn", "TotalCharges"}

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    print("Data validation passed")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning."""
    df = df.copy()

    # Convert TotalCharges to numeric (it has spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save processed dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Clean data saved to {path}")


def main():
    df = load_data(RAW_DATA_PATH)
    validate_data(df)
    df_clean = clean_data(df)
    save_data(df_clean, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
