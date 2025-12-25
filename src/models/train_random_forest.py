import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib


DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/random_forest.joblib")


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f" Random Forest ROC-AUC: {auc:.4f}")


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f" Model saved to {path}")


def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
