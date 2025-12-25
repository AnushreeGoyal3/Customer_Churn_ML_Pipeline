import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")
OUTPUT_DIR = Path("reports/shap")


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()

    print(" SHAP summary plot saved")


if __name__ == "__main__":
    main()
