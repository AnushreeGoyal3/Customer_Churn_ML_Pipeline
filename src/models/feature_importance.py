import pandas as pd
import joblib
from pathlib import Path


MODEL_PATH = Path("models/logistic_regression.joblib")
DATA_PATH = Path("data/processed/features.csv")
OUTPUT_PATH = Path("reports/feature_importance.csv")


model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn"])

importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

OUTPUT_PATH.parent.mkdir(exist_ok=True)
importance.to_csv(OUTPUT_PATH, index=False)

print("Feature importance saved")
