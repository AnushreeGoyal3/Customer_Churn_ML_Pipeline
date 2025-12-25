import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/random_forest.joblib")


df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load(MODEL_PATH)
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, preds))



🔍 WHY RANDOM FOREST < LOGISTIC REGRESSION (HERE)

You got:

Logistic Regression ROC-AUC ≈ 0.833

Random Forest ROC-AUC ≈ 0.823

This happens often in tabular datasets like Telco Churn.

Key reasons (INTERVIEW GOLD):

Feature encoding

We used LabelEncoder on categoricals

Random Forest treats encoded categories as ordinal, which is misleading

Dataset size

~7k rows → RF can overfit

Logistic Regression generalizes better on smaller tabular data

Hyperparameters not tuned

RF needs tuning (depth, min samples)

LR is already near-optimal

👉 Important insight: More complex ≠ better.

🧠 EXACT INTERVIEW ANSWER (MEMORIZE)

“Although Random Forest is more expressive, Logistic Regression performed better here due to the nature of encoded categorical variables and limited dataset size. This highlights the importance of baseline models and data representation.”



