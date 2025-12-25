\# Customer Churn Prediction — End-to-End ML Pipeline



\## Problem

Predict customer churn to enable proactive retention strategies.



\## Dataset

IBM Telco Customer Churn dataset (~7k customers).



\## Pipeline

\- Data ingestion \& validation

\- Feature engineering

\- Model training (Logistic Regression)

\- Evaluation (ROC-AUC, precision, recall)

\- Explainability using SHAP



\## Results

\- ROC-AUC: \*\*0.83\*\*

\- Recall (Churn): \*\*80%\*\*

\- Key drivers: Contract type, tenure, monthly charges



\## Tech Stack

Python, pandas, scikit-learn, SHAP



\## How to Run

```bash

pip install -r requirements.txt

python src/data/make\_dataset.py

python src/features/build\_features.py

python src/models/train.py

python src/models/evaluate.py

python src/models/shap\_explainability.py



