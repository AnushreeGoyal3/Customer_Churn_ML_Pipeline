Project: Customer Churn Prediction — End-to-End ML Pipeline

Problem Statement

Customer churn directly impacts revenue and lifetime value. The objective of this project was to predict whether a customer is likely to churn so that retention actions (discounts, outreach, contract upgrades) can be triggered proactively.

The problem is binary classification with class imbalance, where churners are the minority class.

Dataset

Source: Kaggle Telco Customer Churn Dataset

Size: 7,000 customers

Target Variable: Churn (0 = No, 1 = Yes)

Feature Types:

Demographic (gender, senior citizen)

Service usage (internet service, streaming)

Contract & billing (tenure, monthly charges, contract type)

ML Pipeline Design (End-to-End)

The pipeline was structured to mirror production ML systems:

Data ingestion & validation

Feature engineering

Categorical encoding

Numerical scaling

Train–test split with stratification

Model training

Evaluation using business-aware metrics

Model interpretability using SHAP

Baseline Model — Logistic Regression
Why Logistic Regression?

Strong baseline for binary classification

Interpretable coefficients

Fast to train and easy to deploy

Model Performance
ROC-AUC = 0.833

Indicates the model ranks churners above non-churners 83% of the time

Anything above 0.80 is considered a production-viable baseline in churn prediction

Confusion Matrix
[[724 309]   → Non-churners
 [ 76 298]]  → Churners

Metric	Value
True Positives (Caught churners)	298
False Negatives (Missed churners)	76
Recall (Churn = 1)	80%
Precision (Churn = 1)	49%


Business Interpretation 

The model prioritizes high recall for churners

It deliberately accepts more false positives

Why this is desirable:

Missing a churner = lost revenue

False positive = marketing email (low cost)



I optimized for recall on the churn class since the business cost of missing a churner is significantly higher than a false positive. ROC-AUC was used as the primary evaluation metric due to class imbalance.

Model Upgrade — Random Forest
Motivation

Logistic Regression is linear and may miss complex relationships.

Why Random Forest? 

Random Forest captures nonlinear feature interactions and threshold effects that linear models cannot, while remaining robust to noise and still interpretable via feature importance.

Benefits:

Handles nonlinearities

Less sensitive to outliers

Built-in feature importance

Model Explainability — SHAP
Why Explainability?

In real-world ML systems, stakeholders must trust predictions.

SHAP was used to:

Explain global feature importance

Understand directional impact of features

Validate that the model aligns with domain knowledge

SHAP Insights

Contract type — month-to-month contracts increase churn risk

Tenure — longer tenure reduces churn probability

Monthly charges — higher charges increase churn likelihood

I used SHAP values to explain both global and individual predictions. Contract type, tenure, and monthly charges emerged as the strongest churn drivers, which aligned well with business intuition and increased trust in the model.

 Final Outcomes

 Production-ready ML pipeline

 Strong baseline ROC-AUC (0.83)

 Business-aligned evaluation strategy

 Interpretable predictions using SHAP

 Extensible to advanced models (XGBoost, LightGBM)

 Tech Stack

Languages: Python

Libraries: pandas, scikit-learn, SHAP

Tools: Git, GitHub, PowerShell

ML Concepts: Classification, Imbalanced Learning, Explainability, Model Evaluation
