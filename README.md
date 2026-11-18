# Telco Customer Churn Analysis & Prediction

[![CI](https://github.com/AlexNegulescu111/telco_costumer_churn_analysis/actions/workflows/main.yml/badge.svg)](https://github.com/AlexNegulescu111/telco_costumer_churn_analysis/actions/workflows/main.yml)

This project focuses on analyzing customer churn in the telecom sector and predicting the likelihood of churn using machine learning models. It combines **business analysis**, **EDA**, **feature engineering**, and **model interpretability**, with an interactive **Streamlit dashboard** for BI and predictions.

---

## Business Objectives

- Analyze customer behavior and churn patterns
- Calculate key business KPIs (e.g., churn rate, average revenue)
- Build interpretable ML models to predict churn risk
- Provide an interactive dashboard for business stakeholders

---

## Project Structure

```
telco_costumer_churn_analysis/
├── data/                    # CSV input + saved KPIs
├── models/                  # Trained models, encoders, best params
├── plots/                   # Visualizations generated from EDA
├── src/                     # Modular Python source code
├── dashboard/               # Streamlit dashboard
├── notebooks/               # Optional: Jupyter notebooks (EDA etc.)
├── run_pipeline.py          # End-to-end training pipeline
├── requirements.txt
└── README.md
```

---

## Key KPIs (based on dataset)

| Metric                     | Value        |
|---------------------------|--------------|
| Total Customers           | 7043         |
| Churned Customers         | 1869         |
| Churn Rate                | 26.54%       |
| Avg Tenure (months)       | 32.37        |
| Avg Monthly Charges ($)   | 64.76        |
| Avg Total Charges ($)     | 2279.73      |

---

## Machine Learning Models

Implemented and compared: **Logistic Regression**, **XGBoost** and **CatBoost**.

### Methodology
- **Stratified 80/20 train/test split** at the start. The test set is used only once, for the final report.
- **Feature selection on the training set only**: chi-squared test (categorical) and Welch t-test (numerical), p < 0.05. `customer_id` is always excluded.
- **One scikit-learn `Pipeline` per model** (`ColumnTransformer` + estimator), fitted inside every CV fold:
  - Logistic Regression: one-hot encoding + standard scaling
  - XGBoost: one-hot encoding
  - CatBoost: native categorical handling
- **Hyperparameter tuning with Optuna** (seeded TPE sampler) using stratified 5-fold cross-validation on train, optimising **PR-AUC** (average precision).
- **Decision threshold** chosen to maximise F1 on out-of-fold training predictions, not the default 0.5.
- **Model selection** by cross-validated PR-AUC, never by test score.
- Metrics: ROC-AUC, PR-AUC, accuracy, precision, recall, F1 and confusion matrix, compared with a baseline that predicts the churn rate.
- Interpretability with **SHAP** (`plots/shap_importance.png`).

### Results

Test set (1,409 customers, never used for tuning, threshold or model selection):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.636 | 0.537 | 0.738 | 0.622 |
| XGBoost | 0.848 | 0.664 | 0.552 | 0.743 | 0.633 |
| **CatBoost** (selected) | 0.845 | 0.661 | 0.532 | 0.773 | 0.630 |
| Baseline (churn rate) | 0.500 | 0.265 | – | – | – |

CatBoost has the best cross-validated PR-AUC (0.671), so it is the saved model. On the test set the three models are within about 0.01 of each other. The selected model catches **77% of churners**, and about half of the customers it flags actually churn.

The strongest churn drivers (SHAP) are **contract type**, **internet service**, **tenure** and **tech support**.

Outputs: `models/churn_model.joblib` (the full pipeline, which accepts a raw customer row), `models/model_meta.json` (selected model, threshold, features), `models/metrics.json` (CV and test metrics), `models/feature_selection.json` and `plots/model_evaluation.png` (ROC curves, PR curves, confusion matrices).

---

## Dashboard (Streamlit)

Run the dashboard:

```bash
streamlit run dashboard/dashboard.py
```

Includes:
- KPIs live
- Interactive filtering (Contract, InternetService, etc.)
- Customer churn visualizations
- Churn prediction for a new customer (probability + risk label using the tuned threshold)

---

## Run Pipeline

```bash
python run_pipeline.py            # uses the tuned hyperparameters saved in models/best_params_*.json
python run_pipeline.py --tune     # re-runs Optuna tuning (30 trials per model, about 10-15 min)
```

- Cleans the data, computes and saves the KPIs
- Splits train/test (stratified) and selects features on train
- Tunes (or loads) hyperparameters, cross-validates and chooses the threshold for each model
- Selects the best model by CV PR-AUC and reports all models and the baseline on the test set
- Saves the best pipeline, metrics, the evaluation plots and the SHAP plot

---

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Source

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Original CSV: `data/telco_costumers.csv`

---

## CI/CD Workflow

GitHub Actions workflow runs:

- Code style checks (`flake8`, config in `.flake8`)
- End-to-end pipeline execution (Python 3.12, using the tuned hyperparameters saved in `models/*.json`)

Run the same checks locally:

```bash
flake8 src/ dashboard/ run_pipeline.py
```

---

## Future Improvements

- Add SHAP plot per prediction in dashboard
- Integrate live retraining via Streamlit
- Add customer segmentation or clustering
- Deploy dashboard online (Streamlit Cloud)

---

## Contact

Created by Alexandru NEGULESCU — looking for entry-level roles in **Data analysis/ Data Science / ML Engineering**  
Let's connect on [LinkedIn](https://www.linkedin.com/in/alexandru-negulescu-555a8323b) or check my [GitHub](https://github.com/AlexNegulescu111)
