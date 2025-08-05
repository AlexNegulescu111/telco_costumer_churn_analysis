# Telco Customer Churn Analysis & Prediction

![CI/CD](https://github.com/AlexNegulescu111/telco_costumer_churn_analysis/actions/workflows/ci.yml/badge.svg)

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
customer_churn_project/
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

Implemented and compared:

- **Logistic Regression** (with Optuna tuning)
- **XGBoost** (selected final model)
- **CatBoost**

### Features:
- Irrelevant variables dropped based on Chi2 / t-test
- Categorical encoding:
  - OneHot for Logistic Regression
  - Label Encoding + saved encoders for XGBoost
- Hyperparameter tuning with Optuna
- Model evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
- Interpretability with **SHAP** values

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
- Churn prediction for a new customer (with probability)

---

## Run Pipeline

```bash
python run_pipeline.py
```

- Preprocesses data
- Computes and saves KPIs
- Trains all 3 models with best params
- Compares performance
- Saves best model (XGBoost)
- Generates SHAP explanation

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

- Code style checks (`flake8`)
- End-to-end pipeline execution

---

## Future Improvements

- Add SHAP plot per prediction in dashboard
- Integrate live retraining via Streamlit
- Add customer segmentation or clustering
- Deploy dashboard online (Streamlit Cloud)

---

## Contact

Created by [Your Name] — looking for entry-level roles in **Data analysis/ Data Science / ML Engineering**  
Let's connect on [LinkedIn](linkedin.com/in/alexandru-negulescu-555a8323b) or check my [GitHub](https://github.com/AlexNegulescu111)
