import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from utils import load_data
from utils import save_model
from utils import save_kpi
from utils import save_best_params, load_best_params
from clean_data import rename_columns
from clean_data import fix_totalcharges
from kpi_analysis import compute_kpis
from preprocessing_model import drop_irrelevant_columns, encode_label, encode_one_hot
from build_model import split_data, train_model, predict
from evaluate_model import collect_scores, compare_models
from tunning_model import tune_logistic_params, tune_xgb_params, tune_cat_params
import pandas as pd
import numpy as np
import shap
import json

# loading the data
df = load_data()

# Cleaning the data
df = rename_columns(df)
df = fix_totalcharges(df)
# identifying and saving the KPI's in a jason file at 'ROOT/data/kpi_summary.json'
kpis = compute_kpis(df)
save_kpi(kpis)

# naming the json files used for best params storing via optuna
name_log = "best_params_log"
name_xg = "best_params_xg"
name_cat = "best_params_cat"

# in eda.ipynb the columns 'costumer_id', 'gender', 'phone_service',
#  turned to be statistically irrelevant for churning (chi_2 test) so we'll remove them
#  from the training/testing data in the model 
df_model = drop_irrelevant_columns(df)
#--------------------------------------------LOGISTIC REGRESSION -----------------------------------------
# encoding the categorical for model training
df_logistic = encode_one_hot(df_model)
x_logistic = df_logistic.drop(columns=["churn"])
y_logistic = df_logistic["churn"]

# split the data into train/test -> 80/20
x_log_train, x_log_test, y_log_train, y_log_test = split_data(x_logistic, y_logistic)
#Tunning the model with the best parameters via optuna
# load the params if exist
# tune the model and save the params if not exist
param_path = ROOT / "models" / f"{name_log}.json"
if param_path.exists():
    best_params_log = load_best_params(name_log)
else:
    best_params_log = tune_logistic_params(x_log_train, y_log_train, x_log_test, y_log_test)
    save_best_params(best_params_log, name_log)
# train the model - logistic regression
model_log = train_model(x_log_train, y_log_train, model_type="logreg", **best_params_log)

y_log_pred = predict(model_log, x_log_test)
# collect the mtrics for the model
collect_scores(y_log_test, y_log_pred, "Logistic regression")

#--------------------------------------------XGBOOST -----------------------------------------


# encoding the categorical for model training - XGBoost.
df_xg = encode_label(df_model)

x_xg = df_xg.drop(columns=["churn"])
y_xg = df_xg["churn"]

# split the data into train/test -> 80/20
x_xg_train, x_xg_test, y_xg_train, y_xg_test = split_data(x_xg, y_xg)
#Tunning the model with the best parameters via optuna
# load the params if exist
# tune the model and save the params if not exist
param_path = ROOT / "models" / f"{name_xg}.json"
if param_path.exists():
    best_params_xg = load_best_params(name_xg)
else:
    best_params_xg = tune_xgb_params(x_xg_train, y_xg_train, x_xg_test, y_xg_test)
    save_best_params(best_params_xg, name_xg)
# train the model - XGBoost
model_xg = train_model(x_xg_train, y_xg_train, model_type="xgb", **best_params_xg)

y_xg_pred = predict(model_xg, x_xg_test)
# collect the mtrics for the model
collect_scores(y_xg_test, y_xg_pred, "XGBoost")

#--------------------------------------------CATBOOST -----------------------------------------


df_cat = df_model.copy()
df_cat["churn"] = (df_cat["churn"] == "Yes").astype(int)

X_cat = df_cat.drop(columns=["churn"])
y_cat = df_cat["churn"]

# split
x_cat_train, x_cat_test, y_cat_train, y_cat_test = split_data(X_cat, y_cat)
#Tunning the model with the best parameters via optuna
# load the params if exist
# tune the model and save the params if not exist
param_path = ROOT / "models" / f"{name_cat}.json"
if param_path.exists():
    best_params_cat = load_best_params(name_cat)
else:
    best_params_cat = tune_cat_params(x_cat_train, y_cat_train, x_cat_test, y_cat_test)
    save_best_params(best_params_cat, name_cat)

# train
model_cat = train_model(x_cat_train, y_cat_train, model_type="catboost", **best_params_cat)

# predict
y_cat_pred = predict(model_cat, x_cat_test)
# collect the mtrics for the model
collect_scores(y_cat_test, y_cat_pred, "CatBoost")

# compare the models used via metrics: precision, recall, f1_score, accuracy. Print the confusion matrix for each

scores = compare_models()
print(scores)
# save the best model/ XGBoost after comparison
model_path = ROOT / "models" / "model_xgb.joblib"
save_model(model_xg, model_path)
# Save feature names for later prediction
feature_names = x_xg_train.columns.tolist()
with open(ROOT / "models" / "xgb_features.json", "w") as f:
    json.dump(feature_names, f)

# 9. Interpretation of the model xgb model with shap

explainer = shap.Explainer(model_xg, x_xg_train)

# 2. Obține valorile SHAP pentru setul de test
shap_values = explainer(x_xg_test)

# 3. Vizualizare SHAP summary (plot de tip bar)
shap.summary_plot(shap_values, x_xg_test, plot_type="bar")
