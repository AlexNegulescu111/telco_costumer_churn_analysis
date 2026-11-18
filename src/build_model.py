from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing_model import build_preprocessor
from utils import RANDOM_STATE

MODEL_TYPES = ["logreg", "xgb", "catboost"]
MODEL_NAMES = {"logreg": "Logistic Regression", "xgb": "XGBoost", "catboost": "CatBoost"}


def split_data(x, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split: churn is imbalanced (~26.5%), keep the ratio in both parts."""
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def make_estimator(model_type, params):
    params = dict(params)
    if model_type == "logreg":
        defaults = {"solver": "saga", "max_iter": 5000, "class_weight": "balanced", "random_state": RANDOM_STATE}
        return LogisticRegression(**{**defaults, **params})
    if model_type == "xgb":
        defaults = {"eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1}
        return XGBClassifier(**{**defaults, **params})
    if model_type == "catboost":
        # "None" is stored as a string by Optuna; CatBoost expects the parameter to be absent
        if params.get("auto_class_weights") == "None":
            params.pop("auto_class_weights")
        defaults = {
            "verbose": 0,
            "allow_writing_files": False,
            "random_seed": RANDOM_STATE,
            "thread_count": -1,
        }
        return CatBoostClassifier(**{**defaults, **params})
    raise ValueError(f"Model type '{model_type}' not supported.")


def fit_params(model_type, categorical):
    """Extra fit() arguments: CatBoost receives the categorical columns at fit time
    (passing them to the constructor breaks sklearn.clone, used by cross-validation)."""
    return {"model__cat_features": categorical} if model_type == "catboost" else {}


def make_pipeline(model_type, params, categorical, numerical):
    """Full model: preprocessing + estimator. Fitting it on raw rows avoids train/serve skew."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(categorical, numerical, model_type)),
            ("model", make_estimator(model_type, params)),
        ]
    )
