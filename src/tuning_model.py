import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from build_model import fit_params, make_pipeline
from utils import RANDOM_STATE

# Optimise PR-AUC (average precision): threshold-independent and suited to an imbalanced target.
# The decision threshold is chosen afterwards (see evaluate_model.find_best_threshold).
SCORING = "average_precision"


def cv_splitter(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def suggest_params(trial, model_type, y_train):
    if model_type == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_type == "xgb":
        imbalance = float(np.sum(y_train == 0) / np.sum(y_train == 1))
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, imbalance),
        }
    if model_type == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["None", "Balanced"]),
        }
    raise ValueError(f"Model type '{model_type}' not supported.")


def tune_params(model_type, x_train, y_train, categorical, numerical, n_trials=30, n_splits=5):
    """Tune hyperparameters with Optuna using stratified K-fold CV on the training set only."""
    cv = cv_splitter(n_splits)

    def objective(trial):
        params = suggest_params(trial, model_type, y_train)
        pipeline = make_pipeline(model_type, params, categorical, numerical)
        scores = cross_val_score(
            pipeline, x_train, y_train, cv=cv, scoring=SCORING, params=fit_params(model_type, categorical)
        )
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
