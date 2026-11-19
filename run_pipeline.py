"""End-to-end training pipeline.

1. Load and clean the data, save business KPIs.
2. Stratified train/test split. The test set is used only once, for the final report.
3. Feature selection (chi2 / t-test) on the training set only.
4. For each model: tune hyperparameters with Optuna + stratified 5-fold CV on train
   (or load the saved ones), choose the decision threshold on out-of-fold predictions.
5. Pick the best model by cross-validated PR-AUC (never by test score).
6. Report every model and a baseline on the test set, save plots, metrics and the best pipeline.

Usage:
    python run_pipeline.py            # use saved hyperparameters when available
    python run_pipeline.py --tune     # re-run Optuna tuning for every model
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

# local modules from src/ (importable only after the sys.path update above)
from utils import KPI_PATH, MODELS_DIR, PLOTS_DIR, ROOT as PROJECT_ROOT  # noqa: E402
from utils import load_data, save_model, save_kpi, save_json  # noqa: E402
from utils import save_best_params, load_best_params  # noqa: E402
from clean_data import rename_columns, fix_totalcharges  # noqa: E402
from kpi_analysis import compute_kpis  # noqa: E402
from preprocessing_model import TARGET, encode_target, select_features  # noqa: E402
from build_model import MODEL_NAMES, MODEL_TYPES, fit_params, split_data, make_pipeline  # noqa: E402
from evaluate_model import (  # noqa: E402
    compute_metrics,
    find_best_threshold,
    metrics_table,
    oof_probabilities,
    plot_evaluation,
)
from tuning_model import cv_splitter, tune_params  # noqa: E402

SELECTION_METRIC = "pr_auc"
logger = logging.getLogger("pipeline")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the churn models.")
    parser.add_argument("--tune", action="store_true", help="re-run hyperparameter tuning for every model")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model (default: 30)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="where models/, plots/ and data/kpi_summary.json are written (default: the project root)",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=MODELS_DIR,
        help="where tuned hyperparameters are read from and saved to (default: models/)",
    )
    return parser.parse_args(argv)


def get_params(model_type, x_train, y_train, categorical, numerical, tune, n_trials, params_dir):
    name = f"best_params_{model_type}"
    if not tune and (params_dir / f"{name}.json").exists():
        return load_best_params(name, params_dir)
    logger.info("Tuning %s (%d trials, 5-fold CV on train)...", MODEL_NAMES[model_type], n_trials)
    params = tune_params(model_type, x_train, y_train, categorical, numerical, n_trials=n_trials)
    save_best_params(params, name, params_dir)
    return params


def save_shap_plot(pipeline, model_type, x_train, x_test, categorical, path):
    """Global feature importance (mean |SHAP|) of the selected model on the test set."""
    preprocess, model = pipeline[:-1], pipeline[-1]
    x_test_t = preprocess.transform(x_test)

    if model_type == "catboost":
        from catboost import Pool

        values = model.get_feature_importance(Pool(x_test_t, cat_features=categorical), type="ShapValues")[:, :-1]
    elif model_type == "xgb":
        values = shap.TreeExplainer(model).shap_values(x_test_t)
    else:
        values = shap.LinearExplainer(model, preprocess.transform(x_train)).shap_values(x_test_t)

    shap.summary_plot(values, x_test_t, plot_type="bar", show=False)
    plt.title(f"Feature importance (mean |SHAP|) - {MODEL_NAMES[model_type]}")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    models_dir = args.output_dir / MODELS_DIR.relative_to(PROJECT_ROOT)
    plots_dir = args.output_dir / PLOTS_DIR.relative_to(PROJECT_ROOT)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. load + clean, KPIs on the full dataset (descriptive statistics, no modelling)
    df = fix_totalcharges(rename_columns(load_data()))
    save_kpi(compute_kpis(df), args.output_dir / KPI_PATH.relative_to(PROJECT_ROOT))

    # 2. stratified split; x keeps all raw columns, the pipelines select their own features
    x = df.drop(columns=[TARGET])
    y = encode_target(df[TARGET])
    x_train, x_test, y_train, y_test = split_data(x, y)

    # 3. feature selection on the training data only
    categorical, numerical, selection_log = select_features(x_train, y_train, return_log=True)
    save_json(
        [{"feature": f, "test": t, "p_value": float(p), "kept": bool(k)} for f, t, p, k in selection_log],
        models_dir / "feature_selection.json",
    )
    dropped = [f for f, _, _, kept in selection_log if not kept]
    logger.info("Features kept: %d, dropped: %s (+ customer_id)", len(categorical) + len(numerical), dropped)

    # 4. tune / load params, cross-validate on train, fit, score on test
    cv = cv_splitter()
    cv_results, test_results, test_probas, thresholds, pipelines = {}, {}, {}, {}, {}
    for model_type in MODEL_TYPES:
        name = MODEL_NAMES[model_type]
        params = get_params(
            model_type, x_train, y_train, categorical, numerical, args.tune, args.n_trials, args.params_dir
        )
        pipeline = make_pipeline(model_type, params, categorical, numerical)

        extra_fit_params = fit_params(model_type, categorical)
        oof = oof_probabilities(pipeline, x_train, y_train, cv, extra_fit_params)
        thresholds[name] = find_best_threshold(y_train, oof)
        cv_results[name] = compute_metrics(y_train, oof, thresholds[name])

        pipeline.fit(x_train, y_train, **extra_fit_params)
        test_probas[name] = pipeline.predict_proba(x_test)[:, 1]
        test_results[name] = compute_metrics(y_test, test_probas[name], thresholds[name])
        pipelines[name] = (model_type, pipeline)

    # baseline: predicts the training churn rate for everyone (ROC-AUC 0.5, PR-AUC = churn rate)
    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    test_results["Baseline (churn rate)"] = compute_metrics(y_test, dummy.predict_proba(x_test)[:, 1])

    # 5. model selection on cross-validated score only
    best_name = max(cv_results, key=lambda n: cv_results[n][SELECTION_METRIC])
    best_type, best_pipeline = pipelines[best_name]

    logger.info("Cross-validation on train (5-fold, out-of-fold predictions):\n%s", metrics_table(cv_results))
    logger.info("Test set (held out, decision threshold chosen on train):\n%s", metrics_table(test_results))
    logger.info("Selected model: %s (best CV %s)", best_name, SELECTION_METRIC)

    # 6. save artefacts
    save_model(best_pipeline, models_dir / "churn_model.joblib")
    save_json(
        {
            "model": best_name,
            "model_type": best_type,
            "threshold": thresholds[best_name],
            "selection_metric": SELECTION_METRIC,
            "categorical_features": categorical,
            "numerical_features": numerical,
            "input_columns": x_train.columns.tolist(),
        },
        models_dir / "model_meta.json",
    )
    save_json(
        {"selected_model": best_name, "cross_validation": cv_results, "test": test_results},
        models_dir / "metrics.json",
    )

    plot_evaluation(y_test, test_probas, thresholds, plots_dir / "model_evaluation.png")
    save_shap_plot(best_pipeline, best_type, x_train, x_test, categorical, plots_dir / "shap_importance.png")
    logger.info("Artefacts saved to %s and %s", models_dir, plots_dir)
    logger.info("Test-set churn rate: %.3f", np.mean(y_test))
    return {"selected_model": best_name, "cross_validation": cv_results, "test": test_results}


if __name__ == "__main__":
    main()
