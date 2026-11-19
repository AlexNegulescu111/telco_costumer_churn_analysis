"""End-to-end run of run_pipeline.py on the real data, with the saved hyperparameters (no tuning).
Artefacts go to a temporary directory, so the committed models and plots are not touched."""

import json

import run_pipeline


def test_pipeline_end_to_end(tmp_path):
    results = run_pipeline.main(["--output-dir", str(tmp_path)])

    for path in [
        "models/churn_model.joblib",
        "models/model_meta.json",
        "models/metrics.json",
        "models/feature_selection.json",
        "plots/model_evaluation.png",
        "plots/shap_importance.png",
        "data/kpi_summary.json",
    ]:
        assert (tmp_path / path).exists(), path

    meta = json.loads((tmp_path / "models/model_meta.json").read_text())
    assert meta["model"] == results["selected_model"]
    assert "customer_id" not in meta["categorical_features"] + meta["numerical_features"]
    assert 0 < meta["threshold"] < 1

    test = results["test"]
    baseline = test["Baseline (churn rate)"]
    for name in ["Logistic Regression", "XGBoost", "CatBoost"]:
        assert test[name]["roc_auc"] > 0.8
        assert test[name]["pr_auc"] > baseline["pr_auc"] + 0.3
