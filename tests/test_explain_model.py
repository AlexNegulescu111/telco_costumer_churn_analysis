import numpy as np
import pytest

from build_model import fit_params, make_pipeline
from explain_model import explain_prediction

CAT, NUM = ["contract"], ["tenure"]


def _logit(p):
    return np.log(p / (1 - p))


@pytest.mark.parametrize(
    "model_type, params",
    [("logreg", {"C": 1.0}), ("xgb", {"n_estimators": 20, "max_depth": 2}), ("catboost", {"iterations": 30})],
)
def test_explanation_covers_original_features(synthetic_churn, model_type, params):
    x, y = synthetic_churn
    pipeline = make_pipeline(model_type, params, CAT, NUM).fit(x, y, **fit_params(model_type, CAT))
    contributions = explain_prediction(pipeline, model_type, x.iloc[[0]], CAT, NUM, background=x)
    assert sorted(contributions.index) == ["contract", "tenure"]
    # sorted by absolute contribution
    assert contributions.abs().is_monotonic_decreasing


def test_linear_explanation_is_additive(synthetic_churn):
    x, y = synthetic_churn
    pipeline = make_pipeline("logreg", {"C": 1.0}, CAT, NUM).fit(x, y)
    model, pre = pipeline[-1], pipeline[:-1]
    base = model.intercept_[0] + (model.coef_[0] * pre.transform(x).mean().to_numpy()).sum()
    for i in range(5):
        row = x.iloc[[i]]
        total = explain_prediction(pipeline, "logreg", row, CAT, NUM, background=x).sum()
        assert total + base == pytest.approx(_logit(pipeline.predict_proba(row)[0, 1]))


def test_linear_model_needs_background(synthetic_churn):
    x, y = synthetic_churn
    pipeline = make_pipeline("logreg", {"C": 1.0}, CAT, NUM).fit(x, y)
    with pytest.raises(ValueError):
        explain_prediction(pipeline, "logreg", x.iloc[[0]], CAT, NUM)
