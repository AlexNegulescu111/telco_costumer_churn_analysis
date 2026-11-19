import pytest
from sklearn.base import clone

from build_model import fit_params, make_estimator, make_pipeline, split_data


def test_split_is_stratified(synthetic_churn):
    x, y = synthetic_churn
    _, _, y_train, y_test = split_data(x, y)
    assert len(y_test) == pytest.approx(0.2 * len(y), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_catboost_none_class_weights_are_dropped():
    model = make_estimator("catboost", {"auto_class_weights": "None", "iterations": 10})
    assert "auto_class_weights" not in model.get_params() or model.get_params()["auto_class_weights"] is None


def test_fit_params_only_for_catboost():
    assert fit_params("catboost", ["contract"]) == {"model__cat_features": ["contract"]}
    assert fit_params("xgb", ["contract"]) == {}


@pytest.mark.parametrize(
    "model_type, params",
    [
        ("logreg", {"C": 1.0, "l1_ratio": 0.0}),
        ("xgb", {"n_estimators": 20, "max_depth": 2}),
        ("catboost", {"iterations": 20, "depth": 2}),
    ],
)
def test_pipelines_fit_predict_and_clone(synthetic_churn, model_type, params):
    x, y = synthetic_churn
    pipeline = make_pipeline(model_type, params, ["contract"], ["tenure"])
    clone(pipeline)  # cross-validation clones the pipeline: this must not fail
    pipeline.fit(x, y, **fit_params(model_type, ["contract"]))
    proba = pipeline.predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)
    assert ((proba >= 0) & (proba <= 1)).all()
    # the model learned the signal: month-to-month customers get a higher risk
    month_to_month = (x["contract"] == "Month-to-month").to_numpy()
    assert proba[month_to_month].mean() > proba[~month_to_month].mean()


def test_unknown_model_type():
    with pytest.raises(ValueError):
        make_estimator("svm", {})
