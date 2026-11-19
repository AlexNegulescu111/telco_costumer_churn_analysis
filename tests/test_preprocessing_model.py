import numpy as np
import pandas as pd
import pytest

from preprocessing_model import build_preprocessor, encode_target, select_features


def test_encode_target():
    assert encode_target(pd.Series(["Yes", "No", "No"])).tolist() == [1, 0, 0]


def test_select_features_keeps_signal_and_drops_id_and_noise(synthetic_churn):
    x, y = synthetic_churn
    categorical, numerical, log = select_features(x, y, return_log=True)
    assert categorical == ["contract"]
    assert numerical == ["tenure"]
    assert "customer_id" not in [feature for feature, *_ in log]


@pytest.mark.parametrize("model_type", ["logreg", "xgb", "catboost"])
def test_preprocessor_drops_unselected_columns(synthetic_churn, model_type):
    x, _ = synthetic_churn
    out = build_preprocessor(["contract"], ["tenure"], model_type).fit_transform(x)
    assert "customer_id" not in out.columns and "noise_num" not in out.columns
    if model_type == "catboost":
        assert list(out.columns) == ["contract", "tenure"]
    else:
        assert "contract_Two year" in out.columns


def test_logreg_preprocessor_scales_and_ignores_unknown_categories(synthetic_churn):
    x, _ = synthetic_churn
    pre = build_preprocessor(["contract"], ["tenure"], "logreg").fit(x)
    out = pre.transform(x)
    assert out["tenure"].mean() == pytest.approx(0, abs=1e-9)
    assert out["tenure"].std(ddof=0) == pytest.approx(1)
    unseen = x.iloc[[0]].assign(contract="Three year")
    row = pre.transform(unseen)
    assert np.allclose(row.filter(like="contract_").to_numpy(), 0)


def test_unknown_model_type():
    with pytest.raises(ValueError):
        build_preprocessor([], [], "svm")
