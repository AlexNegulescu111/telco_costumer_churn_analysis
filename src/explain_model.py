import numpy as np
import pandas as pd


def _original_feature_map(preprocess, categorical, numerical, model_type):
    """Map every transformed column back to the original feature it came from
    (one-hot columns such as 'contract_Two year' belong to 'contract')."""
    mapping = {col: col for col in numerical}
    if model_type == "catboost":
        mapping.update({col: col for col in categorical})
    elif categorical:
        encoder = preprocess.named_transformers_["cat"]
        for col, categories in zip(categorical, encoder.categories_):
            for category in categories:
                mapping[f"{col}_{category}"] = col
    return mapping


def explain_prediction(pipeline, model_type, row, categorical, numerical, background=None):
    """SHAP contributions (log-odds) of each original feature for a single customer.

    pipeline: fitted Pipeline(preprocess, model); row: one-row raw DataFrame.
    background: raw DataFrame used as the reference population (needed for the linear model).
    Returns a Series indexed by feature, sorted by absolute contribution.
    """
    preprocess, model = pipeline[:-1], pipeline[-1]
    x = preprocess.transform(row)

    if model_type == "catboost":
        from catboost import Pool

        values = model.get_feature_importance(Pool(x, cat_features=categorical), type="ShapValues")[0, :-1]
    elif model_type == "xgb":
        import shap

        values = shap.TreeExplainer(model).shap_values(x)[0]
    elif model_type == "logreg":
        if background is None:
            raise ValueError("The linear model needs a background dataset to explain a prediction.")
        # exact SHAP for a linear model with independent features: coef * (x - E[x])
        reference = preprocess.transform(background).mean()
        values = (model.coef_[0] * (x.iloc[0] - reference)).to_numpy()
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    mapping = _original_feature_map(preprocess[0], categorical, numerical, model_type)
    contributions = pd.Series(np.asarray(values, dtype=float), index=x.columns).groupby(mapping).sum()
    return contributions.reindex(contributions.abs().sort_values(ascending=False).index)
