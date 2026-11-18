import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
# identifiers carry no signal and must never reach a model
ID_COLUMNS = ["customer_id"]


def encode_target(y):
    """Map the 'Yes'/'No' churn labels to 1/0."""
    return (y == "Yes").astype(int)


def select_features(x_train, y_train, alpha=0.05, return_log=False):
    """Keep the features statistically related to churn.

    Categorical features: chi-squared test of independence.
    Numerical features: Welch t-test between churners and non-churners.

    Must be called on the training split only, so that the test set does not
    influence which features are kept.

    Returns (categorical_features, numerical_features[, log]).
    """
    x = x_train.drop(columns=[c for c in ID_COLUMNS if c in x_train.columns])
    y = pd.Series(y_train, index=x.index)

    categorical, numerical, log = [], [], []
    for col in x.select_dtypes(include="object").columns:
        _, p, _, _ = chi2_contingency(pd.crosstab(x[col], y))
        log.append((col, "chi2", p, p < alpha))
        if p < alpha:
            categorical.append(col)

    for col in x.select_dtypes(include="number").columns:
        _, p = ttest_ind(x.loc[y == 1, col], x.loc[y == 0, col], equal_var=False)
        log.append((col, "ttest", p, p < alpha))
        if p < alpha:
            numerical.append(col)

    if return_log:
        return categorical, numerical, log
    return categorical, numerical


def build_preprocessor(categorical, numerical, model_type):
    """Column transformer fitted inside the model pipeline (no leakage across CV folds).

    - logreg: one-hot encoding + standard scaling of numerical features
    - xgb: one-hot encoding, numerical features passed through
    - catboost: categorical columns passed through as strings (CatBoost encodes them natively)
    Columns that are not selected (e.g. customer_id, gender) are dropped here,
    so the fitted pipeline accepts a raw customer row.
    """
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if model_type == "logreg":
        transformers = [("cat", one_hot, categorical), ("num", StandardScaler(), numerical)]
    elif model_type == "xgb":
        transformers = [("cat", one_hot, categorical), ("num", "passthrough", numerical)]
    elif model_type == "catboost":
        transformers = [("cat", "passthrough", categorical), ("num", "passthrough", numerical)]
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")
