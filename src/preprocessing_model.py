def encode_one_hot(df):

    import pandas as pd
    categorical_cols = df.select_dtypes(include=["object"]).columns.drop("churn")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded["churn"] = (df_encoded["churn"] == "Yes").astype(int)
    return df_encoded

def encode_label(df):

    from sklearn.preprocessing import LabelEncoder
    import joblib
    from utils import ROOT
    categorical_cols = df.select_dtypes(include=["object"]).columns.drop("churn")
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    df_encoded["churn"] = (df["churn"] == "Yes").astype(int)
    joblib.dump(encoders, ROOT / "models" / "label_encoders.joblib")
    return df_encoded

def drop_irrelevant_columns(df, return_log=False):
    import pandas as pd
    from scipy.stats import chi2_contingency, ttest_ind

    categorical_cols = df.select_dtypes(include=["object"]).columns.drop("churn")
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    relevant_cols = []
    log = []

    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df["churn"])
        chi_2, p, dof, _ = chi2_contingency(contingency)
        if p < 0.05:
            relevant_cols.append(col)
        log.append((col, 'chi2', p, p < 0.05))

    for col in numerical_cols:
        yes = df[df['churn'] == 'Yes'][col]
        no  = df[df['churn'] == 'No'][col]
        t_stat, p = ttest_ind(yes, no, equal_var=False)
        if p < 0.05:
            relevant_cols.append(col)
        log.append((col, 'ttest', p, p < 0.05))

    relevant_cols.append("churn")
    if return_log:
        return df[relevant_cols], log
    return df[relevant_cols]
