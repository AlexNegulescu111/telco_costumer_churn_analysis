import pandas as pd


def rename_columns(df):
    column_dict = {
        "customerid": "customer_id",
        "gender": "gender",
        "seniorcitizen": "senior_citizen",
        "partner": "partner",
        "dependents": "dependents",
        "tenure": "tenure",
        "phoneservice": "phone_service",
        "multiplelines": "multiple_lines",
        "internetservice": "internet_service",
        "onlinesecurity": "online_security",
        "onlinebackup": "online_backup",
        "deviceprotection": "device_protection",
        "techsupport": "tech_support",
        "streamingtv": "streaming_tv",
        "streamingmovies": "streaming_movies",
        "contract": "contract",
        "paperlessbilling": "paperless_billing",
        "paymentmethod": "payment_method",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
        "churn": "churn",
    }
    df = df.rename(columns=column_dict)
    return df


def fix_totalcharges(df):
    """Convert total_charges to numbers.

    Blank values belong to brand-new customers (tenure 0) who have not been billed yet, so they become 0.
    Rows with a missing value despite a non-zero tenure are really missing and are dropped.
    Returns a new DataFrame; the input is not modified.
    """
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    new_customers = df["total_charges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "total_charges"] = 0.0
    return df.dropna(subset=["total_charges"])
