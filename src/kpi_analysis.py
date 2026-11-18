import pandas as pd


def compute_kpis(df):
    """Business KPIs for a (possibly filtered) cleaned dataset. Does not modify df."""
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0)

    total_customers = int(df.shape[0])
    churned_customers = int((df["churn"] == "Yes").sum())
    if total_customers == 0:
        churn_rate = average_tenure = average_monthly_charges = average_total_charges = 0.0
    else:
        churn_rate = churned_customers / total_customers
        average_tenure = df["tenure"].mean()
        average_monthly_charges = df["monthly_charges"].mean()
        average_total_charges = df["total_charges"].mean()

    return {
        "Total Customers": total_customers,
        "Churned Customers": churned_customers,
        "Churn Rate (%)": round(churn_rate * 100, 2),
        "Average Tenure (months)": round(float(average_tenure), 2),
        "Average Monthly Charges ($)": round(float(average_monthly_charges), 2),
        "Average Total Charges ($)": round(float(average_total_charges), 2),
    }
