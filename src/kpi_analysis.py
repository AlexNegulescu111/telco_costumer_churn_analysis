
def compute_kpis(df):
    import pandas as pd
    df.columns = [col.lower() for col in df.columns]

    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'] = df['total_charges'].fillna(0.0)

    total_customers = df.shape[0]
    churned_customers = df[df['churn'] == 'Yes'].shape[0]
    churn_rate = churned_customers / total_customers

    average_tenure = df['tenure'].mean()
    average_monthly_charges = df['monthly_charges'].mean()
    average_total_charges = df['total_charges'].mean()

    return {
        "Total Customers": total_customers,
        "Churned Customers": churned_customers,
        "Churn Rate (%)": round(churn_rate * 100, 2),
        "Average Tenure (months)": round(average_tenure, 2),
        "Average Monthly Charges ($)": round(average_monthly_charges, 2),
        "Average Total Charges ($)": round(average_total_charges, 2),
    }
