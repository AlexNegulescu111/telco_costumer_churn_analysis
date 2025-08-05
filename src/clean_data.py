def rename_columns(df):
    import pandas as pd
    column_dict = {
        'customerid':'customer_id',
        'gender':'gender',
        'seniorcitizen':'senior_citizen',
        'partner':'partner',
        'dependents':'dependents',
        'tenure':'tenure',
        'phoneservice':'phone_service',
        'multiplelines':'multiple_lines',
        'internetservice':'internet_service',
        'onlinesecurity':'online_security',
        'onlinebackup':'online_backup',
        'deviceprotection':'device_protection',
        'techsupport':'tech_support',
        'streamingtv':'streaming_tv',
        'streamingmovies':'streaming_movies',
        'contract':'contract',
        'paperlessbilling':'paperless_billing',
       'paymentmethod':'payment_method',
       'monthlycharges':'monthly_charges',
       'totalcharges':'total_charges',
       'churn':'churn'
    }
    df = df.rename(columns = column_dict)
    return df


def fix_totalcharges(df):
    import pandas as pd
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    if df['total_charges'].isna().any():
        # if NaN values with tenure not 0 then delete
        if not (df.loc[df['total_charges'].isna(), 'tenure'] == 0).all():
            df.dropna(subset=['total_charges'], inplace=True)
        else:
            # else replace with 0
            df['total_charges'] = df['total_charges'].fillna(0.0)

    return df

