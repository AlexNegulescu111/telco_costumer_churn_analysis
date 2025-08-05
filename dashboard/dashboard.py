import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
DATA_PATH = ROOT / "data" / "telco_costumers.csv"
KPI_PATH = ROOT / "data" / "kpi_summary.json"
MODEL_PATH = ROOT / "models" / "model_xgb.joblib"
PLOTS_PATH = ROOT / "plots"
FEATURES_PATH = ROOT / "models" / "xgb_features.json"


# === Load data ===
df = pd.read_csv(DATA_PATH)

# === Load KPIs ===
with open(KPI_PATH, "r") as f:
    kpis = json.load(f)

# === Streamlit page config ===
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# === Title ===
st.title("Telco Customer Churn Dashboard")
st.markdown("Understand customer behavior, churn drivers and predict churn risk using machine learning.")
st.markdown("---")

# === KPIs ===
st.subheader("Business Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Churn Rate (%)", f"{kpis['Churn Rate (%)']}%")
col2.metric("Avg Monthly Charges", f"${kpis['Average Monthly Charges ($)']}")
col3.metric("Total Customers", kpis["Total Customers"])

st.markdown("---")

# === Filters ===
st.subheader("Customer Filtering")
st.sidebar.header("Filters")

contract = st.sidebar.multiselect("Contract Type", df["Contract"].unique(), default=df["Contract"].unique())
internet = st.sidebar.multiselect("Internet Service", df["InternetService"].unique(), default=df["InternetService"].unique())

df_filtered = df[df["Contract"].isin(contract) & df["InternetService"].isin(internet)]

with st.expander("View filtered customer data"):
    st.dataframe(df_filtered)

st.markdown("---")

# === Visualizations ===
st.subheader("Churn Visualizations")

plot_files = [
    "pairplot_charges_tenure.png",
    "standard_services.png",
    "advanced_services.png",
    "billing_&_contract.png",
]

for plot_file in plot_files:
    plot_path = PLOTS_PATH / plot_file
    if plot_path.exists():
        st.image(str(plot_path), width=700)
    else:
        st.warning(f"Missing plot: {plot_file}")

st.markdown("---")

# === Prediction Section ===
st.subheader("Predict Churn for a New Customer")

with st.form("prediction_form"):
    st.markdown("**Enter customer details below:**")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 2500.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    input_data = pd.DataFrame([{
        "gender": gender,
        "seniorcitizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phone_service,
        "multiplelines": multiple_lines,
        "internetservice": internet_service,
        "onlinesecurity": online_security,
        "onlinebackup": online_backup,
        "deviceprotection": device_protection,
        "techsupport": tech_support,
        "streamingtv": streaming_tv,
        "streamingmovies": streaming_movies,
        "contract": contract,
        "paperlessbilling": paperless_billing,
        "paymentmethod": payment_method,
        "monthlycharges": monthly_charges,
        "totalcharges": total_charges
    }])
    input_data["churn"]="No"
    # Apply same preprocessing as in training
    from clean_data import rename_columns
    from preprocessing_model import encode_label

    input_data = rename_columns(input_data)
    input_data = input_data.drop(columns=["gender", "phone_service", "churn"])
    encoders = joblib.load(ROOT / "models" / "label_encoders.joblib")

    for col in input_data.select_dtypes(include="object").columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])
        else:
            st.warning(f"Encoder for column '{col}' not found!")


    with open(FEATURES_PATH, "r") as f:
        expected_features = json.load(f)
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Load model and predict
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(input_data)[0][1]  # probability "Yes"
    prediction = int(proba >= 0.5)
    label = "Yes" if prediction == 1 else "No"
  
    st.success(f"There is a probability of {proba:.2%} for this client to churn")
  
