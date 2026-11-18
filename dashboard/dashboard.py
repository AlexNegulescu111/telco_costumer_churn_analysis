"""Streamlit dashboard: live churn KPIs and charts, model performance and churn prediction.

Run with:  streamlit run dashboard/dashboard.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

# local modules from src/ (importable only after the sys.path update above)
from clean_data import fix_totalcharges, rename_columns  # noqa: E402
from explain_model import explain_prediction  # noqa: E402
from kpi_analysis import compute_kpis  # noqa: E402
from utils import load_data  # noqa: E402

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
METRICS_PATH = ROOT / "models" / "metrics.json"
PLOTS_PATH = ROOT / "plots"
EDA_PLOTS = [
    "pairplot_charges_tenure.png",
    "personal_info.png",
    "standard_services.png",
    "advanced_services.png",
    "billing_&_contract.png",
]
MISSING_MODEL_HINT = "Train it first with `python run_pipeline.py`."
CHURN_COLORS = alt.Scale(domain=["No", "Yes"], range=["#4C78A8", "#E45756"])

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")


# ----------------------------------------------------------------------------- loading (cached)
@st.cache_data
def get_data():
    return fix_totalcharges(rename_columns(load_data()))


@st.cache_resource
def get_model():
    """Returns (pipeline, metadata) or (None, None) if the model has not been trained yet."""
    if not (MODEL_PATH.exists() and META_PATH.exists()):
        return None, None
    with open(META_PATH) as f:
        return joblib.load(MODEL_PATH), json.load(f)


@st.cache_data
def get_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


try:
    df = get_data()
except FileNotFoundError:
    st.error("Dataset not found: expected `data/telco_costumers.csv`.")
    st.stop()

model, meta = get_model()


# ----------------------------------------------------------------------------- helpers
def churn_rate_by(data, column, sort=None):
    """Bar chart of the churn rate (%) per category of `column`."""
    rates = (
        data.groupby(column, observed=True)["churn"]
        .agg(churn_rate=lambda s: (s == "Yes").mean() * 100, customers="size")
        .reset_index()
    )
    return (
        alt.Chart(rates)
        .mark_bar(color="#E45756")
        .encode(
            x=alt.X("churn_rate:Q", title="Churn rate (%)"),
            y=alt.Y(f"{column}:N", title=None, sort=sort or "-x"),
            tooltip=[
                alt.Tooltip(f"{column}:N"),
                alt.Tooltip("churn_rate:Q", title="Churn rate (%)", format=".1f"),
                alt.Tooltip("customers:Q", title="Customers"),
            ],
        )
        .properties(title=f"Churn rate by {column.replace('_', ' ')}", height=200)
    )


def show_image(file_name, caption=None):
    path = PLOTS_PATH / file_name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Plot not found: `plots/{file_name}`.")


def options(column):
    return sorted(df[column].dropna().unique().tolist())


# ----------------------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
contract = st.sidebar.multiselect("Contract", options("contract"), default=options("contract"))
internet = st.sidebar.multiselect("Internet service", options("internet_service"), default=options("internet_service"))
payment = st.sidebar.multiselect("Payment method", options("payment_method"), default=options("payment_method"))
senior = st.sidebar.radio("Senior citizen", ["All", "Yes", "No"], horizontal=True)
tenure_range = st.sidebar.slider("Tenure (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))

mask = (
    df["contract"].isin(contract)
    & df["internet_service"].isin(internet)
    & df["payment_method"].isin(payment)
    & df["tenure"].between(*tenure_range)
)
if senior != "All":
    mask &= df["senior_citizen"] == (1 if senior == "Yes" else 0)
df_filtered = df[mask]
st.sidebar.caption(f"{len(df_filtered):,} of {len(df):,} customers selected")

# ----------------------------------------------------------------------------- header
st.title("Telco Customer Churn Dashboard")
st.markdown("Understand customer behavior, churn drivers and predict churn risk using machine learning.")

tab_overview, tab_model, tab_predict, tab_data = st.tabs(["Overview", "Model performance", "Predict churn", "Data"])

# ----------------------------------------------------------------------------- overview (live)
with tab_overview:
    if df_filtered.empty:
        st.warning("No customers match the selected filters.")
    else:
        kpis, kpis_all = compute_kpis(df_filtered), compute_kpis(df)

        def delta(key, fmt="{:+.2f}"):
            return fmt.format(kpis[key] - kpis_all[key]) if len(df_filtered) < len(df) else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{kpis['Total Customers']:,}")
        c2.metric(
            "Churn rate",
            f"{kpis['Churn Rate (%)']:.1f}%",
            delta(key="Churn Rate (%)", fmt="{:+.1f} pp"),
            delta_color="inverse",
        )
        c3.metric(
            "Avg monthly charges", f"${kpis['Average Monthly Charges ($)']:.2f}", delta("Average Monthly Charges ($)")
        )
        c4.metric(
            "Avg tenure", f"{kpis['Average Tenure (months)']:.1f} months", delta("Average Tenure (months)", "{:+.1f}")
        )
        if len(df_filtered) < len(df):
            st.caption("Deltas compare the filtered selection with all customers.")

        left, right = st.columns(2)
        left.altair_chart(churn_rate_by(df_filtered, "contract"), width="stretch")
        right.altair_chart(churn_rate_by(df_filtered, "internet_service"), width="stretch")

        tenure_bins = pd.cut(
            df_filtered["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        )
        left, right = st.columns(2)
        left.altair_chart(
            churn_rate_by(df_filtered.assign(tenure_group=tenure_bins), "tenure_group", sort="ascending"),
            width="stretch",
        )
        right.altair_chart(churn_rate_by(df_filtered, "payment_method"), width="stretch")

        charges = (
            alt.Chart(df_filtered)
            .transform_density("monthly_charges", groupby=["churn"], as_=["monthly_charges", "density"])
            .mark_area(opacity=0.5)
            .encode(
                x=alt.X("monthly_charges:Q", title="Monthly charges ($)"),
                y=alt.Y("density:Q", title="Density", stack=None),
                color=alt.Color("churn:N", scale=CHURN_COLORS, title="Churn"),
            )
            .properties(title="Monthly charges: churned vs retained customers", height=250)
        )
        st.altair_chart(charges, width="stretch")

        with st.expander("Exploratory analysis plots (full dataset, from notebooks/eda.ipynb)"):
            for plot_file in EDA_PLOTS:
                show_image(plot_file)

# ----------------------------------------------------------------------------- model performance
with tab_model:
    metrics = get_metrics()
    if meta is None or metrics is None:
        st.warning(f"No trained model found. {MISSING_MODEL_HINT}")
    else:
        selection_metric = meta["selection_metric"].upper().replace("_", "-")
        st.markdown(
            f"Selected model: **{meta['model']}** (best cross-validated {selection_metric}), "
            f"decision threshold **{meta['threshold']:.2f}**."
        )
        table = pd.DataFrame(metrics["test"]).T.drop(columns=["confusion_matrix"])
        table = table[["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]]
        table.columns = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]
        st.markdown("**Held-out test set** (never used for tuning, threshold or model selection)")
        st.dataframe(table.style.format("{:.3f}"), width="stretch")
        show_image("model_evaluation.png", "ROC and precision-recall curves, confusion matrices (test set)")
        show_image("shap_importance.png", "Global feature importance of the selected model")

# ----------------------------------------------------------------------------- prediction
with tab_predict:
    if model is None:
        st.warning(f"No trained model found. {MISSING_MODEL_HINT}")
    else:
        with st.form("prediction_form"):
            st.markdown("**Customer details**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                gender = st.selectbox("Gender", options("gender"))
                senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", options("partner"))
                dependents = st.selectbox("Dependents", options("dependents"))
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract_in = st.selectbox("Contract", options("contract"))
            with col_b:
                phone_service = st.selectbox("Phone service", options("phone_service"), index=1)
                multiple_lines = st.selectbox("Multiple lines", options("multiple_lines"))
                internet_service = st.selectbox("Internet service", options("internet_service"))
                online_security = st.selectbox("Online security", options("online_security"))
                online_backup = st.selectbox("Online backup", options("online_backup"))
                device_protection = st.selectbox("Device protection", options("device_protection"))
            with col_c:
                tech_support = st.selectbox("Tech support", options("tech_support"))
                streaming_tv = st.selectbox("Streaming TV", options("streaming_tv"))
                streaming_movies = st.selectbox("Streaming movies", options("streaming_movies"))
                paperless_billing = st.selectbox("Paperless billing", options("paperless_billing"))
                payment_method = st.selectbox("Payment method", options("payment_method"))
                monthly_charges = st.slider("Monthly charges ($)", 0.0, 150.0, 70.0)
            auto_total = st.checkbox("Estimate total charges as tenure × monthly charges", value=True)
            total_charges_in = st.number_input(
                "Total charges ($), used when the box above is unchecked", 0.0, 10000.0, 0.0
            )
            submit = st.form_submit_button("Predict churn")

        if submit:
            total_charges = tenure * monthly_charges if auto_total else total_charges_in
            customer = pd.DataFrame(
                [
                    {
                        "customer_id": "new-customer",
                        "gender": gender,
                        "senior_citizen": 1 if senior_citizen == "Yes" else 0,
                        "partner": partner,
                        "dependents": dependents,
                        "tenure": tenure,
                        "phone_service": phone_service,
                        "multiple_lines": multiple_lines,
                        "internet_service": internet_service,
                        "online_security": online_security,
                        "online_backup": online_backup,
                        "device_protection": device_protection,
                        "tech_support": tech_support,
                        "streaming_tv": streaming_tv,
                        "streaming_movies": streaming_movies,
                        "contract": contract_in,
                        "paperless_billing": paperless_billing,
                        "payment_method": payment_method,
                        "monthly_charges": monthly_charges,
                        "total_charges": total_charges,
                    }
                ]
            )[meta["input_columns"]]

            # the saved pipeline does all the preprocessing; unused columns (customer_id, gender...) are ignored
            proba = model.predict_proba(customer)[0][1]
            threshold = meta["threshold"]
            message = f"Churn probability **{proba:.1%}** (decision threshold {threshold:.0%}, model: {meta['model']})."
            if proba >= threshold:
                st.error(f"High churn risk. {message}")
            else:
                st.success(f"Low churn risk. {message}")
            if auto_total:
                st.caption(f"Total charges estimated at ${total_charges:,.2f}.")

            contributions = explain_prediction(
                model,
                meta["model_type"],
                customer,
                meta["categorical_features"],
                meta["numerical_features"],
                background=df[meta["input_columns"]],
            ).head(8)
            explanation = pd.DataFrame(
                {
                    "feature": [f"{f} = {customer.iloc[0][f]}" for f in contributions.index],
                    "contribution": contributions.values,
                    "effect": ["Increases risk" if v > 0 else "Decreases risk" for v in contributions.values],
                }
            )
            chart = (
                alt.Chart(explanation)
                .mark_bar()
                .encode(
                    x=alt.X("contribution:Q", title="Contribution to churn risk (SHAP, log-odds)"),
                    y=alt.Y("feature:N", title=None, sort=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                    color=alt.Color(
                        "effect:N",
                        scale=alt.Scale(domain=["Increases risk", "Decreases risk"], range=["#E45756", "#4C78A8"]),
                        title=None,
                    ),
                    tooltip=["feature", alt.Tooltip("contribution:Q", format=".3f")],
                )
                .properties(title="What drives this prediction (top 8 features)", height=32 * len(explanation))
            )
            st.altair_chart(chart, width="stretch")

# ----------------------------------------------------------------------------- data
with tab_data:
    st.dataframe(df_filtered, width="stretch")
    st.download_button(
        "Download filtered data (CSV)",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="telco_customers_filtered.csv",
        mime="text/csv",
    )
