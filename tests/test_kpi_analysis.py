import pytest

from clean_data import fix_totalcharges, rename_columns
from kpi_analysis import compute_kpis


def test_kpis_values(raw_customers):
    df = fix_totalcharges(rename_columns(raw_customers))
    kpis = compute_kpis(df)
    assert kpis["Total Customers"] == 4
    assert kpis["Churned Customers"] == 1
    assert kpis["Churn Rate (%)"] == 25.0
    assert kpis["Average Tenure (months)"] == 20.0
    assert kpis["Average Monthly Charges ($)"] == pytest.approx(37.28, abs=0.01)


def test_kpis_do_not_modify_input(raw_customers):
    df = fix_totalcharges(rename_columns(raw_customers))
    before = df.copy()
    compute_kpis(df)
    assert df.equals(before)


def test_kpis_of_empty_selection(raw_customers):
    df = fix_totalcharges(rename_columns(raw_customers))
    kpis = compute_kpis(df.iloc[0:0])
    assert kpis["Total Customers"] == 0
    assert kpis["Churn Rate (%)"] == 0.0
