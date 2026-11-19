import pandas as pd

from clean_data import fix_totalcharges, rename_columns


def test_rename_columns_to_snake_case(raw_customers):
    df = rename_columns(raw_customers)
    assert {"customer_id", "senior_citizen", "monthly_charges", "total_charges", "churn"} <= set(df.columns)
    assert "customerid" not in df.columns


def test_blank_total_charges_of_new_customers_become_zero(raw_customers):
    df = fix_totalcharges(rename_columns(raw_customers))
    assert len(df) == 4
    assert df["total_charges"].dtype.kind == "f"
    assert df.loc[df["tenure"] == 0, "total_charges"].tolist() == [0.0]


def test_missing_total_charges_with_tenure_are_dropped(raw_customers):
    raw = raw_customers.copy()
    raw.loc[0, "totalcharges"] = " "  # tenure 1, so the value is really missing
    df = fix_totalcharges(rename_columns(raw))
    assert df["total_charges"].notna().all()
    assert df["customer_id"].tolist() == ["0002-B", "0003-C", "0004-D"]
    # the brand-new customer (tenure 0) is kept with 0, even though another row was dropped
    assert df.loc[df["customer_id"] == "0003-C", "total_charges"].item() == 0.0
    assert pd.api.types.is_float_dtype(df["total_charges"])


def test_fix_totalcharges_does_not_modify_input(raw_customers):
    df = rename_columns(raw_customers)
    before = df.copy()
    fix_totalcharges(df)
    assert df.equals(before)
