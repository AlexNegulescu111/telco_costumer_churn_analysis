import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture
def raw_customers():
    """A few rows in the original CSV format (after lower-casing the column names)."""
    return pd.DataFrame(
        {
            "customerid": ["0001-A", "0002-B", "0003-C", "0004-D"],
            "gender": ["Female", "Male", "Male", "Female"],
            "seniorcitizen": [0, 1, 0, 0],
            "partner": ["Yes", "No", "No", "Yes"],
            "dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 34, 0, 45],
            "phoneservice": ["No", "Yes", "Yes", "No"],
            "multiplelines": ["No phone service", "No", "Yes", "No phone service"],
            "internetservice": ["DSL", "Fiber optic", "No", "DSL"],
            "onlinesecurity": ["No", "Yes", "No internet service", "Yes"],
            "onlinebackup": ["Yes", "No", "No internet service", "No"],
            "deviceprotection": ["No", "Yes", "No internet service", "Yes"],
            "techsupport": ["No", "No", "No internet service", "Yes"],
            "streamingtv": ["No", "Yes", "No internet service", "No"],
            "streamingmovies": ["No", "No", "No internet service", "No"],
            "contract": ["Month-to-month", "One year", "Two year", "One year"],
            "paperlessbilling": ["Yes", "No", "No", "No"],
            "paymentmethod": ["Electronic check", "Mailed check", "Mailed check", "Bank transfer (automatic)"],
            "monthlycharges": [29.85, 56.95, 20.0, 42.3],
            "totalcharges": ["29.85", "1889.5", " ", "1840.75"],
            "churn": ["Yes", "No", "No", "No"],
        }
    )


@pytest.fixture
def synthetic_churn():
    """Larger synthetic dataset with a known signal: month-to-month contracts and short tenure churn more;
    `noise_cat` and `noise_num` are unrelated to churn."""
    rng = np.random.default_rng(0)
    n = 600
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    tenure = rng.integers(0, 73, size=n)
    logit = -1.0 + 2.0 * (contract == "Month-to-month") - 0.04 * tenure
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    x = pd.DataFrame(
        {
            "customer_id": [f"id-{i}" for i in range(n)],
            "contract": contract,
            "noise_cat": rng.choice(["a", "b"], size=n),
            "tenure": tenure,
            "noise_num": rng.normal(size=n),
        }
    )
    y = pd.Series(churn.astype(int), name="churn")
    return x, y
