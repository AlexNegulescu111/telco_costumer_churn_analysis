import json
import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_customers.csv"
KPI_PATH = ROOT / "data" / "kpi_summary.json"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Load the raw Telco CSV with lower-cased column names."""
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]
    return df


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path):
    return joblib.load(path)


def save_kpi(kpis, path=KPI_PATH):
    save_json(kpis, path)
    logger.info("KPIs saved to %s", path)


def save_best_params(best_params, name, models_dir=MODELS_DIR):
    path = Path(models_dir) / f"{name}.json"
    save_json(best_params, path)
    logger.info("Best params saved to %s", path)


def load_best_params(name, models_dir=MODELS_DIR):
    path = Path(models_dir) / f"{name}.json"
    logger.info("Best params loaded from %s", path)
    return load_json(path)
