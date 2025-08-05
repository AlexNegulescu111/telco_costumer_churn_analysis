from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    df = pd.read_csv(ROOT / "data" / "telco_costumers.csv")
    df.columns = [col.lower() for col in df.columns]
    return df

def save_model(model, path):
    import joblib
    joblib.dump(model, path)
    print("Model saved successfully")

def load_model(path):
    import joblib
    return joblib.load(path)


def save_kpi(kpis):
    import json
    with open(ROOT / "data" / "kpi_summary.json", "w") as f:
        json.dump(kpis, f, indent=4)
        print("KPI's saved successfully")

def save_best_params(best_params, name):
    import json
    with open(ROOT / "models" / f"{name}.json", "w") as f:
        json.dump(best_params, f, indent=4)
        print("Best params saved successfully")

def load_best_params(name):
    import json
    with open(ROOT / "models" / f"{name}.json", "r") as f:
        print("Best params loaded successfully")
        return json.load(f)
