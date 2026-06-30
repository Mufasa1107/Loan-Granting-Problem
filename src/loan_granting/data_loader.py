from pathlib import Path
import pandas as pd


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    data = pd.read_csv(raw_path)
    data["Dependents"] = data["Dependents"].astype(str)
    data["Credit_History"] = data["Credit_History"].astype(float)
    return data
