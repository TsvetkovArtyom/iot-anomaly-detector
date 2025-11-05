import pandas as pd, numpy as np
from typing import Any

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def safe_float(x: Any) -> float:
    try: return float(x)
    except: return 0.0

def port_bin(x: Any) -> int:
    try: v = float(x)
    except: v = 0.0
    if v <= 1023: return 0
    if v <= 49151: return 1
    return 2
