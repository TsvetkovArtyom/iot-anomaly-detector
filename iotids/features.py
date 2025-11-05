import numpy as np, pandas as pd
from .config import FeatureConfig
from .dataio import ensure_datetime, safe_float, port_bin

class FeatureExtractor:
    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.cfg.time_col in df.columns:
            df[self.cfg.time_col] = ensure_datetime(df[self.cfg.time_col])

        df["protocol"] = df.get("protocol","TCP").fillna("TCP").astype(str)
        df["proto_id"] = df["protocol"].map(self.cfg.protocol_map).fillna(0).astype(int)

        df["bytes"] = df.get("bytes",0).apply(safe_float)
        df["packets"] = df.get("packets",0).apply(safe_float).clip(lower=1)

        if {"flow_start","flow_end"}.issubset(df.columns):
            df["flow_duration_ms"] = (ensure_datetime(df["flow_end"]) - ensure_datetime(df["flow_start"])).dt.total_seconds()*1000
        else:
            df["flow_duration_ms"] = (df["packets"]*10.0).clip(lower=100.0)  # мягкая эвристика

        dur_s = (df["flow_duration_ms"].clip(lower=1.0)/1000.0)
        df["pps"] = df["packets"]/dur_s
        df["bps"] = (df["bytes"]*8.0)/dur_s

        df["bytes_log"]   = np.log1p(df["bytes"])
        df["packets_log"] = np.log1p(df["packets"])
        df["pps_log"]     = np.log1p(df["pps"])
        df["bps_log"]     = np.log1p(df["bps"])

        for col in ("src_port","dst_port"):
            if col not in df: df[col] = 0
        df["src_port_bin"] = df["src_port"].apply(port_bin)
        df["dst_port_bin"] = df["dst_port"].apply(port_bin)

        if "device_id" not in df:
            df["device_id"] = "unknown"

        X = df[self.cfg.numeric_cols].fillna(0.0)
        return X
