from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Optional imports
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

try:
    from flask import Flask, request, jsonify  # type: ignore
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt  # type: ignore
    MQTT_AVAILABLE = True
except Exception:
    MQTT_AVAILABLE = False


# -----------------------------
# Configuration dataclasses
# -----------------------------
@dataclass
class FeatureConfig:
    # Updated feature set: log features and port bins (avoid raw large port magnitudes)
    numeric_cols: List[str] = field(default_factory=lambda: [
    "bytes_log", "packets_log", "pps_log", "bps_log", "flow_duration_ms"
])
    categorical_cols: List[str] = field(default_factory=lambda: [
        "protocol", "device_id",
    ])
    time_col: str = "timestamp"

    # Mapping for protocols to integers (extend as needed)
    protocol_map: Dict[str, int] = field(default_factory=lambda: {
        "TCP": 6, "UDP": 17, "ICMP": 1, "HTTP": 80, "HTTPS": 443,
        "MQTT": 1883, "CoAP": 5683, "DNS": 53,
    })


@dataclass
class TrainConfig:
    algo: str = "isolation_forest"  # or "autoencoder" (if TF)
    n_estimators: int = 300
    contamination: float = 0.02
    max_samples: str | int = "auto"
    random_state: int = 42


@dataclass
class ThresholdConfig:
    threshold: float = 0.6  # Higher => less sensitive


# -----------------------------
# Utilities
# -----------------------------

def _ensure_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _port_bin(x: Any) -> int:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v <= 1023:
        return 0  # well-known
    if v <= 49151:
        return 1  # registered
    return 2      # dynamic/private


# -----------------------------
# Feature Extraction
# -----------------------------
class FeatureExtractor:
    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Timestamp
        if self.cfg.time_col in df.columns:
            df[self.cfg.time_col] = _ensure_datetime(df[self.cfg.time_col])

        # Protocol → proto_id
        df["protocol"] = df.get("protocol", "TCP").fillna("TCP").astype(str)
        df["proto_id"] = df["protocol"].map(self.cfg.protocol_map).fillna(0).astype(int)

        # Base numeric
        df["bytes"] = df.get("bytes", 0).apply(_safe_float)
        df["packets"] = df.get("packets", 0).apply(_safe_float).clip(lower=1)

        # Flow duration: softened heuristic if no explicit start/end
        if "flow_start" in df.columns and "flow_end" in df.columns:
            df["flow_start"] = _ensure_datetime(df["flow_start"]) 
            df["flow_end"] = _ensure_datetime(df["flow_end"]) 
            df["flow_duration_ms"] = (df["flow_end"] - df["flow_start"]).dt.total_seconds() * 1000
        else:
            # 10 ms per packet, but not lower than 50 ms per flow
            df["flow_duration_ms"] = (df["packets"] * 10.0).clip(lower=50.0)

        duration_s = (df["flow_duration_ms"].clip(lower=1.0) / 1000.0)
        df["pps"] = df["packets"] / duration_s
        df["bps"] = (df["bytes"] * 8.0) / duration_s

        # Log-stabilized features
        df["bytes_log"] = np.log1p(df["bytes"])
        df["packets_log"] = np.log1p(df["packets"])
        df["pps_log"] = np.log1p(df["pps"])
        df["bps_log"] = np.log1p(df["bps"])

        # Port bins
        for col in ["src_port", "dst_port"]:
            if col not in df:
                df[col] = 0
        df["src_port_bin"] = df["src_port"].apply(_port_bin)
        df["dst_port_bin"] = df["dst_port"].apply(_port_bin)

        # Device ID fallback
        if "device_id" not in df:
            df["device_id"] = "unknown"

        cols = list(dict.fromkeys(self.cfg.numeric_cols))
        X = df[cols].fillna(0.0)
        return X


# -----------------------------
# Model Manager (with calibration)
# -----------------------------
class ModelManager:
    def __init__(self, feature_extractor: FeatureExtractor, train_cfg: TrainConfig):
        self.fx = feature_extractor
        self.train_cfg = train_cfg
        self.scaler = StandardScaler()
        self.model = None  # type: ignore
        # Calibration for IsolationForest → stable single-record scores
        self.if_shift: Optional[float] = None
        self.if_scale: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> None:
        X = self.fx.transform_frame(df)
        X_scaled = self.scaler.fit_transform(X)

        if self.train_cfg.algo == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=self.train_cfg.n_estimators,
                contamination=self.train_cfg.contamination,
                max_samples=self.train_cfg.max_samples,
                random_state=self.train_cfg.random_state,
                n_jobs=-1,
            )
            self.model.fit(X_scaled)

            # Calibration using train-set percentiles (5th/95th)
            train_scores = -self.model.score_samples(X_scaled)
            self.if_shift = float(np.percentile(train_scores, 1))
            self.if_scale = float(np.percentile(train_scores, 99) - self.if_shift)
            if self.if_scale <= 1e-8:
                self.if_scale = 1.0

        elif self.train_cfg.algo == "autoencoder":
            if not TENSORFLOW_AVAILABLE:
                raise RuntimeError("TensorFlow is not available for autoencoder option.")
            self.model = self._fit_autoencoder(X_scaled)
        else:
            raise ValueError(f"Unknown algo: {self.train_cfg.algo}")

    def score_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        X = self.fx.transform_frame(df)
        X_scaled = self.scaler.transform(X)

        if isinstance(self.model, IsolationForest):
            raw = -self.model.score_samples(X_scaled)
            if self.if_shift is not None and self.if_scale is not None:
                scores = (raw - self.if_shift) / self.if_scale
                return np.clip(scores, 0, 1)
            # Fallback for legacy bundles
            p95 = np.percentile(raw, 95)
            return np.clip(raw / (p95 + 1e-8), 0, 1)
        else:
            # Autoencoder: reconstruction error
            preds = self.model.predict(X_scaled, verbose=0)
            mse = np.mean(np.square(X_scaled - preds), axis=1)
            p95 = np.percentile(mse, 95)
            return np.clip(mse / (p95 + 1e-8), 0, 1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dump({
            "scaler": self.scaler,
            "model": self.model,
            "train_cfg": self.train_cfg,
            "if_shift": self.if_shift,
            "if_scale": self.if_scale,
        }, path)

    def load(self, path: str) -> None:
        bundle = load(path)
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.train_cfg = bundle.get("train_cfg", self.train_cfg)
        self.if_shift = bundle.get("if_shift", None)
        self.if_scale = bundle.get("if_scale", None)

    # --- Autoencoder training ---
    def _fit_autoencoder(self, X: np.ndarray):
        input_dim = X.shape[1]
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(max(8, input_dim//2), activation='relu'),
            keras.layers.Dense(max(4, input_dim//4), activation='relu'),
            keras.layers.Dense(max(8, input_dim//2), activation='relu'),
            keras.layers.Dense(input_dim, activation='linear'),
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=20, batch_size=256, verbose=0, validation_split=0.1)
        return model


# -----------------------------
# Alerting & realtime pipeline
# -----------------------------
class AlertSink:
    def __init__(self):
        pass
    def send(self, record: Dict[str, Any], score: float, threshold: float) -> None:
        print(json.dumps({"event": "anomaly", "score": score, "threshold": threshold, "record": record}, ensure_ascii=False))


class RealtimeDetector:
    def __init__(self, manager: ModelManager, thr_cfg: ThresholdConfig, alert_sink: Optional[AlertSink] = None):
        self.mgr = manager
        self.thr = thr_cfg
        self.alert = alert_sink or AlertSink()

    def score_record(self, record: Dict[str, Any]) -> float:
        df = pd.DataFrame([record])
        score = float(self.mgr.score_anomaly(df)[0])
        if score >= self.thr.threshold:
            self.alert.send(record, score, self.thr.threshold)
        return score


# -----------------------------
# Data Loading helpers
# -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# -----------------------------
# CLI Commands
# -----------------------------

def cmd_train(args: argparse.Namespace) -> None:
    df = load_csv(args.input)
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig(
        algo=args.algo,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        random_state=args.random_state,
    ))
    if args.label_col and args.normal_label is not None and args.label_col in df.columns:
        normal_df = df[df[args.label_col] == args.normal_label]
        if len(normal_df) < 100:
            print("[WARN] Very few normal samples for training; using all data.")
            normal_df = df
        mgr.fit(normal_df)
    else:
        mgr.fit(df)
    mgr.save(args.model)
    print(f"[OK] Model saved to {args.model}")


def cmd_eval(args: argparse.Namespace) -> None:
    df = load_csv(args.input)
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig())
    mgr.load(args.model)
    scores = mgr.score_anomaly(df)
    if args.label_col and args.label_col in df.columns and args.positive_label is not None:
        y_true = (df[args.label_col] == args.positive_label).astype(int).values
        y_pred = (scores >= args.threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        print(json.dumps({"precision": float(prec), "recall": float(rec), "f1": float(f1)}, indent=2))
        try:
            auc = roc_auc_score(y_true, scores)
            print(json.dumps({"roc_auc": float(auc)}, indent=2))
        except Exception:
            pass
    else:
        print("[INFO] No labels provided; outputting scores only.")
    out = df.copy()
    out["anomaly_score"] = scores
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[OK] Scored data -> {args.output}")


def cmd_serve(args: argparse.Namespace) -> None:
    if not FLASK_AVAILABLE:
        print("[ERR] flask is not installed. pip install flask")
        sys.exit(2)
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig())
    mgr.load(args.model)
    rtd = RealtimeDetector(mgr, ThresholdConfig(args.threshold))

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}

    @app.route("/score", methods=["POST"])
    def score():
        payload = request.get_json(force=True)
        if isinstance(payload, dict):
            score = rtd.score_record(payload)
            return jsonify({"score": score, "threshold": rtd.thr.threshold, "anomaly": score >= rtd.thr.threshold})
        elif isinstance(payload, list):
            scores = [float(rtd.score_record(p)) for p in payload]
            return jsonify({"scores": scores, "threshold": rtd.thr.threshold})
        else:
            return jsonify({"error": "Invalid payload"}), 400

    app.run(host=args.host, port=args.port)


def cmd_mqtt(args: argparse.Namespace) -> None:
    if not MQTT_AVAILABLE:
        print("[ERR] paho-mqtt is not installed. pip install paho-mqtt")
        sys.exit(2)
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig())
    mgr.load(args.model)
    rtd = RealtimeDetector(mgr, ThresholdConfig(args.threshold))

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        client.subscribe(args.topic)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            if isinstance(payload, dict):
                score = rtd.score_record(payload)
                print(f"[MQTT] score={score:.3f} anomaly={score >= rtd.thr.threshold}")
            elif isinstance(payload, list):
                for rec in payload:
                    score = rtd.score_record(rec)
            else:
                print("[MQTT] Unsupported payload type")
        except Exception as e:
            print(f"[MQTT] Error: {e}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port)
    client.loop_forever()


# -----------------------------
# Main / argparse
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IoT Anomaly Detection Module")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    sp = sub.add_parser("train", help="Train model on CSV")
    sp.add_argument("--input", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--algo", default="isolation_forest", choices=["isolation_forest", "autoencoder"])
    sp.add_argument("--n-estimators", type=int, default=300)
    sp.add_argument("--contamination", type=float, default=0.02)
    sp.add_argument("--max-samples", default="auto")
    sp.add_argument("--random-state", type=int, default=42)
    sp.add_argument("--label-col", default=None)
    sp.add_argument("--normal-label", default=None)
    sp.set_defaults(func=cmd_train)

    # eval
    sp = sub.add_parser("eval", help="Evaluate model on CSV")
    sp.add_argument("--input", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--label-col", default=None)
    sp.add_argument("--positive-label", default=None)
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.set_defaults(func=cmd_eval)

    # serve
    sp = sub.add_parser("serve", help="Serve HTTP scoring API (Flask)")
    sp.add_argument("--model", required=True)
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.set_defaults(func=cmd_serve)

    # mqtt
    sp = sub.add_parser("mqtt", help="MQTT ingest and scoring")
    sp.add_argument("--model", required=True)
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.add_argument("--broker", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=1883)
    sp.add_argument("--topic", default="iot/flows")
    sp.set_defaults(func=cmd_mqtt)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"[ERR] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

