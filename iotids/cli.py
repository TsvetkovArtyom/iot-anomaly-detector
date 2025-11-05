import argparse, json, os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from .dataio import load_csv
from .features import FeatureExtractor
from .model import ModelManager
from .config import TrainConfig
from .api import serve as serve_api
from .mqtt_worker import run as run_mqtt

def cmd_train(a):
    df = load_csv(a.input)
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig(
        algo=a.algo, n_estimators=a.n_estimators, contamination=a.contamination,
        max_samples=a.max_samples, random_state=a.random_state
    ))
    if a.label_col and a.normal_label is not None and a.label_col in df.columns:
        df = df[df[a.label_col].astype(str).str.lower()==str(a.normal_label).lower()]
    mgr.fit(df); mgr.save(a.model); print(f"[OK] saved {a.model}")

def cmd_eval(a):
    df = load_csv(a.input); fx = FeatureExtractor(); mgr = ModelManager(fx, TrainConfig()); mgr.load(a.model)
    scores = mgr.score(df)
    if a.label_col and a.label_col in df.columns and a.positive_label is not None:
        y_true = (df[a.label_col].astype(str).str.lower()==str(a.positive_label).lower()).astype(int)
        y_pred = (scores >= a.threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        print(json.dumps({"precision":float(prec),"recall":float(rec),"f1":float(f1)}, indent=2))
        try: print(json.dumps({"roc_auc": float(roc_auc_score(y_true, scores))}, indent=2))
        except: pass
    out = df.copy(); out["anomaly_score"] = scores
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    out.to_csv(a.output, index=False); print(f"[OK] scored -> {a.output}")

def build_parser():
    p = argparse.ArgumentParser("iotids")
    s = p.add_subparsers(dest="cmd", required=True)

    sp = s.add_parser("train"); sp.add_argument("--input", required=True); sp.add_argument("--model", required=True)
    sp.add_argument("--algo", default="isolation_forest", choices=["isolation_forest"])
    sp.add_argument("--n-estimators", type=int, default=300)
    sp.add_argument("--contamination", type=float, default=0.08)
    sp.add_argument("--max-samples", default="auto"); sp.add_argument("--random-state", type=int, default=42)
    sp.add_argument("--label-col"); sp.add_argument("--normal-label"); sp.set_defaults(func=cmd_train)

    sp = s.add_parser("eval"); sp.add_argument("--input", required=True); sp.add_argument("--model", required=True)
    sp.add_argument("--output", required=True); sp.add_argument("--label-col"); sp.add_argument("--positive-label")
    sp.add_argument("--threshold", type=float, default=0.7); sp.set_defaults(func=cmd_eval)

    sp = s.add_parser("serve"); sp.add_argument("--model", required=True); sp.add_argument("--threshold", type=float, default=0.7)
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8080)
    sp.set_defaults(func=lambda a: serve_api(a.model, a.threshold, a.host, a.port))

    sp = s.add_parser("mqtt"); sp.add_argument("--model", required=True); sp.add_argument("--threshold", type=float, default=0.7)
    sp.add_argument("--broker", default="127.0.0.1"); sp.add_argument("--port", type=int, default=1883); sp.add_argument("--topic", default="iot/flows")
    sp.set_defaults(func=lambda a: run_mqtt(a.broker, a.port, a.topic, a.model, a.threshold))
    return p

def main(argv=None):
    a = build_parser().parse_args(argv); a.func(a)

if __name__ == "__main__":
    main()
