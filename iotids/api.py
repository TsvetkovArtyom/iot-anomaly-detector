from flask import Flask, request, jsonify
import pandas as pd
from .config import TrainConfig, ThresholdConfig
from .features import FeatureExtractor
from .model import ModelManager
from .alerts import AlertSink
from .actions import SecurityActionManager

def create_app(model_path: str, threshold: float):
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig())
    mgr.load(model_path)
    thr = ThresholdConfig(threshold)
    alert = AlertSink()
    actions = SecurityActionManager(enable_actions=False)

    app = Flask(__name__)

    @app.get("/health")
    def health(): return {"status":"ok"}

    @app.post("/score")
    def score():
        payload = request.get_json(force=True)
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
            s = float(mgr.score(df)[0])
            if s >= thr.threshold:
                alert.send(payload, s, thr.threshold)
                actions.quarantine(payload.get("src_ip","unknown"))
            return jsonify({"score":s, "threshold":thr.threshold, "anomaly": s>=thr.threshold})
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
            scores = [float(x) for x in mgr.score(df)]
            return jsonify({"scores":scores, "threshold":thr.threshold})
        return jsonify({"error":"invalid payload"}), 400

    return app

def serve(model: str, threshold: float, host: str, port: int):
    app = create_app(model, threshold)
    app.run(host=host, port=port)
