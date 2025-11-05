import json, pandas as pd
import paho.mqtt.client as mqtt
from .config import TrainConfig, ThresholdConfig
from .features import FeatureExtractor
from .model import ModelManager
from .alerts import AlertSink
from .actions import SecurityActionManager

def run(broker: str, port: int, topic: str, model_path: str, threshold: float):
    fx = FeatureExtractor()
    mgr = ModelManager(fx, TrainConfig()); mgr.load(model_path)
    thr = ThresholdConfig(threshold)
    alert = AlertSink()
    actions = SecurityActionManager(enable_actions=False)

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] connected rc={rc}"); client.subscribe(topic)

    def on_message(client, userdata, msg):
        try:
        # Убедись, что внутри try есть правильные отступы
            payload = json.loads(msg.payload.decode("utf-8"))
            df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
            scores = mgr.score(df)

            for rec, s in zip(df.to_dict(orient="records"), scores):
                is_anom = float(s) >= thr.threshold
            # ВСЕГДА логируем:
                print(f"[MQTT] topic={msg.topic} device={rec.get('device_id')} score={float(s):.3f} anomaly={is_anom}")
                if is_anom:
                    alert.send(rec, float(s), thr.threshold)
                    actions.quarantine(rec.get("src_ip","unknown"))
        except Exception as e:
            print("[MQTT] error:", e, "raw:", msg.payload[:200])

    client = mqtt.Client()
    client.on_connect = on_connect; client.on_message = on_message
    client.connect(broker, port); client.loop_forever()
