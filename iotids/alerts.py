import json, threading, requests

class AlertSink:
    def send(self, record: dict, score: float, threshold: float):  # console default
        print(json.dumps({"event":"anomaly","score":score,"threshold":threshold,"record":record}, ensure_ascii=False))

class AlertSinkWebhook(AlertSink):
    def __init__(self, webhook_url: str):
        self.url = webhook_url
    def send(self, record: dict, score: float, threshold: float):
        text = f"⚠️ Anomaly\nscore={score:.3f} thr={threshold} device={record.get('device_id')}"
        threading.Thread(target=requests.post, args=(self.url,), kwargs={"json":{"text":text}, "timeout":3}).start()
