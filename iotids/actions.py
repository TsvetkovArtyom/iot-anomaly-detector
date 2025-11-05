import os, datetime

class SecurityActionManager:
    def __init__(self, enable_actions=False, log_path="artifacts/quarantine.log"):
        self.enable = enable_actions
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.log_path = log_path
    def quarantine(self, ip: str):
        ts = datetime.datetime.now().isoformat()
        msg = f"[{ts}] quarantine {ip}\n"
        with open(self.log_path, "a", encoding="utf-8") as f: f.write(msg)
        if self.enable:
            os.system(f'netsh advfirewall firewall add rule name="Block {ip}" dir=in interface=any action=block remoteip={ip}')
