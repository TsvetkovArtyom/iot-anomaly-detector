from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FeatureConfig:
    numeric_cols: List[str] = field(default_factory=lambda: [
        "bytes_log", "packets_log", "pps_log", "bps_log", "flow_duration_ms"
    ])
    categorical_cols: List[str] = field(default_factory=lambda: ["protocol","device_id"])
    time_col: str = "timestamp"
    protocol_map: Dict[str, int] = field(default_factory=lambda: {
        "TCP":6,"UDP":17,"ICMP":1,"HTTP":80,"HTTPS":443,"MQTT":1883,"CoAP":5683,"DNS":53
    })

@dataclass
class TrainConfig:
    algo: str = "isolation_forest"
    n_estimators: int = 300
    contamination: float = 0.08
    max_samples: str | int = "auto"
    random_state: int = 42

@dataclass
class ThresholdConfig:
    threshold: float = 0.7
