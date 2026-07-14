from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_frame(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = np.cumsum(rng.exponential(scale=1.0, size=rows))
    devices = np.array([f"10.0.0.{i}" for i in range(1, 31)])
    servers = np.array([f"192.168.1.{i}" for i in range(1, 11)])
    src = rng.choice(devices, size=rows)
    dst = rng.choice(servers, size=rows)
    proto = rng.choice(["tcp", "udp", "icmp"], size=rows, p=[0.65, 0.30, 0.05])
    duration = rng.gamma(shape=2.0, scale=1.0, size=rows)
    bytes_out = rng.lognormal(mean=6.0, sigma=1.0, size=rows)
    packets = rng.poisson(lam=12.0, size=rows)
    attack_signal = (
        (bytes_out > np.quantile(bytes_out, 0.80)).astype(int)
        + (packets > 17).astype(int)
        + np.isin(proto, ["icmp"]).astype(int)
    )
    probability = 1 / (1 + np.exp(-(attack_signal - 1.8)))
    labels = np.where(rng.random(rows) < probability, "anomaly", "normal")
    labels[:2] = ["normal", "anomaly"]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "src_ip": src,
            "dst_ip": dst,
            "src_port": rng.integers(1024, 65535, size=rows),
            "dst_port": rng.choice([22, 53, 80, 443, 1883], size=rows),
            "protocol": proto,
            "service": rng.choice(["dns", "http", "mqtt", "ssh", "unknown"], size=rows),
            "conn_state": rng.choice(["SF", "S0", "REJ"], size=rows),
            "duration": duration,
            "bytes_out": bytes_out,
            "packets": packets,
            "capture_id": [f"capture_{i // 100}" for i in range(rows)],
            "scenario_id": [f"scenario_{i // 300}" for i in range(rows)],
            "label": labels,
        }
    )
