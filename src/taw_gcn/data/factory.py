from __future__ import annotations

from typing import Any

from taw_gcn.data.base_adapter import BaseTrafficAdapter
from taw_gcn.data.demo_adapter import DemoAdapter
from taw_gcn.data.iot23_adapter import IoT23Adapter
from taw_gcn.data.ton_iot_adapter import TONIoTAdapter


def create_adapter(config: dict[str, Any]) -> BaseTrafficAdapter:
    dataset_config = config["dataset"]
    name = str(dataset_config.get("adapter", dataset_config.get("name", ""))).lower()
    mapping: dict[str, type[BaseTrafficAdapter]] = {
        "iot23": IoT23Adapter,
        "ton_iot": TONIoTAdapter,
        "demo": DemoAdapter,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dataset adapter: {name}")
    return mapping[name](dataset_config)
