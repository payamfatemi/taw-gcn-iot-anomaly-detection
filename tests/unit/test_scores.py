import numpy as np
import pytest

from taw_gcn.graph.scores import (
    behavioral_score,
    combined_edge_weight,
    communication_score,
    temporal_score,
)


def test_temporal_score_monotonic() -> None:
    assert temporal_score(10, 10, 5) == pytest.approx(1.0)
    assert temporal_score(10, 11, 5) > temporal_score(10, 20, 5)


def test_communication_score() -> None:
    left = {"src_ip": "a", "protocol": "tcp"}
    right = {"src_ip": "b", "protocol": "tcp"}
    assert communication_score(left, right, ["src_ip", "protocol"]) == 1.0
    assert communication_score(left, {"src_ip": "c", "protocol": "udp"}, ["src_ip", "protocol"]) == 0.0


def test_behavioral_score_identity() -> None:
    x = np.asarray([1.0, 2.0, 3.0])
    assert behavioral_score(x, x) == pytest.approx(1.0)


def test_combined_weight_and_ablation_renormalization() -> None:
    full = combined_edge_weight(1.0, 0.0, 0.0, 0.5, 0.25, 0.25)
    no_temp = combined_edge_weight(1.0, 1.0, 0.0, 0.5, 0.25, 0.25, (False, True, True))
    assert full == pytest.approx(0.5)
    assert no_temp == pytest.approx(0.5)
