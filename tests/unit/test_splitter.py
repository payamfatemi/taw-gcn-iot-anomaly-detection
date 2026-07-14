from taw_gcn.preprocessing.leakage_checker import check_split_leakage
from taw_gcn.preprocessing.splitter import split_frame


def test_reproducible_group_split(canonical_frame, config) -> None:
    a = split_frame(canonical_frame, config["split"])
    b = split_frame(canonical_frame, config["split"])
    assert a[0]["record_id"].tolist() == b[0]["record_id"].tolist()
    report = check_split_leakage(a[0], a[1], a[2], a[3]["group_columns_used"])
    assert report["train_test_record_overlap"] == 0
