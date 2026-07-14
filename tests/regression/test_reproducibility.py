from taw_gcn.preprocessing.splitter import split_frame


def test_split_hash_stability(canonical_frame, config) -> None:
    first = split_frame(canonical_frame, config["split"])
    second = split_frame(canonical_frame, config["split"])
    for index in range(3):
        assert first[index]["record_id"].tolist() == second[index]["record_id"].tolist()
