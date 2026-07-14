ABLATION_VARIANTS = {
    "full": {},
    "without_temporal": {"graph.temporal_enabled": False},
    "without_communication": {"graph.communication_enabled": False},
    "without_behavioral": {"graph.behavioral_enabled": False},
    "unweighted_graph": {"graph.weighted_edges": False},
    "without_graph_propagation": {
        "graph.temporal_enabled": False,
        "graph.communication_enabled": False,
        "graph.behavioral_enabled": False,
        "graph.weighted_edges": False,
        "graph.graph_propagation": False,
    },
}
