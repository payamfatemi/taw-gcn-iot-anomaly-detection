import torch

from taw_gcn.models.taw_gcn import TAWGCN


def test_model_output_shape() -> None:
    model = TAWGCN(4, [8, 4])
    x = torch.randn(5, 4)
    indices = torch.arange(5)
    edge_index = torch.stack([indices, indices])
    edge_weight = torch.ones(5)
    logits = model(x, edge_index, edge_weight)
    assert logits.shape == (5,)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
