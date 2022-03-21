from torch.nn import ReLU

from finite_element_networks import MLP


def test_no_hidden_layers():
    mlp = MLP(8, 5, 10, n_layers=0, non_linearity=ReLU)

    assert len(mlp) == 1
    assert mlp[0].weight.shape == (5, 8)
    assert mlp[0].bias.nelement() == 5

def test_two_hidden_layers():
    mlp = MLP(3, 6, 20, n_layers=2, non_linearity=ReLU)

    assert len(mlp) == 5
    assert mlp[0].weight.shape == (20, 3)
    assert mlp[0].bias.nelement() == 20
    assert isinstance(mlp[1], ReLU)
    assert mlp[2].weight.shape == (20, 20)
    assert mlp[2].bias.nelement() == 20
    assert isinstance(mlp[3], ReLU)
    assert mlp[4].weight.shape == (6, 20)
    assert mlp[4].bias.nelement() == 6
