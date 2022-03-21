import torch

from . import FEN, FENDynamics, FreeFormTerm, ODESolver, TransportTerm


def load_checkpoint(ckpt: dict):
    config = ckpt["config"]
    terms = []
    if "free_form" in config["dynamics"]["terms"]:
        ff_cfg = config["dynamics"]["terms"]["free_form"].copy()
        assert ff_cfg["non_linearity"] == "tanh"
        ff_cfg["non_linearity"] = torch.nn.Tanh
        free_form_term = FreeFormTerm(
            FreeFormTerm.build_coefficient_mlp(**ff_cfg),
            stationary=ff_cfg["stationary"],
            autonomous=ff_cfg["autonomous"],
            zero_init=True,
        )
        terms.append(free_form_term)
    if "transport" in config["dynamics"]["terms"]:
        tp_cfg = config["dynamics"]["terms"]["transport"].copy()
        assert tp_cfg["non_linearity"] == "tanh"
        tp_cfg["non_linearity"] = torch.nn.Tanh
        transport_term = TransportTerm(
            TransportTerm.build_flow_field_mlp(**tp_cfg),
            stationary=tp_cfg["stationary"],
            autonomous=tp_cfg["autonomous"],
            zero_init=True,
        )
        terms.append(transport_term)
    dynamics = FENDynamics(terms)

    ode_solver = ODESolver(**config["ode_solver"])
    model = FEN(dynamics, ode_solver)
    model.load_state_dict(ckpt["state_dict"])

    return model
