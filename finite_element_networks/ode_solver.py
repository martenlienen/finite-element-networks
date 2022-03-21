from dataclasses import dataclass
from typing import Optional

import einops as eo
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from torchtyping import TensorType

from .data import TimeEncoder


class ODESolver(nn.Module):
    def __init__(
        self,
        method: str,
        atol: float,
        rtol: float,
        options: Optional[dict] = None,
        adjoint: bool = False,
        adjoint_options: Optional[dict] = None,
    ):
        super().__init__()

        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.options = options
        self.adjoint = adjoint
        self.adjoint_options = adjoint_options

        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, f: nn.Module, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        kwargs = dict(
            rtol=self.rtol, atol=self.atol, method=self.method, options=self.options
        )
        if self.adjoint:
            kwargs["adjoint_options"] = self.adjoint_options

        y = self.odeint(f, y0, t, **kwargs)

        return y


class SolverStats:
    def __init__(self, batch_size: int):
        super().__init__()

        self.eval_t = [[] for _ in range(batch_size)]

    @property
    def total_nfe(self):
        return sum(len(t) for t in self.eval_t)

    @property
    def mean_nfe(self):
        return self.total_nfe / len(self.eval_t)


@dataclass
class ODEQuery:
    """A batched query of ODE problems.

    `t` contains the time of `u0` followed by the prediction time steps.
    """

    t: TensorType["batch", "time"]
    u0: TensorType["batch", "node", "feature"]
    time_encoder: TimeEncoder

    @property
    def batch_size(self):
        return self.t.shape[0]


class WithTimeEncoding(nn.Module):
    def __init__(self, dynamics, query: ODEQuery, stats: SolverStats, instance: int):
        super().__init__()

        self.dynamics = dynamics
        self.query = query
        self.stats = stats
        self.instance = instance

        self.t_start = torch.amin(query.t, dim=1)
        self.t_span = torch.amax(query.t, dim=1) - self.t_start

    def forward(self, t: TensorType[float], u: TensorType["batch", "node", "feature"]):
        idx = self.instance

        # Transform t from 0..1 to the actual integration interval
        actual_t = self.t_start + t * self.t_span
        self.stats.eval_t[idx].append(actual_t.detach()[idx])

        true_time = self.query.time_encoder.encode(actual_t)[idx : idx + 1]
        du = self.dynamics(true_time, u)

        # Scale du to account for the squeezed integration range of 0..1
        return du * self.t_span[idx]


def solve_ode(
    dynamics, solver: ODESolver, query: ODEQuery
) -> tuple[TensorType["batch", "time", "node", "feature"], SolverStats]:
    # Ensure that we have at least one query in the batch and at least one target time
    # step.
    assert query.t.shape[0] >= 1
    assert query.t.shape[1] >= 2

    # Iterate over the instances in the batch so that the instances do not interact in the
    # solver. If we would apply the solver over the whole batch at once, the instances
    # would influence the total error estimate, therefore interact with each other. I have
    # observed this to trigger much more function evaluations than would be necessary if
    # the instances were integrated individually.
    #
    # This is very much a restriction of torchdiffeq and if we ever get truly parallel
    # solving as in diffrax for example, this loop can be vectorized again.
    us = []
    stats = SolverStats(query.batch_size)
    for instance in range(query.batch_size):
        with_time_enc = WithTimeEncoding(dynamics, query, stats, instance)
        # Integrate over 0..1 so that the initial step size heuristic makes reasonable
        # suggestions. At least to my eye, there are some suspicious constants in there
        # that probably don't generalize well to arbitrary integration intervals.
        t = query.t[instance]
        t_min = t.min()
        eval_t = (t - t_min) / (t.max() - t_min)
        u = solver(with_time_enc, query.u0[instance : instance + 1], eval_t)

        # The first step is not actually a prediction but just the input
        u = u[1:]

        us.append(u)

    return eo.rearrange(torch.cat(us, dim=1), "t b n f -> b t n f"), stats
