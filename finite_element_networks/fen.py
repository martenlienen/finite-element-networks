from dataclasses import dataclass
from typing import Optional

import einops as eo
import numpy as np
import skfem
import torch
import torch.nn as nn
from cachetools import cachedmethod
from skfem.assembly.form.form import FormExtraParams
from skfem.helpers import grad
from torch_scatter import scatter
from torchtyping import TensorType

from .data import STBatch, TimeEncoder
from .domain import Domain
from .mlp import MLP
from .ode_solver import ODEQuery, ODESolver, solve_ode


@skfem.BilinearForm
def mass_form(u, v, w):
    return u * v


def lumped_mass_matrix(domain: Domain) -> TensorType["node"]:
    """Compute the diagonal of the lumped mass matrix."""

    A = skfem.asm(mass_form, domain.basis)
    return torch.from_numpy(np.array(A.sum(axis=1))[:, 0])


def assemble_linear_cell_contributions(
    basis: skfem.assembly.Basis, form: skfem.assembly.LinearForm
) -> tuple[TensorType["vertex", "cell"], TensorType["vertex", "cell"]]:
    """Integrate a linear form over all basis functions over all mesh cells.

    This is adapated from `skfem.LinearForm._assemble`.

    The returned rows tell you which coefficient from each cell contributes to which node.
    """

    def kernel(v, w, dx):
        return np.sum(form.form(*v, w) * dx, axis=1)

    vbasis = basis

    nt = vbasis.nelems
    dx = vbasis.dx
    w = FormExtraParams(vbasis.default_parameters())

    # initialize COO data structures
    data = np.zeros((vbasis.Nbfun, nt), dtype=np.float64)
    rows = np.zeros((vbasis.Nbfun, nt), dtype=np.int64)

    for i in range(vbasis.Nbfun):
        rows[i] = vbasis.element_dofs[i]
        data[i] = kernel(vbasis.basis[i], w, dx)

    assert np.all(rows == vbasis.mesh.t)

    rows = torch.from_numpy(rows).long()
    data = torch.from_numpy(data).float()

    return data, rows


def assemble_free_form_parts(domain: Domain):
    """Compute the free-form inner products of test functions."""

    @skfem.LinearForm
    def test_function(v, w):
        return v

    return assemble_linear_cell_contributions(domain.basis, test_function)


def assemble_bilinear_cell_contributions(
    basis: skfem.assembly.Basis, form: skfem.assembly.BilinearForm
) -> tuple[
    TensorType["j", "i", "cell", "space"],
    TensorType["j", "i", "cell"],
    TensorType["j", "i", "cell"],
]:
    """Integrate a bilinear form over all pairs of basis functions over all mesh cells.

    This is adapated from `skfem.BilinearForm._assemble`.

    In the output shapes, `j` is the basis function associated with the trial function u
    and `i` is associated with the test function v.
    """

    def kernel(u, v, w, dx):
        return np.sum(form.form(*u, *v, w) * dx, axis=1)

    ubasis = vbasis = basis

    nt = ubasis.nelems
    dx = ubasis.dx
    wdict = FormExtraParams(ubasis.default_parameters())

    # initialize COO data structures
    data = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=np.float64)
    rows = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=np.int64)
    cols = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=np.int64)

    # loop over the indices of local stiffness matrix
    for j in range(ubasis.Nbfun):
        for i in range(vbasis.Nbfun):
            rows[j, i] = vbasis.element_dofs[i]
            cols[j, i] = ubasis.element_dofs[j]
            data[j, i] = kernel(ubasis.basis[j], vbasis.basis[i], wdict, dx)

    rows = torch.from_numpy(rows).long()
    cols = torch.from_numpy(cols).long()
    data = torch.from_numpy(data).float()

    return data, rows, cols


def assemble_convection_parts(domain: Domain):
    """Compute the convection inner products of trial and test function for each pair of
    cell vertices.
    """

    parts = []
    for i in range(domain.dim):

        @skfem.BilinearForm
        def convection_component(u, v, w, i=i):
            return grad(u)[i] * v

        data, rows, cols = assemble_bilinear_cell_contributions(
            domain.basis, convection_component
        )
        parts.append(data)

    return -torch.stack(parts, dim=-1), rows, cols


@dataclass
class FENDomainInfo:
    """Precomputed domain attributes relevant to FENs."""

    @staticmethod
    def from_domain(domain: Domain):
        """Construct a FEN specific domain from a general domain object."""

        n_nodes = len(domain)

        node_pos = torch.from_numpy(domain.x).float()
        triangulation = torch.from_numpy(domain.mesh.t.T)
        vertex_pos: TensorType["cell", "vertex", "space"] = node_pos[triangulation]
        cell_centers = vertex_pos.mean(dim=1)
        cell_local_vertex_pos = vertex_pos - cell_centers.unsqueeze(dim=1)

        inverse_lumped_mass_matrix = (1 / lumped_mass_matrix(domain)).float()
        fixed_values_mask = (
            torch.from_numpy(domain.fixed_values_mask)
            if domain.fixed_values_mask is not None
            else None
        )
        free_form_parts = assemble_free_form_parts(domain)
        convection_parts = assemble_convection_parts(domain)

        # Assert that the parts are in the same order as the vertices in the
        # triangulation, so that we can rely on that order implicitly when combining the
        # parts with learned coefficients.
        _, free_form_rows = free_form_parts
        assert torch.all(free_form_rows.T == triangulation)

        return FENDomainInfo(
            n_nodes=n_nodes,
            triangulation=triangulation,
            cell_centers=cell_centers,
            cell_local_vertex_pos=cell_local_vertex_pos,
            inverse_lumped_mass_matrix=inverse_lumped_mass_matrix,
            fixed_values_mask=fixed_values_mask,
            free_form_parts=free_form_parts,
            convection_parts=convection_parts,
        )

    n_nodes: int
    triangulation: TensorType["cell", "vertex"]
    cell_centers: TensorType["cell", "space"]
    cell_local_vertex_pos: TensorType["cell", "vertex", "space"]
    inverse_lumped_mass_matrix: TensorType["node"]
    fixed_values_mask: Optional[TensorType["node"]]
    free_form_parts: tuple[TensorType["vertex", "cell"], TensorType["vertex", "cell"]]
    convection_parts: tuple[
        TensorType["j", "i", "cell", "space"],
        TensorType["j", "i", "cell"],
        TensorType["j", "i", "cell"],
    ]

    @property
    def n_vertices(self):
        return self.triangulation.shape[-1]

    @property
    def space_dim(self):
        return self.cell_centers.shape[-1]


@dataclass
class FENQuery:
    """A prediction query for an FEN model.

    `t` contains the time of `u0` followed by the prediction time steps.
    """

    domain: FENDomainInfo
    t: TensorType["batch", "time"]
    u0: TensorType["batch", "node", "feature"]
    time_encoder: TimeEncoder

    @property
    def batch_size(self):
        return self.t.shape[0]

    @staticmethod
    def from_batch(batch: STBatch[FENDomainInfo], *, standardize: bool = True):
        u = batch.context_u
        if standardize:
            u = batch.context_standardizer.do(u)
        u0 = u[:, -1]
        t = batch.t[:, batch.context_steps - 1 :]
        return FENQuery(batch.domain_info, t, u0, batch.time_encoder)


class SystemState:
    """The current state of a physical system, possibly batched.

    Attributes
    ----------
    domain
        The domain mesh that the system is discretized on
    t
        The current timestamp
    u
        The node features
    """

    def __init__(
        self,
        domain: FENDomainInfo,
        t: TensorType["batch", "time_encoding"],
        u: TensorType["batch", "node", "feature"],
    ):
        self.domain = domain
        self.t = t
        self.u = u
        self._cache = {}

    # Cache the cell features to minimize the backwards graph
    @cachedmethod(lambda self: self._cache)
    def cell_features(
        self, stationary: bool, autonomous: bool
    ) -> TensorType["batch", "cell", "cell-feature"]:
        """Assemble the feature matrix for each cell."""

        T = self.domain.triangulation
        vertex_pos = self.domain.cell_local_vertex_pos
        vertex_features = self.u[:, T]

        ncells = T.shape[0]
        batch_size = self.u.shape[0]

        # Collect all the information about a cell into a per-cell feature matrix
        cell_features = [
            eo.repeat(vertex_pos, "c v s -> b c (v s)", b=batch_size),
            eo.rearrange(vertex_features, "b c v f -> b c (v f)"),
        ]
        if not stationary:
            cell_pos = self.domain.cell_centers
            cell_features.insert(0, eo.repeat(cell_pos, "c s -> b c s", b=batch_size))
        if not autonomous:
            time = self.t
            cell_features.insert(0, eo.repeat(time, "b f -> b c f", c=ncells))

        return torch.cat(cell_features, dim=-1)


class PDETerm(nn.Module):
    def forward(
        self, state: SystemState
    ) -> TensorType["batch", "cell", "vertex", "feature"]:
        pass


class FreeFormTerm(PDETerm):
    """A PDE term that does not make any assumptions on the form of the dynamics."""

    @staticmethod
    def build_coefficient_mlp(
        *,
        n_features: int,
        space_dim: int,
        time_dim: int,
        hidden_dim: int,
        n_layers: int,
        non_linearity,
        stationary: bool,
        autonomous: bool,
    ):
        """Build an MLP to estimate the free-form coefficients with the correct in/out
        dimensions.
        """
        n_vertices = space_dim + 1
        extra_in = 0
        if not stationary:
            extra_in += space_dim
        if not autonomous:
            extra_in += time_dim
        return MLP(
            in_dim=n_vertices * (space_dim + n_features) + extra_in,
            out_dim=n_vertices * n_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

    def __init__(
        self,
        coefficient_mlp: MLP,
        *,
        stationary: bool,
        autonomous: bool,
        zero_init: bool,
    ):
        super().__init__()

        self.coefficient_mlp = coefficient_mlp
        self.stationary = stationary
        self.autonomous = autonomous
        self.zero_init = zero_init

        # The zero (constant) dynamic is often a good first approximation, so we start
        # with that.
        if self.zero_init:
            self.coefficient_mlp[-1].weight.data.zero_()
            self.coefficient_mlp[-1].bias.data.zero_()

    def forward(self, state: SystemState):
        return self.build_messages(state.domain, self.estimate_coefficients(state))

    def estimate_coefficients(
        self, state: SystemState
    ) -> TensorType["batch", "cell", "vertex", "feature"]:
        """Estimate the free-form coefficients as in Equation (16)."""
        return eo.rearrange(
            self.coefficient_mlp(state.cell_features(self.stationary, self.autonomous)),
            "b c (v f) -> b c v f",
            v=state.domain.n_vertices,
        )

    def build_messages(
        self,
        domain: FENDomainInfo,
        coefficients: TensorType["batch", "cell", "vertex", "feature"],
    ) -> TensorType["batch", "cell", "vertex", "feature"]:
        """Build the free-form messages as n Equation (17)."""
        free_form_data, _ = domain.free_form_parts
        return torch.einsum("vc,bcvf->bcvf", free_form_data, coefficients)


class TransportTerm(PDETerm):
    """A PDE term that models transport through convection."""

    @staticmethod
    def build_flow_field_mlp(
        *,
        n_features: int,
        space_dim: int,
        time_dim: int,
        hidden_dim: int,
        n_layers: int,
        non_linearity,
        stationary: bool,
        autonomous: bool,
    ):
        """Build an MLP to estimate the flow field with the correct in/out dimensions."""
        n_vertices = space_dim + 1
        extra_in = 0
        if not stationary:
            extra_in += space_dim
        if not autonomous:
            extra_in += time_dim
        return MLP(
            in_dim=n_vertices * (space_dim + n_features) + extra_in,
            out_dim=n_features * space_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

    def __init__(
        self,
        flow_field_mlp: MLP,
        *,
        stationary: bool,
        autonomous: bool,
        zero_init: bool,
    ):
        super().__init__()

        self.flow_field_mlp = flow_field_mlp
        self.stationary = stationary
        self.autonomous = autonomous
        self.zero_init = zero_init

        # The zero (constant) dynamic is often a good first approximation, so we start
        # with that.
        if self.zero_init:
            self.flow_field_mlp[-1].weight.data.zero_()
            self.flow_field_mlp[-1].bias.data.zero_()

    def forward(self, state: SystemState):
        flow_field = self.estimate_flow_field(state)
        return self.build_messages(state.domain, flow_field, state.u)

    def estimate_flow_field(
        self, state: SystemState
    ) -> TensorType["batch", "cell", "feature", "space"]:
        """Estimate the cell-wise velocity field as in Equation (20)."""
        return eo.rearrange(
            self.flow_field_mlp(state.cell_features(self.stationary, self.autonomous)),
            "b c (f s) -> b c f s",
            s=state.domain.space_dim,
        )

    def build_messages(
        self,
        domain: FENDomainInfo,
        flow_field: TensorType["batch", "cell", "feature", "space"],
        u: TensorType["batch", "node", "feature"],
    ) -> TensorType["batch", "cell", "vertex", "feature"]:
        """Build the transport messages as in Equation (21)."""

        convection_data, _, convection_cols = domain.convection_parts

        # The contration order was optimized with opt-einsum, but the torch implementation
        # runs faster (according to %timeit), so we use torch.einsum anyway.
        return torch.einsum(
            "jics,bjicf,bcfs -> bcif",
            convection_data,
            u[:, convection_cols, :],
            flow_field,
        )


class FENDynamics(nn.Module):
    def __init__(self, terms: list[PDETerm]):
        super().__init__()

        assert len(terms) > 0
        self.terms = nn.ModuleList(terms)

    def forward(self, state: SystemState):
        msgs = sum(term(state) for term in self.terms)
        du = self._send_msgs(msgs, state.domain)

        # Don't change anything for fixed values by setting the derivative to 0
        if state.domain.fixed_values_mask is not None:
            du = torch.where(state.domain.fixed_values_mask, du.new_zeros(1), du)

        return du

    def _send_msgs(self, cell_msgs, domain: FENDomainInfo):
        # Send the messages from each cell to its vertices
        msgs = eo.rearrange(cell_msgs, "b c v f -> b (c v) f")
        target = domain.triangulation.ravel()
        received: TensorType["batch", "node", "feature"] = scatter(
            msgs, target, dim=1, reduce="sum", dim_size=domain.n_nodes
        )

        return torch.einsum("bnf,n -> bnf", received, domain.inverse_lumped_mass_matrix)

    @property
    def free_form_terms(self):
        return [term for term in self.terms if isinstance(term, FreeFormTerm)]

    @property
    def transport_terms(self):
        return [term for term in self.terms if isinstance(term, TransportTerm)]


class WithDomain(nn.Module):
    def __init__(self, dynamics: FENDynamics, domain: FENDomainInfo):
        super().__init__()

        self.dynamics = dynamics
        self.domain = domain

    def forward(self, t, u):
        return self.dynamics(SystemState(self.domain, t, u))


class FEN(nn.Module):
    def __init__(self, dynamics: FENDynamics, ode_solver: ODESolver):
        super().__init__()

        self.dynamics = dynamics
        self.ode_solver = ode_solver

        self.stats = None

    def forward(
        self, query: FENQuery
    ) -> TensorType["batch", "time", "node", "feature"]:
        dynamics = WithDomain(self.dynamics, query.domain)
        ode_query = ODEQuery(query.t, query.u0, query.time_encoder)
        u_hat, self.stats = solve_ode(dynamics, self.ode_solver, ode_query)
        return u_hat
