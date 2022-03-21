"""PyTorch reference implementation of Finite Element Networks"""

__version__ = "1.0.0"

from .data import STBatch
from .domain import Domain
from .fen import FEN, FENDomainInfo, FENDynamics, FENQuery, FreeFormTerm, TransportTerm
from .mlp import MLP
from .ode_solver import ODESolver
