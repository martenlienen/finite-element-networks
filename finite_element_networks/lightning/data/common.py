from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import Delaunay

from ...data import TimeEncoder
from ...domain import (
    BoundaryAnglePredicate,
    CellPredicate,
    Domain,
    select_boundary_mesh_cells,
)


@dataclass(frozen=True)
class MeshConfig:
    """Configuration for the generation of a sparse mesh from a larger set of points.

    Attributes
    ----------
    k
        Number of nodes to select
    epsilon
        Maximum angle of boundary cells to filter out in degrees
    seed
        Random seed for reproducibility
    """

    k: int
    epsilon: float
    seed: int

    def random_state(self):
        return np.random.RandomState(int(self.seed) % 2**32)

    def epsilon_radians(self):
        return self.epsilon * np.pi / 180

    def angle_predicate(self, tri: Delaunay):
        return BoundaryAnglePredicate(tri.points, self.epsilon_radians())


def sample_mesh(
    config: MeshConfig,
    points: np.ndarray,
    predicate_factory: Optional[Callable[[Delaunay], CellPredicate]] = None,
) -> tuple[np.ndarray, Domain]:
    """Create a domain from a subset of points, optionally filtering out some cells.

    Returns
    -------
    Indices of the points that were selected as mesh nodes and the domain
    """

    import skfem
    from sklearn_extra.cluster import KMedoids

    # Select k sparse observation points uniformly-ish
    km = KMedoids(
        n_clusters=config.k, init="k-medoids++", random_state=config.random_state()
    )
    km.fit(points)
    node_indices = km.medoid_indices_

    # Mesh the points with Delaunay triangulation
    tri = Delaunay(points[node_indices])

    # Filter out mesh boundary cells that are too acute or contain mostly land
    if predicate_factory is not None:
        predicate = predicate_factory(tri)
        filter = select_boundary_mesh_cells(tri, predicate)
        tri.simplices = tri.simplices[~filter]

    # Ensure that every node is in at least one mesh cell
    cell_counts = np.zeros(config.k, dtype=int)
    np.add.at(cell_counts, tri.simplices, 1)
    assert all(cell_counts >= 1)

    mesh = skfem.MeshTri(
        np.ascontiguousarray(tri.points.T), np.ascontiguousarray(tri.simplices.T)
    )
    domain = Domain(tri.points, mesh=mesh)
    return node_indices, domain
