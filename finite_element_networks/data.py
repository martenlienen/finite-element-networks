from dataclasses import dataclass
from typing import Generic, TypeVar, Union

import torch
from torchtyping import TensorType

from .domain import Domain


class TimeEncoder:
    """A time encoder encodes strictly increasing timestamps into other time
    representations.

    This can, for example, be used to implement periodically recurring behavior in
    non-autonomous models.
    """

    def encode(self, t: TensorType["batch"]) -> TensorType["batch", "time_encoding"]:
        raise NotImplementedError()


@dataclass
class InvariantEncoder(TimeEncoder):
    """Make all sequences start at 0 when they are translation invariant."""

    start: TensorType["batch"]

    def encode(self, t: TensorType["batch"]) -> TensorType["batch", "time_encoding"]:
        return (t - self.start).unsqueeze(dim=-1)


@dataclass
class PeriodicEncoder(TimeEncoder):
    """Encode linear, periodic time into 2-dimensional positions on a circle.

    This ensures that the time features have a smooth transition at the periodicity
    boundary, e.g. new year's eve for data with yearly periodicity.
    """

    base: TensorType["batch"]
    period: TensorType["batch"]

    def encode(self, t: TensorType["batch"]) -> TensorType["batch", "time_encoding"]:
        phase = (t - self.base) * (2 * torch.pi / self.period)
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


StandardizerTensor = Union[
    TensorType["feature"],
    TensorType["node", "feature"],
    TensorType["time", "node", "feature"],
    TensorType["batch", "time", "node", "feature"],
]
DomainInfo = TypeVar("DomainInfo")


@dataclass
class Standardizer:
    """Node feature standardization

    The standardization means and standard deviation can optionally be node-, time-, and
    batch-specific.
    """

    mean: StandardizerTensor
    std: StandardizerTensor

    def __post_init__(self):
        assert self.mean.shape == self.std.shape
        assert 1 <= self.mean.ndim <= 4

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def slice_time(self, start: int, end: int):
        if self.mean.ndim == 3:
            return Standardizer(self.mean[start:end], self.std[start:end])
        elif self.mean.ndim == 4:
            return Standardizer(self.mean[:, start:end], self.std[:, start:end])
        else:
            return self


@dataclass
class STBatch(Generic[DomainInfo]):
    """A batch of spatio-temporal data.

    This type is the common interface for the provided and custom data loaders.

    Attributes
    ----------
    domain
        The domain that the data in this batch is from.
    domain_info
        Precomputed information domain information that the model needs for training as
        tensors so that it can be transferred to the GPU.
    t
        Batched time steps
    u
        Batched sensor data
    context_steps
        Says how many time steps in the beginning of the batch are meant as context for
        the prediction. The rest is meant as prediction targets.
    standardizer
        Node feature standardizer
    time_encoder
        Encodes the time steps in `t` into the values that are actually fed to
        non-autonomous models.
    """

    domain: Domain
    domain_info: DomainInfo
    t: TensorType["batch", "time"]
    u: TensorType["batch", "time", "node", "feature"]
    context_steps: int
    standardizer: Standardizer
    time_encoder: TimeEncoder

    @property
    def batch_size(self) -> int:
        return self.u.shape[0]

    @property
    def context_t(self) -> TensorType["batch", "context_time"]:
        return self.t[:, : self.context_steps]

    @property
    def target_t(self) -> TensorType["batch", "target_time"]:
        return self.t[:, self.context_steps :]

    @property
    def context_u(self) -> TensorType["batch", "context_time", "node", "feature"]:
        return self.u[:, : self.context_steps]

    @property
    def target_u(self) -> TensorType["batch", "target_time", "node", "feature"]:
        return self.u[:, self.context_steps :]

    @property
    def context_standardizer(self):
        return self.standardizer.slice_time(0, self.context_steps)

    @property
    def target_standardizer(self):
        return self.standardizer.slice_time(self.context_steps, None)
