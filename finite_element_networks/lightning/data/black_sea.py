import logging
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops as eo
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from more_itertools import chunked
from scipy.spatial import Delaunay
from torch.utils.data import DataLoader, Dataset, Sampler
from torchtyping import TensorType

from ...data import DomainInfo, PeriodicEncoder, Standardizer, STBatch
from ...domain import CellPredicate, Domain
from .common import MeshConfig, sample_mesh

log = logging.getLogger("fen.black_sea")

MOTUCLIENT_INSTRUCTIONS = """
The Black Sea dataset needs to be downloaded with motuclient from the Copernicus
Marine Service. To get access, register an account on their website [1] and put
your credentials into your ~/.netrc as follows.

machine my.cmems-du.eu
  login <your-user>
  password <your-password>

[1] https://marine.copernicus.eu
""".strip()


@dataclass(frozen=True)
class BlackSeaBatchKey:
    ranges: list[tuple[Optional[int], Optional[int]]]
    context_steps: int


@dataclass
class BlackSeaStats:
    mean: TensorType["time", "feature"]
    std: TensorType["time", "feature"]

    def get_standardizer(self, t: TensorType["batch", "time"]):
        day = t.long() % 365
        mean = self.mean[day]
        std = self.std[day]
        return Standardizer(
            eo.repeat(mean, "b t f -> b t 1 f"), eo.repeat(std, "b t f -> b t 1 f")
        )


class BlackSeaDataset(Dataset):
    def __init__(
        self, file: Path, domain: Domain, domain_info: DomainInfo, stats: BlackSeaStats
    ):
        super().__init__()

        self.file = file
        self.domain = domain
        self.domain_info = domain_info
        self.stats = stats

        data = torch.load(self.file)
        self.t = torch.from_numpy(data["t"]).float()
        self.u = torch.from_numpy(data["u"]).float()

        self.time_encoder = PeriodicEncoder(
            base=torch.tensor(np.datetime64("2012-01-01").astype(float)).float(),
            period=torch.tensor(365.0).float(),
        )

    def __getitem__(self, key: BlackSeaBatchKey) -> STBatch:
        t = []
        u = []
        for start, end in key.ranges:
            slice_ = slice(start, end)
            t.append(self.t[slice_])
            u.append(self.u[slice_])

        t = torch.stack(t)
        return STBatch(
            domain=self.domain,
            domain_info=self.domain_info,
            t=t,
            u=torch.stack(u),
            context_steps=key.context_steps,
            standardizer=self.stats.get_standardizer(t),
            time_encoder=self.time_encoder,
        )


class BlackSeaSampler(Sampler):
    def __init__(
        self,
        dataset: BlackSeaDataset,
        target_steps: int,
        context_steps: int,
        batch_size: int,
        shuffle: bool,
    ):
        super().__init__(dataset)

        self.dataset = dataset
        self.target_steps = target_steps
        self.context_steps = context_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        seq_len = target_steps + context_steps
        self.indices = [
            (start, start + seq_len) for start in range(len(self.dataset.t) - seq_len)
        ]

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            indices = indices.copy()
            random.shuffle(indices)
        for chunk in chunked(indices, self.batch_size):
            yield BlackSeaBatchKey(ranges=chunk, context_steps=self.context_steps)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)


class OutOfDomainPredicate(CellPredicate):
    """Filter out all mesh cells that include mostly out-of-domain points."""

    def __init__(self, tri: Delaunay, x: np.ndarray, in_domain: np.ndarray):
        """
        Arguments
        ---------
        tri
            A mesh defined over some points
        x
            A set of "trial points" to check the mesh cells against
        in_domain
            A mask that says which points in `x` are in-domain
        """

        vertices = tri.points[tri.simplices]

        a, b, c = np.split(vertices, 3, axis=-2)
        ab, bc, ca = b - a, c - b, a - c
        ax, bx, cx = x - a, x - b, x - c

        # A point is inside a triangle if all these cross-products have the same sign
        abx = np.cross(ab, ax)
        bcx = np.cross(bc, bx)
        cax = np.cross(ca, cx)
        inside = ((abx * bcx) >= 0) & ((bcx * cax) >= 0) & ((cax * abx) >= 0)

        n_in_domain_in_cell = np.logical_and(inside, in_domain).sum(axis=-1)
        n_out_of_domain_in_cell = np.logical_and(inside, ~in_domain).sum(axis=-1)
        self.filter = n_out_of_domain_in_cell > n_in_domain_in_cell

    def __call__(self, cell_idx, cell, boundary_faces):
        return self.filter[cell_idx]


class BlackSeaDataModule(pl.LightningDataModule):
    """The Black Sea dataset available from Copernicus Marine Service [1].

    [1] https://resources.marine.copernicus.eu/product-detail/BLKSEA_MULTIYEAR_PHY_007_004
    """

    def __init__(
        self,
        root: Path,
        domain_preprocessing,
        *,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: int = 8,
        context_steps: int = 1,
        train_target_steps: int = 10,
        eval_target_steps: int = 10,
        raw_root: Optional[Path] = None,
        mesh_config: Optional[MeshConfig] = None,
    ):
        super().__init__()

        self.root = root
        self.raw_root = root / "raw" if raw_root is None else raw_root
        self.domain_preprocessing = domain_preprocessing
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.context_steps = context_steps
        self.train_target_steps = train_target_steps
        self.eval_target_steps = eval_target_steps
        self.mesh_config = (
            self.default_mesh_config() if mesh_config is None else mesh_config
        )
        self.mesh_file = self.root / "domain.pt"
        self.stats_file = self.root / "stats.npz"

        self.domain = None
        self.domain_info = None
        self.stats = None

    def prepare_data(self):
        self.root.mkdir(exist_ok=True, parents=True)

        # Don't check for raw data if the processed data is available
        files = [self.mesh_file, self.stats_file] + [
            self.root / f"{stage}.pt" for stage in ["train", "val", "test"]
        ]
        if all(f.is_file() for f in files):
            return

        self._download_raw_data()
        self._mesh_subset_of_grid()
        self._process_data()

    def setup(self, stage: Optional[str] = None):
        if self.domain is None:
            self.domain = torch.load(self.mesh_file)["domain"].normalize()
            self.domain_info = self.domain_preprocessing(self.domain)
        if self.stats is None:
            data = np.load(self.stats_file)
            mean, std = (
                torch.from_numpy(data["mean"]).float(),
                torch.from_numpy(data["std"]).float(),
            )
            self.stats = BlackSeaStats(mean, std)

        if stage in (None, "train", "fit"):
            self.train_data = BlackSeaDataset(
                self.root / "train.pt", self.domain, self.domain_info, self.stats
            )
        if stage in (None, "validate", "fit"):
            self.val_data = BlackSeaDataset(
                self.root / "val.pt", self.domain, self.domain_info, self.stats
            )
        if stage in (None, "test"):
            self.test_data = BlackSeaDataset(
                self.root / "test.pt", self.domain, self.domain_info, self.stats
            )

    def train_dataloader(self):
        return self._dataloader(
            self.train_data, target_steps=self.train_target_steps, shuffle=True
        )

    def val_dataloader(self):
        return self._dataloader(
            self.val_data, target_steps=self.eval_target_steps, shuffle=False
        )

    def test_dataloader(self):
        return self._dataloader(
            self.test_data, target_steps=self.eval_target_steps, shuffle=False
        )

    def _dataloader(self, dataset, *, target_steps, shuffle):
        sampler = BlackSeaSampler(
            dataset,
            target_steps=target_steps,
            context_steps=self.context_steps,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_interesting_batch(self):
        start = 130
        return self.val_data[
            BlackSeaBatchKey(
                ranges=[(start - self.context_steps, start + 15)],
                context_steps=self.context_steps,
            )
        ]

    @staticmethod
    def default_mesh_config():
        # Arbitrarily chosen, fixed seed
        return MeshConfig(k=1000, epsilon=10.0, seed=803452658411725)

    def download_motu(self, product_id: str, variables: list[str], target: Path):
        variable_options = []
        for var in variables:
            variable_options.append("--variable")
            variable_options.append(var)
        cmd = [
            "motuclient",
            "--motu",
            "https://my.cmems-du.eu/motu-web/Motu",
            "--service-id",
            "BLKSEA_MULTIYEAR_PHY_007_004-TDS",
            "--product-id",
            product_id,
            "--longitude-min",
            "27.37",
            "--longitude-max",
            "41.9626",
            "--latitude-min",
            "40.86",
            "--latitude-max",
            "46.8044",
            "--date-min",
            f"2012-01-01 12:00:00",
            "--date-max",
            f"2019-12-31 12:00:00",
            "--depth-min",
            "12.5",
            "--depth-max",
            "12.5362",
            *variable_options,
            "--out-dir",
            str(target.parent),
            "--out-name",
            target.name,
            "--config-file",
            "~/.config/motuclient/motuclient-python.ini",
        ]
        subprocess.run(cmd, check=True)
        # motuclient does not indicate failure in its exit code
        assert target.is_file()

    def _download_raw_data(self):
        self.raw_root.mkdir(exist_ok=True, parents=True)
        features = [
            ("thetao", "bs-cmcc-tem-rean-d", ["thetao"]),
            ("uo", "bs-cmcc-cur-rean-d", ["uo"]),
            ("vo", "bs-cmcc-cur-rean-d", ["vo"]),
        ]
        jobs = [
            (product_id, variables, target)
            for feature, product_id, variables in features
            if not (target := self.raw_root / f"{feature}.nc").is_file()
        ]
        if len(jobs) == 0:
            return
        print(MOTUCLIENT_INSTRUCTIONS)
        for args in jobs:
            self.download_motu(*args)

    def _mesh_subset_of_grid(self):
        if self.mesh_file.is_file():
            existing_config = torch.load(self.mesh_file)["config"]
            if existing_config != self.mesh_config:
                raise RuntimeError(
                    "Mesh already exists but has been created with a different configuration! \n\n"
                    f"existing: {existing_config}\n"
                    f"     new: {self.mesh_config}"
                )
            else:
                return

        log.info("Generate random sub-mesh to sample on.")
        ds = xr.open_mfdataset(f"{self.raw_root}/*.nc")
        lat, lon = np.array(ds.lat), np.array(ds.lon)

        # Construct all grid points
        x, y = np.meshgrid(lon, lat)
        grid_points = np.vstack((x.ravel(), y.ravel())).T

        # Find points where data is available in all years
        temperature = ds.thetao
        mask = temperature.to_masked_array().mask
        in_domain = np.logical_and.reduce(~mask, axis=0).ravel()

        valid_points = grid_points[in_domain]

        def predicate(tri: Delaunay):
            # Filter out mesh boundary cells that are too acute or contain mostly land
            angle_predicate = self.mesh_config.angle_predicate(tri)
            ood_predicate = OutOfDomainPredicate(tri, grid_points, in_domain)

            return lambda *args: ood_predicate(*args) or angle_predicate(*args)

        node_indices, domain = sample_mesh(self.mesh_config, valid_points, predicate)

        # Translate node indices from `valid_points` to `grid_points`
        node_indices = np.nonzero(in_domain)[0][node_indices]

        data = {
            "config": self.mesh_config,
            "domain": domain,
            "all_points": grid_points,
            "in_domain": in_domain,
            "node_indices": node_indices,
        }
        torch.save(data, self.mesh_file)

    def _process_data(self):
        if self.stats_file.is_file() and all(
            self.root / f"{stage}.pt" for stage in ["train", "val", "test"]
        ):
            return

        ds = xr.open_mfdataset(f"{self.raw_root}/*.nc")
        time = np.array(ds.time)
        velocity_east = ds.uo
        velocity_north = ds.vo
        temperature = ds.thetao
        features = (velocity_east.values, velocity_north.values, temperature.values)
        u = np.stack(features, axis=-1)

        u = eo.rearrange(u, "t 1 y x f -> t (y x) f")
        node_indices = torch.load(self.mesh_file)["node_indices"]
        u = u[:, node_indices]

        assert not np.isnan(u).any()

        jan_1st = lambda year: np.datetime64(f"{year}-01-01")
        train = time < jan_1st(2018)
        val = (time >= jan_1st(2018)) & (time < jan_1st(2019))
        test = time >= jan_1st(2019)

        # Convert timestamps to days
        t = time.astype("datetime64[D]").astype(float)

        torch.save({"t": t[train], "u": u[train]}, self.root / "train.pt")
        torch.save({"t": t[val], "u": u[val]}, self.root / "val.pt")
        torch.save({"t": t[test], "u": u[test]}, self.root / "test.pt")

        mean, std = self._compute_stats(t[train], u[train])
        np.savez(self.stats_file, mean=mean, std=std)

    def _compute_stats(self, t: np.ndarray, u: np.ndarray):
        n_nodes = u.shape[1]
        velocity, temperature = u[..., :2], u[..., 2]

        # For the temperature we compute separate stats for each calendar day because
        # there is a strong periodicity throughout the year. Also we just ignore leap
        # years.
        day = t.astype(int) % 365

        # Count the number of training samples per calendar day
        counts = np.zeros(365)
        np.add.at(counts, day, 1)

        sums = np.zeros(365)
        np.add.at(sums, day, temperature.sum(axis=1))
        daily_temperature_mean = sums / counts / n_nodes

        sums = np.zeros(365)
        np.add.at(
            sums,
            day,
            ((temperature - daily_temperature_mean[day, None]) ** 2).sum(axis=1),
        )
        daily_temperature_std = np.sqrt(sums / counts / n_nodes)

        all_velocities = eo.rearrange(velocity, "t n f -> (t n) f")
        mean = np.concatenate(
            (
                eo.repeat(np.mean(all_velocities, axis=0), "f -> 365 f"),
                daily_temperature_mean[:, None],
            ),
            axis=-1,
        )
        std = np.concatenate(
            (
                eo.repeat(np.std(all_velocities, axis=0), "f -> 365 f"),
                daily_temperature_std[:, None],
            ),
            axis=-1,
        )
        return mean, std
