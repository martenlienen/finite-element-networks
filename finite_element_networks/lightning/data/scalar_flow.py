import logging
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from more_itertools import chunked
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from ...data import DomainInfo, InvariantEncoder, Standardizer, STBatch
from ...domain import Domain
from .common import MeshConfig, sample_mesh
from .utils import MeanStdAccumulator

log = logging.getLogger("fen.scalar_flow")

NOTICE = """The ScalarFlow dataset comes with the following notice:

    ScalarFlow, https://ge.in.tum.de/publications/2019-scalarflow-eckert/

    This data may only be used after agreeing to the terms of use:
    https://docs.google.com/forms/d/e/1FAIpQLSeC7nW4pvlXdkHee034ytG4UYBUtpN9DHBGRmQ4EZJdGGGG9w/viewform?usp=sf_link
    (the license is repeated again below)

    If you use a dataset please cite our paper:

    @article{ScalarFlow2019, author = {Marie-Lena Eckert, Kiwon Um, Nils Thuerey}, title = {ScalarFlow: A Large-Scale Volumetric Data Set of Real-world Scalar Transport Flows for Computer Animation and Machine Learning}, journal={ACM Transactions on Graphics}, volume={38(6):239}, year={2019}}

The license referred to is the CC-BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/

Type y or yes if you agree to their terms of use: """


def download_raw_data(root: Path):
    agree = input(NOTICE)
    if agree.lower() not in ("y", "yes"):
        raise RuntimeError(
            "You cannot use the ScalarFlow dataset without agreeing to the author's terms of use."
        )

    cmd = [
        "rsync",
        # Keep partial files in case download gets interrupted (not unlikely with almost
        # 500G)
        "--partial",
        # Show progress in terms of total download size
        "--info=progress2",
        # Show individual files as they are transferred
        "--info=name",
        # Compress files on the network
        "--compress",
        # Download recursively
        "--recursive",
        "rsync://m1521788@dataserv.ub.tum.de/m1521788/",
        str(root),
    ]
    env = {**os.environ, "RSYNC_PASSWORD": "m1521788"}
    log.info(" ".join(cmd))
    return subprocess.run(cmd, check=True, env=env)


def process_recording(args):
    archive, target_file, node_indices = args

    if target_file.is_file():
        mean_std = MeanStdAccumulator()
        mean_std.add(np.load(target_file)["u"])
        return mean_std

    volume_re = re.compile("^.*/velocity_[0-9]{6}.npz$")
    density_re = re.compile("^.*/density_[0-9]{6}.npz$")
    with tarfile.open(archive) as f:
        v_files = sorted([name for name in f.getnames() if volume_re.match(name)])
        d_files = sorted([name for name in f.getnames() if density_re.match(name)])

        # Velocity is derived from difference between densities, so it has a lag of one
        assert len(v_files) + 1 == len(d_files)

        # All recordings should have 150 steps
        n = len(v_files)
        assert n == 150

        frames = []
        for vf, df in zip(v_files, d_files[1:]):
            v = np.load(f.extractfile(vf))["data"]
            d = np.load(f.extractfile(df))["data"]
            frame_data = np.concatenate((v, d), axis=-1)

            # Take the mean along the z axis
            frame_data = frame_data.mean(axis=2)

            # Select data at mesh nodes
            frame_data = frame_data.reshape((-1, frame_data.shape[-1]))[node_indices]

            frames.append(frame_data)
    frames = np.stack(frames).astype(np.float32)

    np.savez_compressed(target_file, u=frames)

    mean_std = MeanStdAccumulator()
    mean_std.add(frames)
    return mean_std


@dataclass(frozen=True)
class ScalarFlowBatchKey:
    seq_ranges: list[tuple[int, tuple[Optional[int], Optional[int]]]]
    context_steps: int


class ScalarFlowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        domain: Domain,
        domain_info: DomainInfo,
        standardizer: Standardizer,
    ):
        super().__init__()

        self.root = root
        self.domain = domain
        self.domain_info = domain_info
        self.standardizer = standardizer
        self.cache = {}

        # All recordings are 2 seconds long and have 150 steps
        self.n_steps = 150
        self.t = torch.linspace(0.0, 2.0, self.n_steps, dtype=torch.float32)
        self.seq_idxs = [int(f.stem[4:]) for f in self.root.glob("sim_*.npz")]

    def __getitem__(self, key: ScalarFlowBatchKey) -> STBatch:
        t = []
        u = []
        for seq_idx, start_end in key.seq_ranges:
            seq_u = self.cache.get(seq_idx)
            if seq_u is None:
                np_u = np.load(self.root / f"sim_{seq_idx:06d}.npz")["u"]
                seq_u = torch.from_numpy(np_u).float()
                self.cache[seq_idx] = seq_u

            slice_ = slice(*start_end)
            t.append(self.t[slice_])
            u.append(seq_u[slice_])

        t = torch.stack(t)
        return STBatch(
            domain=self.domain,
            domain_info=self.domain_info,
            t=t,
            u=torch.stack(u),
            context_steps=key.context_steps,
            standardizer=self.standardizer,
            time_encoder=InvariantEncoder(t[:, 0]),
        )


class ScalarFlowSampler(Sampler):
    def __init__(
        self,
        dataset: ScalarFlowDataset,
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
            (seq_idx, (start, start + seq_len))
            for seq_idx in self.dataset.seq_idxs
            for start in range(self.dataset.n_steps - seq_len)
        ]

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            indices = indices.copy()
            random.shuffle(indices)
        for chunk in chunked(indices, self.batch_size):
            yield ScalarFlowBatchKey(seq_ranges=chunk, context_steps=self.context_steps)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)


class ScalarFlowDataModule(pl.LightningDataModule):
    """A downsampled 2D version of the ScalarFlow dataset.

    The ScalarFlow dataset is a collection of recordings of rising smoke columns published
    by Eckert et al. [1].

    [1] Marie-Lena Eckert, Kiwon Um, Nils Thuerey, "ScalarFlow: A Large-Scale Volumetric
        Data Set of Real-world Scalar Transport Flows for Computer Animation and Machine
        Learning"
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
        test_only: bool = False,
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
        self.test_only = test_only
        self.mesh_file = self.root / "domain.pt"
        self.stats_file = self.root / "stats.npz"

        self.domain = None
        self.domain_info = None
        self.standardizer = None

    def prepare_data(self):
        self.root.mkdir(exist_ok=True, parents=True)

        # Don't check for raw data if the processed data is available
        if (
            self.mesh_file.is_file()
            and ((not self.test_only) and self.stats_file.is_file())
            and self._data_files_exist()
        ):
            return

        self._download_raw_data()
        self._mesh_subset_of_grid()
        self._process_recordings()

    def setup(self, stage: Optional[str] = None):
        if self.domain is None:
            self.domain = torch.load(self.mesh_file)["domain"].normalize()
            self.domain_info = self.domain_preprocessing(self.domain)
        if self.standardizer is None:
            data = np.load(self.stats_file)
            mean, std = torch.from_numpy(data["mean"]), torch.from_numpy(data["std"])
            self.standardizer = Standardizer(mean, std)

        if stage in (None, "train", "fit"):
            self.train_data = ScalarFlowDataset(
                self.root / "train", self.domain, self.domain_info, self.standardizer
            )
        if stage in (None, "validate", "fit"):
            self.val_data = ScalarFlowDataset(
                self.root / "val", self.domain, self.domain_info, self.standardizer
            )
        if stage in (None, "test"):
            self.test_data = ScalarFlowDataset(
                self.root / "test", self.domain, self.domain_info, self.standardizer
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
        sampler = ScalarFlowSampler(
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
        return self.val_data[
            ScalarFlowBatchKey(
                seq_ranges=[(66, (65 - self.context_steps, 90))],
                context_steps=self.context_steps,
            )
        ]

    @staticmethod
    def default_mesh_config():
        # Arbitrarily chosen, fixed seed
        return MeshConfig(k=1000, epsilon=10.0, seed=366224077660046)

    def _download_raw_data(self):
        self.raw_root.mkdir(exist_ok=True, parents=True)
        if len(list(self.raw_root.glob("sim_*.tar"))) < 104:
            log.info("Download ScalarFlow data")
            download_raw_data(self.raw_root)

    def _mesh_subset_of_grid(self):
        if self.mesh_file.is_file():
            existing_config = torch.load(self.mesh_file)["config"]
            if existing_config != self.mesh_config:
                raise RuntimeError(
                    "Mesh already exists but has been created with a different "
                    f"configuration! \n\n {existing_config}\n{self.mesh_config}"
                )
            else:
                return

        log.info("Create mesh from random subset of nodes")

        # Create the grid of all cell centers in the 2D projection of the data
        height = np.linspace(0.5, 177.5, num=178) / 100
        width = np.linspace(0.5, 99.5, num=100) / 100
        x, y = np.meshgrid(width, height, indexing="ij")
        grid_points = np.vstack((x.ravel(), y.ravel())).T

        # Select cells in the center of the domain where stuff is actually happening
        x, y = grid_points.T
        interesting = (x >= 0.2) & (x <= 0.8) & (y >= 0.0) & (y <= 1.4)
        valid_points = grid_points[interesting]

        predicate = self.mesh_config.angle_predicate
        node_indices, domain = sample_mesh(self.mesh_config, valid_points, predicate)

        # Translate node indices from `valid_points` to `grid_points`
        node_indices = np.nonzero(interesting)[0][node_indices]

        data = {
            "config": self.mesh_config,
            "domain": domain,
            "all_points": grid_points,
            "in_domain": interesting,
            "node_indices": node_indices,
        }
        torch.save(data, self.mesh_file)

    def _data_files_exist(self):
        n_files = {"train": 64, "val": 20, "test": 20}
        stages = ["test"] if self.test_only else ["train", "val", "test"]
        return all(
            len(list((self.root / stage).glob("sim_*.npz"))) == n_files[stage]
            for stage in stages
        )

    def _process_recordings(self):
        if self.stats_file.is_file() and self._data_files_exist():
            return

        sim_archives = list(self.raw_root.glob("sim_*.tar"))
        idx = lambda archive: int(archive.stem[4:])
        train_archives = [arch for arch in sim_archives if idx(arch) <= 63]
        val_archives = [arch for arch in sim_archives if 63 < idx(arch) <= 83]
        test_archives = [arch for arch in sim_archives if 83 < idx(arch)]

        node_indices = torch.load(self.mesh_file)["node_indices"]
        with mp.Pool(processes=8) as pool:
            if not self.test_only:
                mean_std = MeanStdAccumulator.sum(
                    self._process_archives("train", train_archives, node_indices, pool)
                )
                self._process_archives("val", val_archives, node_indices, pool)

            self._process_archives("test", test_archives, node_indices, pool)

        if not self.test_only:
            mean, std = mean_std.mean_and_std()
            np.savez(self.stats_file, mean=mean, std=std)

    def _process_archives(
        self, stage: str, archives: list[Path], node_indices, pool: mp.Pool
    ):
        target_dir = self.root / stage
        target_dir.mkdir(exist_ok=True)
        tasks = [
            (archive, target_dir / f"{archive.stem}.npz", node_indices)
            for archive in archives
        ]
        return list(
            tqdm(pool.imap(process_recording, tasks), total=len(tasks), desc=stage)
        )
