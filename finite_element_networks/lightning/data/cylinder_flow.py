import enum
import json
import math
import multiprocessing as mp
import os
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import skfem
import torch
from more_itertools import chunked
from tfrecord_lite import tf_record_iterator
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from ...data import DomainInfo, InvariantEncoder, Standardizer, STBatch
from ...domain import Domain
from .utils import MeanStdAccumulator


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


@dataclass(frozen=True)
class CylinderFlowBatchKey:
    seq_idx: int
    ranges: list[tuple[Optional[int], Optional[int], Optional[int]]]
    context_steps: int


class CylinderFlowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        standardizer: Standardizer,
        domain_preprocessing: Callable[[Domain], DomainInfo],
    ):
        super().__init__()

        self.root = root
        self.domain_preprocessing = domain_preprocessing
        self.cache = {}
        self.standardizer = standardizer
        self.trajectory_length, self.dt = 600, 0.01
        self.t = torch.arange(self.trajectory_length) * self.dt
        self.num_sequences = len(
            [1 for f in self.root.iterdir() if re.match("[0-9]+.pt", f.name)]
        )

    def __getitem__(self, key: CylinderFlowBatchKey) -> STBatch:
        u, domain, info = self.cache.get(key.seq_idx, (None, None, None))
        if u is None or domain is None:
            seq_data = torch.load(self.root / f"{key.seq_idx}.pt")
            u = seq_data["u"]
            domain = seq_data["domain"]
            info = self.domain_preprocessing(domain)

            self.cache[key.seq_idx] = (u, domain, info)

        t = torch.stack([self.t[slice(*r)] for r in key.ranges])
        u = torch.stack([u[slice(*r)] for r in key.ranges])

        return STBatch(
            domain=domain,
            domain_info=info,
            t=t,
            u=u,
            context_steps=key.context_steps,
            standardizer=self.standardizer,
            time_encoder=InvariantEncoder(t[:, 0]),
        )


class CylinderFlowSampler(Sampler):
    def __init__(
        self,
        dataset: CylinderFlowDataset,
        target_steps: int,
        context_steps: int,
        batch_size: int,
        shuffle: bool,
        step_size: int,
        skip_steps: int,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(dataset)

        self.dataset = dataset
        self.target_steps = target_steps
        self.context_steps = context_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        seq_len = (target_steps + context_steps) * step_size
        if n_sequences is None:
            seq_idxs = range(self.dataset.num_sequences)
        else:
            seq_idxs = range(min(self.dataset.num_sequences, n_sequences))
        self.indices = {
            seq: [
                (start, start + seq_len, step_size)
                for start in range(
                    0, self.dataset.trajectory_length - seq_len, 1 + skip_steps
                )
            ]
            for seq in seq_idxs
        }

    def __iter__(self):
        indices = []
        for seq_id in self.indices.keys():
            ranges = self.indices[seq_id].copy()
            if self.shuffle:
                random.shuffle(ranges)
            for chunk in chunked(ranges, self.batch_size):
                indices.append(CylinderFlowBatchKey(seq_id, chunk, self.context_steps))
        if self.shuffle:
            random.shuffle(indices)
        yield from indices

    def __len__(self):
        return sum(
            math.ceil(len(ranges) / self.batch_size) for ranges in self.indices.values()
        )


def download_file(url: str, to: Path):
    import requests

    response = requests.get(url, stream=True)
    if "Content-Length" in response.headers:
        total = int(response.headers["Content-Length"])
    else:
        total = 0
    with to.open("wb") as f:
        pbar = tqdm(
            response.iter_content(chunk_size=10**5),
            desc=to.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        for data in pbar:
            f.write(data)
            pbar.update(len(data))


def process_recording(args):
    example, target_file, meta = args

    def read_feature(name, type):
        dtype = np.dtype(type).newbyteorder("<")
        shape = meta["features"][name]["shape"]
        data = np.frombuffer(example[name][0], dtype)
        return data.reshape(shape).squeeze()

    cells = read_feature("cells", np.int32)
    nodes = read_feature("mesh_pos", np.float32)
    node_type = read_feature("node_type", np.int32)
    velocity = read_feature("velocity", np.float32)
    pressure = read_feature("pressure", np.float32)

    mesh = skfem.MeshTri(np.ascontiguousarray(nodes.T), np.ascontiguousarray(cells.T))
    # Only the velocity is fixed on boundaries and inflows
    velocity_mask = (node_type == NodeType.INFLOW) | (
        node_type == NodeType.WALL_BOUNDARY
    )
    no_mask = np.zeros_like(velocity_mask)
    mask = np.stack([velocity_mask, velocity_mask, no_mask], axis=-1)
    domain = Domain(nodes, mesh=mesh, fixed_values_mask=mask)

    u = np.concatenate([velocity, pressure[..., None]], axis=-1)
    contents = {"domain": domain, "u": torch.from_numpy(u)}
    torch.save(contents, target_file)

    stats = MeanStdAccumulator()
    stats.add(u)
    return stats


class CylinderFlowDataModule(pl.LightningDataModule):
    """The cylinder flow dataset as seen in the MeshGraphNet paper [1].

    [1] Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia,
        "Learning Mesh-Based Simulation with Graph Networks",
        https://arxiv.org/abs/2010.03409
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
        step_size: int = 1,
        skip_steps: int = 0,
        train_target_steps: int = 10,
        eval_target_steps: int = 10,
        n_train_sequences: int = 100,
        raw_root: Optional[Path] = None,
    ):
        super().__init__()

        self.root = root
        self.raw_root = root / "raw" if raw_root is None else raw_root
        self.domain_preprocessing = domain_preprocessing
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.context_steps = context_steps
        self.step_size = step_size
        self.skip_steps = skip_steps
        self.train_target_steps = train_target_steps
        self.eval_target_steps = eval_target_steps
        self.n_train_sequences = n_train_sequences

        self.stats_file = self.root / "stats.npz"
        self.standardizer = None

    def prepare_data(self):
        self.raw_root.mkdir(exist_ok=True, parents=True)

        # Don't check for raw data if the processed data is available
        n_files = {"train": 1000, "valid": 100, "test": 100}
        if self.stats_file.is_file() and all(
            len(list((self.root / stage).glob("*.pt"))) == n_files[stage]
            for stage in ["train", "valid", "test"]
        ):
            return

        # Hashing these files takes way too long, so this is just a sanity check that
        # downloads were not interrupted.
        FILES = [
            ("meta.json", 883),
            ("train.tfrecord", 13645805387),
            ("valid.tfrecord", 1363987289),
            ("test.tfrecord", 1355376404),
        ]
        for file, size in FILES:
            path = self.raw_root / file
            if path.is_file():
                actual_size = os.path.getsize(path)
                if actual_size == size:
                    continue
                else:
                    raise RuntimeError(
                        f"{path} has unexpected file size of {actual_size}, expected {size}"
                    )

            download_file(
                f"https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/{file}",
                path,
            )

        meta = json.loads((self.raw_root / "meta.json").read_text())
        with mp.Pool() as pool:
            for stage in ["train", "valid", "test"]:
                stage_dir = self.root / stage
                stage_dir.mkdir(exist_ok=True)

                def gen_job(args):
                    i, example = args
                    return (example, self.root / stage / f"{i}.pt", meta)

                jobs = map(
                    gen_job,
                    enumerate(tf_record_iterator(self.raw_root / f"{stage}.tfrecord")),
                )
                totals = {"train": 1000, "valid": 100, "test": 100}
                stats = list(
                    tqdm(
                        pool.imap(process_recording, jobs),
                        total=totals[stage],
                        desc=f"Decode {stage}.tfrecord",
                    )
                )

                if stage == "train":
                    mean, std = MeanStdAccumulator.sum(stats).mean_and_std()
                    np.savez(self.stats_file, mean=mean, std=std)

    def setup(self, stage: Optional[str] = None):
        if self.standardizer is None:
            data = np.load(self.stats_file)
            self.standardizer = Standardizer(
                torch.from_numpy(data["mean"]), torch.from_numpy(data["std"])
            )

        if stage in (None, "train", "fit"):
            self.train_data = CylinderFlowDataset(
                self.root / "train", self.standardizer, self.domain_preprocessing
            )
        if stage in (None, "validate", "fit"):
            self.val_data = CylinderFlowDataset(
                self.root / "valid", self.standardizer, self.domain_preprocessing
            )
        if stage in (None, "test"):
            self.test_data = CylinderFlowDataset(
                self.root / "test", self.standardizer, self.domain_preprocessing
            )

    def train_dataloader(self):
        return self._dataloader(
            self.train_data,
            target_steps=self.train_target_steps,
            shuffle=True,
            n_sequences=self.n_train_sequences,
        )

    def val_dataloader(self):
        return self._dataloader(
            self.val_data, target_steps=self.eval_target_steps, shuffle=False
        )

    def test_dataloader(self):
        return self._dataloader(
            self.test_data, target_steps=self.eval_target_steps, shuffle=False
        )

    def _dataloader(self, dataset, *, target_steps, shuffle, n_sequences=None):
        sampler = CylinderFlowSampler(
            dataset,
            target_steps=target_steps,
            context_steps=self.context_steps,
            batch_size=self.batch_size,
            shuffle=shuffle,
            step_size=self.step_size,
            skip_steps=self.skip_steps,
            n_sequences=n_sequences,
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
            CylinderFlowBatchKey(
                seq_idx=50,
                ranges=[
                    (
                        300 - self.step_size * self.context_steps,
                        300 + self.step_size * self.eval_target_steps,
                        self.step_size,
                    )
                ],
                context_steps=self.context_steps,
            )
        ]
