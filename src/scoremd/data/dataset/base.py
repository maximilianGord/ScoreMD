import abc
from dataclasses import dataclass
from typing import Optional, Tuple
import jax.numpy as jnp
import logging
from flax import struct

log = logging.getLogger(__name__)


class Datapoints(struct.PyTreeNode):
    data: jnp.ndarray
    features: Optional[jnp.ndarray]
    forces: Optional[jnp.ndarray] = struct.field(default=None, kw_only=True)
    # Positions paired with ``forces`` for the force-based terms (TSM, SC anchor). ``None`` means the
    # forces belong to ``data``. Enhanced sampling uses this so DSM trains on unbiased ``data`` while
    # the force targets are evaluated on biased samples.
    force_data: Optional[jnp.ndarray] = struct.field(default=None, kw_only=True)
    sigma_mode_sq: Optional[jnp.ndarray] = struct.field(default=None, kw_only=True)
    # Precomputed TSM targets (precompute_tsm_mean_force.py): per sample M draws of the noise time ``tsm_t``
    # (N, M), the normalized noise ``tsm_noise`` (N, M, D) and the force target ``tsm_forces`` (N, M, D) for
    # that noisy sample (e.g. the posterior mean force E[F(x0) | x_t]). Training draws one of the M per step.
    tsm_t: Optional[jnp.ndarray] = struct.field(default=None, kw_only=True)
    tsm_noise: Optional[jnp.ndarray] = struct.field(default=None, kw_only=True)
    tsm_forces: Optional[jnp.ndarray] = struct.field(default=None, kw_only=True)

    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {self.data.ndim}D.")
        if self.features is not None:
            if self.features.ndim != 3:
                raise ValueError(f"Features must be 3D, got {self.features.ndim}D. (BS, num_atoms, num_features)")
            if self.data.shape[0] != self.features.shape[0]:
                raise ValueError(
                    f"First dimension of data ({self.data.shape[0]}) and features ({self.features.shape[0]}) must match."
                )
        if self.forces is not None:
            if self.forces.shape != self.data.shape:
                raise ValueError(
                    f"forces must have the same shape as data; got {self.forces.shape} and {self.data.shape}."
                )
        if self.force_data is not None:
            if self.force_data.shape != self.data.shape:
                raise ValueError(
                    f"force_data must have the same shape as data; got {self.force_data.shape} and {self.data.shape}."
                )
        if self.sigma_mode_sq is not None:
            if self.sigma_mode_sq.shape != (self.data.shape[0],):
                raise ValueError(
                    "sigma_mode_sq must have shape (num_samples,); got "
                    f"{self.sigma_mode_sq.shape} for data shape {self.data.shape}."
                )
        bank = (self.tsm_t, self.tsm_noise, self.tsm_forces)
        if any(b is not None for b in bank):
            if any(b is None for b in bank):
                raise ValueError("tsm_t, tsm_noise and tsm_forces must be given together.")
            n, d = self.data.shape
            m = self.tsm_t.shape[1] if self.tsm_t.ndim == 2 else -1
            if self.tsm_t.shape != (n, m) or self.tsm_noise.shape != (n, m, d) or self.tsm_forces.shape != (n, m, d):
                raise ValueError(
                    f"tsm_t, tsm_noise, tsm_forces must have shapes (N, M), (N, M, D), (N, M, D) for data {self.data.shape}; "
                    f"got {self.tsm_t.shape}, {self.tsm_noise.shape}, {self.tsm_forces.shape}."
                )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return Datapoints(
            self.data[idx],
            self.features[idx] if self.features is not None else None,
            forces=self.forces[idx] if self.forces is not None else None,
            force_data=self.force_data[idx] if self.force_data is not None else None,
            sigma_mode_sq=self.sigma_mode_sq[idx] if self.sigma_mode_sq is not None else None,
            tsm_t=self.tsm_t[idx] if self.tsm_t is not None else None,
            tsm_noise=self.tsm_noise[idx] if self.tsm_noise is not None else None,
            tsm_forces=self.tsm_forces[idx] if self.tsm_forces is not None else None,
        )


@dataclass
class Dataset:
    name: str
    sample_shape: Tuple[int, ...]
    kbT: float

    def __init__(self, name: str, sample_shape: Tuple[int, ...], kbT: float):
        self.name = name
        self.sample_shape = sample_shape
        self.kbT = kbT
        self._train: Optional[Datapoints] = None
        self._val: Optional[Datapoints] = None
        self._test: Optional[Datapoints] = None

    @abc.abstractmethod
    def _get_data(self) -> Tuple[Datapoints, Optional[Datapoints], Optional[Datapoints]]:
        raise NotImplementedError()

    def load_data(self):
        if self._train is None:
            self._train, self._val, self._test = self._get_data()

    def _std(self, data: jnp.ndarray) -> jnp.ndarray:
        return jnp.std(
            data
        )  # we don't want to compute std for each coordinate, because we assume that the data can be arbitrarily rotated

    @property
    def train(self) -> Datapoints:
        self.load_data()
        return self._train

    @property
    def val(self) -> Optional[Datapoints]:
        self.load_data()
        return self._val

    @property
    def test(self) -> Optional[Datapoints]:
        self.load_data()
        return self._test

    @property
    def std(self) -> jnp.ndarray:
        return self._std(self.train.data.reshape(self.train.data.shape[0], -1, 3))
