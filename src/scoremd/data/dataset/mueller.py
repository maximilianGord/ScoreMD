import os.path
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
from deeptime.util import energy2d
import jax.numpy as jnp
import jax
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from enum import Enum
from scoremd.data.dataset.base import Datapoints
from scoremd.utils.file import get_persistent_storage
import hashlib

from scoremd.utils.plots import rasterize_contour
from . import Dataset
from ...simulation import create_langevin_step_function, simulate

log = logging.getLogger(__name__)


class MuellerBrownCoarseGrainingLevel(Enum):
    """Coordinates retained from a two-dimensional Müller--Brown frame."""

    NONE = "NONE"
    X_MARGINAL = "X_MARGINAL"
    Y_MARGINAL = "Y_MARGINAL"


def mueller_brown_potential(xs: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """
    Compute the energy of the mueller brown potential.

    Coordinates are normally supplied as ``(..., 2)``.  TSM preprocessing
    represents a single toy-system frame as ``(2, 1)`` to match the dataset's
    ``sample_shape``; flatten that representation back to coordinate pairs
    before extracting ``x`` and ``y``.
    """
    xs = jnp.asarray(xs)
    if xs.ndim == 1:
        if xs.shape[0] != 2:
            raise ValueError(f"Expected two Müller-Brown coordinates, got shape {xs.shape}.")
        xs = xs.reshape(1, 2)
    elif xs.shape[-1] == 2:
        xs = xs.reshape(-1, 2)
    elif xs.shape[-2:] == (2, 1):
        xs = xs.reshape(-1, 2)
    else:
        raise ValueError(f"Expected coordinates shaped (..., 2) or (..., 2, 1), got {xs.shape}.")

    x, y = xs[:, 0], xs[:, 1]
    e1 = -200 * jnp.exp(-((x - 1) ** 2) - 10 * y**2)
    e2 = -100 * jnp.exp(-(x**2) - 10 * (y - 0.5) ** 2)
    e3 = -170 * jnp.exp(-6.5 * (0.5 + x) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
    e4 = 15.0 * jnp.exp(0.7 * (1 + x) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
    return beta * (e1 + e2 + e3 + e4)


@dataclass
class MuellerBrownSimulation(Dataset):
    n_samples: int = 10_000
    n_steps: int = 50
    mass: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 1.0]))
    gamma: float = 1.0
    dt: float = 1e-4
    beta: float = 1.0
    seed: int = 0
    mode_var_computation: Literal["potential", "data_hessian", "data_empirical"] = "data_hessian"
    coarse_graining_level: MuellerBrownCoarseGrainingLevel = MuellerBrownCoarseGrainingLevel.NONE

    def __init__(
        self,
        n_samples: int = 100_000,
        n_steps: int = 50,
        kbT: float = 23.0,
        mass: jnp.ndarray = jnp.array([1.0, 1.0]),
        gamma: float = 1.0,
        dt: float = 1e-4,
        beta: float = 1.0,
        seed: int = 0,
        mode_var_computation: Literal["potential", "data_hessian", "data_empirical"] = "data_hessian",
        coarse_graining_level: MuellerBrownCoarseGrainingLevel | str = MuellerBrownCoarseGrainingLevel.NONE,
        name="mueller_brown",
    ):
        if mode_var_computation not in {"potential", "data_hessian", "data_empirical"}:
            raise ValueError(
                "mode_var_computation must be 'potential', 'data_hessian', or 'data_empirical'; "
                f"got {mode_var_computation!r}."
            )
        self.coarse_graining_level = (
            coarse_graining_level
            if isinstance(coarse_graining_level, MuellerBrownCoarseGrainingLevel)
            else MuellerBrownCoarseGrainingLevel(coarse_graining_level.upper())
        )
        self._coordinate_index: Optional[int] = {
            MuellerBrownCoarseGrainingLevel.X_MARGINAL: 0,
            MuellerBrownCoarseGrainingLevel.Y_MARGINAL: 1,
        }.get(self.coarse_graining_level)
        self._train_force_coordinates = None
        super().__init__(
            name=name,
            sample_shape=(2, 1) if self._coordinate_index is None else (1, 1),
            kbT=kbT,
        )
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.full_mass = jnp.array(mass)
        self.mass = self.full_mass if self._coordinate_index is None else self.full_mass[self._coordinate_index : self._coordinate_index + 1]
        self.gamma = gamma
        self.dt = dt
        self.beta = beta
        self.seed = seed
        self.mode_var_computation = mode_var_computation

    @property
    def is_coarse_grained(self) -> bool:
        return self._coordinate_index is not None

    @property
    def coordinate_name(self) -> str:
        if self._coordinate_index is None:
            raise ValueError("The full Müller--Brown system has no single retained coordinate.")
        return ("x", "y")[self._coordinate_index]

    def force_coordinates_for(self, datapoints: Datapoints) -> Optional[jnp.ndarray]:
        """Return paired full frames for one-time coarse-grained force projection."""
        if datapoints is self._train:
            return self._train_force_coordinates
        if self.is_coarse_grained:
            raise ValueError("Projected Müller--Brown forces require paired full coordinates.")
        return None

    def release_force_coordinates(self, datapoints: Datapoints) -> None:
        if datapoints is self._train:
            self._train_force_coordinates = None

    def project_forces(self, full_forces: jnp.ndarray) -> jnp.ndarray:
        """Return the instantaneous force along the retained CG coordinate."""
        if self._coordinate_index is None:
            return full_forces
        full_forces = jnp.asarray(full_forces).reshape(-1)
        if full_forces.shape != (2,):
            raise ValueError(f"Expected a two-component Müller--Brown force, got {full_forces.shape}.")
        return full_forces[self._coordinate_index : self._coordinate_index + 1]

    def sigma_mode_sq_from_potential(self) -> float:
        """Return the harmonic mode variance at the global Müller-Brown minimum."""
        mode = jnp.array([-0.55828035, 1.44169])
        hessian = jax.hessian(lambda x: self.potential(x[None, :]).sum())(mode)
        curvatures = jnp.linalg.eigvalsh(hessian)
        if bool(jnp.any(curvatures <= 0.0)):
            raise ValueError(f"Müller-Brown mode has non-positive curvatures: {curvatures}.")
        return float(jnp.mean(self.kbT / curvatures))

    def range(self) -> jnp.ndarray:
        return jnp.array([[-1.8, 1.1], [-0.5, 2.0]])

    def potential(self, xs: jnp.ndarray) -> jnp.ndarray:
        return mueller_brown_potential(xs, self.beta)

    def likelihood(self, xs: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.potential(xs) / self.kbT)

    def force(self, xs: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda _x: -self.potential(_x).sum())(xs)

    def _get_data(self) -> Tuple[Datapoints, None, None]:
        key = jax.random.PRNGKey(self.seed)

        dir = os.path.join(get_persistent_storage(), "MuellerBrown")
        os.makedirs(dir, exist_ok=True)
        sha256 = hashlib.sha256()
        sha256.update(repr(self).encode("utf-8"))

        file_name = f"{sha256.hexdigest()}.npy"
        data = None
        if os.path.exists(os.path.join(dir, file_name)):
            # try loading file
            try:
                data = jnp.load(os.path.join(dir, file_name))
                log.info(f"Loaded data from {os.path.join(dir, file_name)}")
            except Exception as e:
                log.warning(f"Failed to load data: {e}")

        if data is None:
            log.info(f"Generating data for {self.name} dataset.")
            data = self._generate_data(key)
            jnp.save(os.path.join(dir, file_name), data)

        if self._coordinate_index is not None:
            self._train_force_coordinates = data
            data = data[:, self._coordinate_index : self._coordinate_index + 1]
        return Datapoints(data, None), None, None

    def _generate_data(self, key):
        key, velocity_key = jax.random.split(key)

        starting_point = jnp.array([-0.55828035, 1.44169])
        # Sample the starting velocity from the Boltzmann distribution
        starting_velocity = jnp.sqrt(self.kbT / self.full_mass) * jax.random.normal(velocity_key, (2,))

        step = jax.jit(
            create_langevin_step_function(self.force, self.full_mass, self.gamma, self.n_steps, self.dt, self.kbT)
        )
        trajectory, _ = simulate(starting_point, starting_velocity, step, self.n_samples, key)
        return trajectory

    def plot(self, samples: jnp.ndarray, cbar_range: Tuple[float, float] = None, cbar: bool = True):
        assert samples.ndim == 2 and samples.shape[1] == 2, "Data should be a 2D vector."

        bins = 150
        (x_min, x_max), (y_min, y_max) = self.range()
        x, y = jnp.linspace(x_min, x_max, bins), jnp.linspace(y_min, y_max, bins)
        x, y = jnp.meshgrid(x, y, indexing="ij")
        z = self.potential(jnp.stack([x, y], -1).reshape(-1, 2)).reshape([bins, bins])

        plt.contour(x, y, z, levels=[-120, -90, -50, -20, 0, 20, 35, 70, 150, 250, 500, 1000], colors="black")

        energies = energy2d(*samples.T, bins=(100, 100), kbt=self.kbT, shift_energy=True)
        contourf_kws = {"cmap": "turbo"}
        if cbar_range is not None:
            contourf_kws["vmin"] = cbar_range[0]
            contourf_kws["vmax"] = cbar_range[1]
        ax, contour, cbar_obj = energies.plot(contourf_kws=contourf_kws, cbar=cbar_range is None and cbar)

        rasterize_contour(contour)

        if cbar:
            if cbar_range is not None:
                cbar_obj = ax.figure.colorbar(
                    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(cbar_range[0], cbar_range[1])), ax=ax
                )

            cbar_obj.set_label(r"Energy / $k_BT$")

        ax.set_xticks([])
        ax.set_yticks([])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        return energies
