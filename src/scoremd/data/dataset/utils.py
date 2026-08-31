import os
from os import PathLike
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
import openmm.app as app


Array = np.ndarray


def write_animation_with_topology(trajectory: jnp.ndarray, topology: app.Topology, out: PathLike):
    """Write a trajectory to a PDB file. The trajectory is in nanometers."""
    with open(os.path.expanduser(out), "w") as pdbfile:
        app.PDBFile.writeHeader(topology, pdbfile)
        for i, xyz in enumerate(trajectory):
            positions = xyz.reshape(-1, 3) * 10  # in Angstrom
            app.PDBFile.writeModel(topology, positions, pdbfile, modelIndex=i + 1)
        app.PDBFile.writeFooter(topology, pdbfile)


def _validate_full_atom_frames(data_full: Array) -> Array:
    frames = np.asarray(data_full, dtype=float)
    if frames.ndim != 3 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError("data_full must be a non-empty array with shape (n_frames, n_atoms, 3).")
    if not np.all(np.isfinite(frames)):
        raise ValueError("data_full must contain only finite coordinates.")
    return frames


def _rigid_body_basis(frame_full: Array) -> Array:
    """Return orthonormal translation/rotation modes for one full-atom frame."""
    positions = np.asarray(frame_full, dtype=float)
    centered = positions - positions.mean(axis=0, keepdims=True)
    n_atoms = positions.shape[0]

    translations = np.zeros((n_atoms, 3, 3), dtype=float)
    translations[:, np.arange(3), np.arange(3)] = 1.0
    rotations = np.stack([np.cross(axis, centered) for axis in np.eye(3)], axis=-1)
    modes = np.concatenate(
        [translations.reshape(n_atoms * 3, 3), rotations.reshape(n_atoms * 3, 3)],
        axis=1,
    )
    left_singular_vectors, singular_values, _ = np.linalg.svd(modes, full_matrices=False)
    tolerance = np.finfo(float).eps * max(modes.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    return left_singular_vectors[:, :rank]


def _project_out_rigid_body_modes(direction: Array, frame_full: Array) -> Array:
    """Project a Cartesian direction away from global translations and rotations."""
    vector = np.asarray(direction, dtype=float).reshape(-1)
    basis = _rigid_body_basis(frame_full)
    projected = vector - basis @ (basis.T @ vector)
    norm = np.linalg.norm(projected)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Direction lies entirely in the rigid-body subspace.")
    return (projected / norm).reshape(np.asarray(frame_full).shape)


def _sample_full_atom_directions(frame_full: Array, n_directions: int, rng: np.random.Generator) -> Array:
    """Sample Cartesian basis directions for a full-atom frame."""
    n_coordinates = int(np.asarray(frame_full).size)
    if n_directions <= 0:
        raise ValueError("directions_per_frame must be positive.")
    indices = rng.choice(n_coordinates, size=min(n_directions, n_coordinates), replace=False)
    directions = np.zeros((len(indices), n_coordinates), dtype=float)
    directions[np.arange(len(indices)), indices] = 1.0
    return directions.reshape((len(indices), *np.asarray(frame_full).shape))


def directional_hessian_curvature(
    force_fn: Callable[[Array], Array],
    frame_full: Array,
    direction: Array,
    *,
    eps: float = 5e-4,
) -> float:
    """Estimate ``v.T @ Hessian(U) @ v`` from central force differences.

    Frames and ``eps`` are in nm; ``force_fn`` must return kJ/(mol nm), so the
    curvature is in kJ/(mol nm²).  ``direction`` is normalized internally.
    """
    frame = np.asarray(frame_full, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if frame.ndim != 2 or frame.shape[-1] != 3:
        raise ValueError(f"frame_full must have shape (n_atoms, 3); got {frame.shape}.")
    if frame.shape != direction.shape:
        raise ValueError(
            "direction must have the same shape as a full frame; "
            f"got frame {frame.shape} and direction {direction.shape}."
        )
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite displacement in nm.")

    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be finite and nonzero.")
    direction = direction / norm

    # F = -grad(U), hence -dF/ds is the directional Hessian curvature.
    force_plus = np.asarray(force_fn(frame + eps * direction), dtype=float)
    force_minus = np.asarray(force_fn(frame - eps * direction), dtype=float)
    if force_plus.shape != frame.shape or force_minus.shape != frame.shape:
        raise ValueError("force_fn must return an array with the same shape as its frame input.")
    force_derivative = (force_plus - force_minus) / (2.0 * eps)
    return float(-np.sum(force_derivative * direction))


def compute_sigma_mode(
    data_full: Array,
    force_fn: Callable[[Array], Array],
    *,
    beta: float,
    n_subsample: Optional[int] = 500,
    directions_per_frame: int = 6,
    eps: float = 5e-4,
    seed: Optional[int] = 0,
    min_curvature: float = 1e-3,
    project_rigid_body_modes: bool = True,
    direction_fn: Optional[Callable[[Array], Array]] = None,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, float | int]]:
    """Estimate a full-atom TSM mode variance from local force curvatures.

    The estimator returns ``mean[1 / (beta * v.T @ Hessian(U) @ v)]`` over a
    reproducible frame/direction subsample.  By default it samples Cartesian
    atom directions and projects out rigid translations and rotations.  The
    optional ``direction_fn`` is reserved for future CG use: it must return
    directions with shape ``(n_directions, n_atoms, 3)`` for one full frame.

    Coordinates are in nm and the returned physical variance is in nm².  When
    training uses ``x_normalized = norm_factor * x``, pass
    ``norm_factor**2 * returned_variance`` to the normalized TSM loss.
    """
    frames = _validate_full_atom_frames(data_full)
    if not callable(force_fn):
        raise TypeError("force_fn must accept a full-atom frame and return its force array.")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite.")
    if n_subsample is not None and n_subsample <= 0:
        raise ValueError("n_subsample must be positive or None.")
    if directions_per_frame <= 0:
        raise ValueError("directions_per_frame must be positive.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite displacement in nm.")
    if not np.isfinite(min_curvature) or min_curvature <= 0.0:
        raise ValueError("min_curvature must be positive and finite.")

    rng = np.random.default_rng(seed)
    n_draw = frames.shape[0] if n_subsample is None else min(int(n_subsample), frames.shape[0])
    frame_indices = rng.choice(frames.shape[0], size=n_draw, replace=False)

    curvatures: list[float] = []
    local_variances: list[float] = []
    discarded = 0
    evaluated = 0
    for frame_index in frame_indices:
        frame = frames[frame_index]
        if direction_fn is None:
            directions = _sample_full_atom_directions(frame, directions_per_frame, rng)
        else:
            directions = np.asarray(direction_fn(frame), dtype=float)
            if directions.shape == frame.shape:
                directions = directions[None, ...]
            if directions.ndim != 3 or directions.shape[1:] != frame.shape:
                raise ValueError(
                    "direction_fn must return shape (n_directions, n_atoms, 3) "
                    f"or (n_atoms, 3); got {directions.shape}."
                )
            if directions.shape[0] > directions_per_frame:
                selected = rng.choice(directions.shape[0], size=directions_per_frame, replace=False)
                directions = directions[selected]

        for direction in directions:
            try:
                if project_rigid_body_modes:
                    direction = _project_out_rigid_body_modes(direction, frame)
                else:
                    norm = np.linalg.norm(direction)
                    if not np.isfinite(norm) or norm == 0.0:
                        raise ValueError("Direction must be finite and nonzero.")
                    direction = direction / norm
            except ValueError:
                discarded += 1
                continue

            curvature = directional_hessian_curvature(force_fn, frame, direction, eps=eps)
            evaluated += 1
            if not np.isfinite(curvature) or curvature < min_curvature:
                discarded += 1
                continue
            curvatures.append(curvature)
            local_variances.append(1.0 / (beta * curvature))

    if not local_variances:
        raise ValueError(
            "No sampled direction had curvature >= min_curvature; the harmonic "
            "mode-variance estimate is undefined for the chosen settings."
        )

    local = np.asarray(local_variances, dtype=float)
    estimate = float(local.mean())
    if not return_diagnostics:
        return estimate

    curvature_values = np.asarray(curvatures, dtype=float)
    diagnostics: dict[str, float | int] = {
        "n_sampled_frames": int(n_draw),
        "directions_per_frame": int(directions_per_frame),
        "n_direction_evaluations": int(evaluated),
        "n_used": int(len(local)),
        "n_discarded": int(discarded),
        "mean_curvature": float(curvature_values.mean()),
        "local_sigma2_std": float(local.std(ddof=0)),
        "standard_error": float(local.std(ddof=1) / np.sqrt(len(local))) if len(local) > 1 else 0.0,
    }
    return estimate, diagnostics


def compute_full_atom_sigma_mode(dataset, **kwargs) -> float | tuple[float, dict[str, float | int]]:
    """Estimate physical ``sigma_mode_sq`` from a full-atom ScoreMD dataset.

    The dataset must expose full-atom ``train.data``, ``sample_shape``,
    ``kbT``, and ``force(frame)``.  The returned value is in nm² and must be
    multiplied by ``norm_factor**2`` before use by a normalized score model.
    """
    sample_shape = tuple(dataset.sample_shape)
    if len(sample_shape) != 2 or sample_shape[-1] != 3:
        raise ValueError("Full-atom sigma-mode estimation requires dataset.sample_shape = (n_atoms, 3).")
    if not hasattr(dataset, "force") or not callable(dataset.force):
        raise TypeError("Dataset must provide a callable force(frame) for Hessian estimation.")
    frames = np.asarray(dataset.train.data, dtype=float).reshape((-1, *sample_shape))
    return compute_sigma_mode(frames, dataset.force, beta=1.0 / float(dataset.kbT), **kwargs)
