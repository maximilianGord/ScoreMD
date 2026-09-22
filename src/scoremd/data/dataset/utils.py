import os
from os import PathLike
from typing import Any, Callable, Optional

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
    if frames.ndim < 2 or frames.shape[0] == 0:
        raise ValueError("data_full must be a non-empty array with a leading frame dimension.")
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
    if frame.ndim < 1:
        raise ValueError(f"frame_full must contain at least one coordinate; got {frame.shape}.")
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
    """Estimate a TSM mode variance from local force curvatures.

    Cartesian atom coordinates have rigid-body modes removed; other coordinate
    shapes, such as Mueller-Brown's (2, 1), use coordinate directions directly.

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
            if directions.ndim != frame.ndim + 1 or directions.shape[1:] != frame.shape:
                raise ValueError(
                    "direction_fn must return shape (n_directions, *frame.shape) "
                    f"or frame.shape; got {directions.shape}."
                )
            if directions.shape[0] > directions_per_frame:
                selected = rng.choice(directions.shape[0], size=directions_per_frame, replace=False)
                directions = directions[selected]

        for direction in directions:
            try:
                if project_rigid_body_modes and frame.ndim == 2 and frame.shape[-1] == 3:
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
    """Estimate physical ``sigma_mode_sq`` from full coordinates when a dataset provides them.

    Coarse-grained ALDP datasets retain paired full-atom frames exclusively for
    force-related preprocessing.  Use those frames here so OpenMM still sees
    the 22-atom configuration while the resulting scalar can be used by the
    CG loss.  Other datasets fall back to their training coordinates.
    """
    sample_shape = tuple(dataset.sample_shape)
    if not hasattr(dataset, "force") or not callable(dataset.force):
        raise TypeError("Dataset must provide a callable force(frame) for Hessian estimation.")

    datapoints = dataset.train
    full_coordinates = (
        dataset.force_coordinates_for(datapoints) if hasattr(dataset, "force_coordinates_for") else None
    )
    coordinate_source = full_coordinates if full_coordinates is not None else datapoints.data
    coordinates = np.asarray(coordinate_source, dtype=float)
    if full_coordinates is None:
        frame_shape = sample_shape
    elif coordinates.ndim > 2:
        # Preserve an explicitly provided frame layout.
        frame_shape = coordinates.shape[1:]
    elif sample_shape and sample_shape[-1] == 3 and coordinates.shape[1] % 3 == 0:
        # Molecular datasets retain flattened Cartesian full-atom frames.
        frame_shape = (-1, 3)
    else:
        # Toy systems such as coarse-grained Mueller--Brown retain a flat
        # coordinate vector (two coordinates), not Cartesian atom triples.
        frame_shape = coordinates.shape[1:]
    frames = coordinates.reshape((len(datapoints), *frame_shape))
    return compute_sigma_mode(frames, dataset.force, beta=1.0 / float(dataset.kbT), **kwargs)


def compute_empirical_sigma_mode(dataset) -> float:
    """Return the mean coordinate variance of centered training datapoints."""
    data = np.asarray(dataset.train.data, dtype=float).reshape((len(dataset.train), -1))
    centered = data - data.mean(axis=0, keepdims=True)
    return float(np.mean(np.square(centered)))


def compute_cg_local_sigma_mode(
    data: Array,
    *,
    n_subsample: int = 256,
    n_neighbors: int = 64,
    candidate_subsample: int = 8192,
    seed: Optional[int] = 0,
) -> tuple[float, dict[str, float | int]]:
    """Estimate a scalar CG mode variance from local coordinate neighborhoods.

    ``data`` must be the physical, already aligned coarse-grained coordinates.
    The median local coordinate variance avoids pooling separate metastable
    basins, unlike a global empirical variance. The result is in the squared
    coordinate unit of ``data`` (nm² for ALDP) and must be normalized by the
    caller when training coordinates are normalized.
    """
    frames = np.asarray(data, dtype=float)
    if frames.ndim < 2 or frames.shape[0] < 2:
        raise ValueError("data must contain at least two coarse-grained frames.")
    if not np.all(np.isfinite(frames)):
        raise ValueError("data must contain only finite coarse-grained coordinates.")
    if n_subsample <= 0 or n_neighbors < 2 or candidate_subsample < 2:
        raise ValueError("n_subsample, n_neighbors, and candidate_subsample must be positive; n_neighbors >= 2.")

    flattened = frames.reshape((frames.shape[0], -1))
    rng = np.random.default_rng(seed)
    n_candidates = min(int(candidate_subsample), len(flattened))
    n_references = min(int(n_subsample), len(flattened))
    n_neighbors = min(int(n_neighbors), n_candidates)
    candidates = flattened[rng.choice(len(flattened), size=n_candidates, replace=False)]
    references = flattened[rng.choice(len(flattened), size=n_references, replace=False)]

    local_variances: list[float] = []
    for reference in references:
        squared_distances = np.sum(np.square(candidates - reference), axis=1)
        neighbor_indices = np.argpartition(squared_distances, n_neighbors - 1)[:n_neighbors]
        neighborhood = candidates[neighbor_indices]
        local_variances.append(float(np.mean(np.var(neighborhood, axis=0, ddof=1))))

    local = np.asarray(local_variances, dtype=float)
    estimate = float(np.median(local))
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("CG local covariance produced a non-positive or non-finite mode variance.")
    diagnostics: dict[str, float | int] = {
        "n_references": n_references,
        "n_candidates": n_candidates,
        "n_neighbors": n_neighbors,
        "local_sigma2_median": estimate,
        "local_sigma2_mean": float(local.mean()),
        "local_sigma2_std": float(local.std(ddof=0)),
    }
    return estimate, diagnostics


def _component_scale(covariance: Array, covariance_type: str) -> float:
    """Reduce one Gaussian-mixture component's covariance to an isotropic scalar.

    Uses the geometric mean of the covariance's eigenvalues -- the variance of
    the isotropic Gaussian with the same differential entropy as the fitted
    (possibly anisotropic) component. This avoids collapsing an anisotropic
    mode (e.g. stiff bonds mixed with a soft dihedral) into an arithmetic
    mean, which would be dominated by whichever eigenvalue is largest.
    """
    if covariance_type == "full":
        eigenvalues = np.linalg.eigvalsh(covariance)
    elif covariance_type == "diag":
        eigenvalues = np.asarray(covariance)
    elif covariance_type == "spherical":
        eigenvalues = np.full(1, covariance)
    else:
        raise ValueError(f"Unsupported covariance_type for scale reduction: {covariance_type!r}.")
    eigenvalues = np.clip(eigenvalues, a_min=np.finfo(float).tiny, a_max=None)
    return float(np.exp(np.mean(np.log(eigenvalues))))


def _partition_into_blocks(n_coords: int, block_size: int, coords_per_atom: int = 3) -> list[Array]:
    """Partition flattened Cartesian coordinate indices into contiguous atom blocks.

    Assumes coordinates are ordered atom-major (atom 0's x,y,z, then atom 1's,
    ...), matching how full-atom frames are flattened elsewhere in this
    module. ``block_size`` is in atoms rather than raw coordinates so it
    stays meaningful regardless of ``coords_per_atom``. The final block is
    truncated (not dropped) when ``n_atoms`` isn't a multiple of
    ``block_size``.
    """
    if coords_per_atom <= 0:
        raise ValueError("coords_per_atom must be positive.")
    if n_coords % coords_per_atom != 0:
        raise ValueError(
            "Block covariance requires coordinates ordered atom-major with "
            f"coords_per_atom={coords_per_atom}; got {n_coords} total coordinates."
        )
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer number of atoms.")
    n_atoms = n_coords // coords_per_atom
    blocks = []
    for start_atom in range(0, n_atoms, block_size):
        end_atom = min(start_atom + block_size, n_atoms)
        blocks.append(np.arange(start_atom * coords_per_atom, end_atom * coords_per_atom))
    return blocks


def _block_log_probs(X: Array, weights: Array, means: Array, covariances: Array, blocks: list[Array]) -> Array:
    """log(weight_k * density_k(x)) for a block-diagonal-covariance mixture, per sample per component.

    Shared between fitting (to score the training log-likelihood for BIC) and
    :meth:`_BlockDiagonalGMM.predict_proba` (to score arbitrary new samples,
    e.g. a held-out validation set, against an already-fitted mixture).
    """
    from scipy.stats import multivariate_normal

    n_components = len(weights)
    log_probs = np.zeros((X.shape[0], n_components))
    for k in range(n_components):
        block_log_density = np.zeros(X.shape[0])
        for block in blocks:
            block_log_density += multivariate_normal.logpdf(
                X[:, block], mean=means[k, block], cov=covariances[k][np.ix_(block, block)], allow_singular=True
            )
        log_probs[:, k] = np.log(np.maximum(weights[k], np.finfo(float).tiny)) + block_log_density
    return log_probs


class _BlockDiagonalGMM:
    """Minimal ``GaussianMixture``-like result for a block-diagonal-covariance fit.

    Exposes just enough of sklearn's ``GaussianMixture`` interface
    (``n_components``, ``weights_``, ``covariances_``, ``bic``, ``predict_proba``)
    for :func:`compute_gmm_sigma_mode`'s k-selection loop and
    :func:`compute_gmm_sigma_mode_per_sample`'s responsibility computation to
    treat it the same as a real ``GaussianMixture``.
    """

    def __init__(self, weights: Array, covariances: Array, bic_value: float, *, means: Array, blocks: list[Array]):
        self.n_components = len(weights)
        self.weights_ = weights
        self.covariances_ = covariances  # (n_components, d, d); zero outside each block
        self.means_ = means
        self._blocks = blocks
        self._bic_value = bic_value

    def bic(self, X: Array) -> float:
        del X
        return self._bic_value

    def predict_proba(self, X: Array) -> Array:
        """Posterior responsibility of each component for each row of X."""
        from scipy.special import logsumexp

        log_probs = _block_log_probs(
            np.asarray(X, dtype=float), self.weights_, self.means_, self.covariances_, self._blocks
        )
        log_norm = logsumexp(log_probs, axis=1, keepdims=True)
        return np.exp(log_probs - log_norm)


def _fit_block_diagonal_gmm(
    flattened: Array,
    n_components: int,
    blocks: list[Array],
    *,
    reg_covar: float,
    n_init: int,
    seed: Optional[int],
) -> _BlockDiagonalGMM:
    """Fit a Gaussian mixture whose per-component covariance is block-diagonal.

    sklearn has no ``covariance_type`` for this. Responsibilities instead
    come from a converged ``covariance_type="diag"`` fit (cheap, stable), and
    each component's block covariances are computed in closed form from
    those responsibilities -- an exact M-step under the block-diagonal
    constraint, just not re-iterated against a re-derived E-step. That's a
    deliberate simplification: this only needs the resulting per-mode scale,
    not a maximum-likelihood block-covariance mixture model.
    """
    from sklearn.mixture import GaussianMixture
    from scipy.special import logsumexp

    seed_gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=seed,
    ).fit(flattened)

    resp = seed_gmm.predict_proba(flattened)  # (n_samples, n_components)
    n_k = resp.sum(axis=0)
    weights = n_k / flattened.shape[0]
    means = (resp.T @ flattened) / n_k[:, None]

    n_coords = flattened.shape[1]
    covariances = np.zeros((n_components, n_coords, n_coords))
    for k in range(n_components):
        diff = flattened - means[k]
        weighted_diff = diff * resp[:, k, None]
        for block in blocks:
            block_cov = (weighted_diff[:, block].T @ diff[:, block]) / n_k[k]
            block_cov.flat[:: len(block) + 1] += reg_covar
            covariances[k][np.ix_(block, block)] = block_cov

    log_probs = _block_log_probs(flattened, weights, means, covariances, blocks)
    log_likelihood = float(np.sum(logsumexp(log_probs, axis=1)))
    n_block_cov_params = sum(len(block) * (len(block) + 1) // 2 for block in blocks)
    n_params = (n_components - 1) + n_components * n_coords + n_components * n_block_cov_params
    bic = -2.0 * log_likelihood + n_params * np.log(flattened.shape[0])

    return _BlockDiagonalGMM(weights, covariances, bic, means=means, blocks=blocks)


def _validate_gmm_args(
    flattened: Array,
    *,
    max_components: int,
    covariance_type: str,
    block_size: Optional[int],
    coords_per_atom: int,
) -> Optional[list[Array]]:
    """Shared validation for :func:`compute_gmm_sigma_mode` and its per-sample variant.

    Returns the block partition (or ``None`` for non-block covariance types).
    """
    if flattened.shape[0] < 2:
        raise ValueError("data must contain at least two frames.")
    if max_components <= 0:
        raise ValueError("max_components must be positive.")
    if covariance_type not in ("full", "diag", "spherical", "block"):
        raise ValueError(
            f"Unsupported covariance_type={covariance_type!r}; use 'full', 'diag', 'spherical', or 'block'."
        )
    if covariance_type != "block":
        return None
    if block_size is None:
        raise ValueError("block_size (number of atoms per block) is required when covariance_type='block'.")
    return _partition_into_blocks(flattened.shape[1], block_size, coords_per_atom=coords_per_atom)


def _fit_gmm_with_selection(
    flattened: Array,
    *,
    n_components: Optional[int],
    max_components: int,
    covariance_type: str,
    blocks: Optional[list[Array]],
    reg_covar: float,
    n_init: int,
    seed: Optional[int],
) -> tuple[Any, dict[int, float]]:
    """Fit a Gaussian mixture, selecting ``n_components`` by BIC when not given.

    Shared by :func:`compute_gmm_sigma_mode` and
    :func:`compute_gmm_sigma_mode_per_sample` so both estimators are fit
    identically and only differ in how they collapse components to a scalar.
    """
    from sklearn.mixture import GaussianMixture

    def _fit(k: int):
        if covariance_type == "block":
            return _fit_block_diagonal_gmm(flattened, k, blocks, reg_covar=reg_covar, n_init=n_init, seed=seed)
        return GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            random_state=seed,
        ).fit(flattened)

    bic_by_k: dict[int, float] = {}
    if n_components is None:
        candidates = range(1, max(1, min(max_components, flattened.shape[0] - 1)) + 1)
        best_gmm = None
        for k in candidates:
            gmm = _fit(k)
            bic_by_k[k] = float(gmm.bic(flattened))
            if best_gmm is None or bic_by_k[k] < bic_by_k[best_gmm.n_components]:
                best_gmm = gmm
        gmm = best_gmm
    else:
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        gmm = _fit(n_components)
        bic_by_k[n_components] = float(gmm.bic(flattened))
    return gmm, bic_by_k


def compute_gmm_sigma_mode(
    data: Array,
    *,
    n_components: Optional[int] = None,
    max_components: int = 8,
    covariance_type: str = "full",
    block_size: Optional[int] = None,
    coords_per_atom: int = 3,
    reg_covar: float = 1e-6,
    n_init: int = 5,
    seed: Optional[int] = 0,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, float | int]]:
    """Estimate a mixture-of-Gaussians mode variance directly from training data.

    Fits a Gaussian mixture model to ``data`` (already aligned/normalized the
    same way training coordinates are) instead of deriving a mode variance
    from force curvature. Each component's covariance is reduced to a scalar
    via :func:`_component_scale`, and the returned estimate is the
    mixing-weight-averaged scalar across components, so within-mode
    anisotropy and between-mode heterogeneity are both represented without
    conflating them with the *separation* between modes.

    ``covariance_type="block"`` is a middle ground between ``"diag"``
    (ignores all correlation) and ``"full"`` (a dense ``d x d`` matrix per
    component, expensive and data-hungry in high dimensions): it keeps
    covariance only within contiguous groups of ``block_size`` atoms
    (``coords_per_atom`` coordinates each) and zeroes it between groups, so
    it captures local correlation (e.g. one atom's x/y/z, or a small bonded
    neighborhood) while staying far cheaper and more sample-efficient than
    ``"full"``. Requires ``block_size`` and atom-major-flattened Cartesian
    coordinates.

    When ``n_components`` is ``None``, the number of components is selected
    by BIC over ``1..max_components``. ``data`` may have any trailing shape;
    it is flattened per frame before fitting.
    """
    flattened = np.asarray(data, dtype=float).reshape((np.asarray(data).shape[0], -1))
    blocks = _validate_gmm_args(
        flattened, max_components=max_components, covariance_type=covariance_type,
        block_size=block_size, coords_per_atom=coords_per_atom,
    )

    gmm, bic_by_k = _fit_gmm_with_selection(
        flattened,
        n_components=n_components,
        max_components=max_components,
        covariance_type=covariance_type,
        blocks=blocks,
        reg_covar=reg_covar,
        n_init=n_init,
        seed=seed,
    )

    scale_covariance_type = "full" if covariance_type == "block" else covariance_type
    component_scales = np.asarray(
        [_component_scale(gmm.covariances_[k], scale_covariance_type) for k in range(gmm.n_components)]
    )
    weights = np.asarray(gmm.weights_)
    estimate = float(np.sum(weights * component_scales))
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("GMM mode-variance estimate is non-positive or non-finite.")

    if not return_diagnostics:
        return estimate

    diagnostics: dict[str, float | int] = {
        "n_components": int(gmm.n_components),
        "bic_by_k": bic_by_k,
        "component_weights": weights.tolist(),
        "component_scale": component_scales.tolist(),
        "covariance_type": covariance_type,
    }
    if covariance_type == "block":
        diagnostics["block_size"] = int(block_size)
        diagnostics["n_blocks"] = len(blocks)
    return estimate, diagnostics


def compute_gmm_sigma_mode_per_sample(
    data: Array,
    *,
    val_data: Optional[Array] = None,
    n_components: Optional[int] = None,
    max_components: int = 8,
    covariance_type: str = "full",
    block_size: Optional[int] = None,
    coords_per_atom: int = 3,
    reg_covar: float = 1e-6,
    n_init: int = 5,
    seed: Optional[int] = 0,
    return_diagnostics: bool = False,
) -> (
    tuple[Array, Optional[Array]]
    | tuple[Array, Optional[Array], dict[str, Any]]
):
    """Assign each frame its own basin-specific mode variance via GMM responsibilities.

    Fits a Gaussian mixture to ``data`` exactly like :func:`compute_gmm_sigma_mode`
    (same fitting/model-selection code, so the two are directly comparable),
    but instead of collapsing components with the mixture's global weights, it
    scores each frame against the fitted mixture with ``predict_proba`` and
    takes the responsibility-weighted average of the per-component scales:

        sigma_mode_sq[i] = sum_k( predict_proba(x_i)_k * component_scale_k )

    This is a soft assignment: a frame deep inside one basin gets essentially
    that basin's scale, while a frame near a boundary between basins gets a
    smooth blend of their scales rather than a discontinuous jump.

    ``val_data``, if given, is scored against the *same* fitted mixture
    (never refit) so validation frames are assigned responsibilities over the
    same basins as the training frames, and returned as the second element.
    """
    flattened = np.asarray(data, dtype=float).reshape((np.asarray(data).shape[0], -1))
    blocks = _validate_gmm_args(
        flattened, max_components=max_components, covariance_type=covariance_type,
        block_size=block_size, coords_per_atom=coords_per_atom,
    )

    gmm, bic_by_k = _fit_gmm_with_selection(
        flattened,
        n_components=n_components,
        max_components=max_components,
        covariance_type=covariance_type,
        blocks=blocks,
        reg_covar=reg_covar,
        n_init=n_init,
        seed=seed,
    )

    scale_covariance_type = "full" if covariance_type == "block" else covariance_type
    component_scales = np.asarray(
        [_component_scale(gmm.covariances_[k], scale_covariance_type) for k in range(gmm.n_components)]
    )
    weights = np.asarray(gmm.weights_)

    resp_train = gmm.predict_proba(flattened)  # (n_train, n_components)
    train_estimate = resp_train @ component_scales
    if not np.all(np.isfinite(train_estimate)) or np.any(train_estimate <= 0.0):
        raise ValueError("GMM per-sample mode-variance estimate is non-positive or non-finite.")

    val_estimate: Optional[Array] = None
    if val_data is not None:
        flattened_val = np.asarray(val_data, dtype=float).reshape((np.asarray(val_data).shape[0], -1))
        resp_val = gmm.predict_proba(flattened_val)
        val_estimate = resp_val @ component_scales
        if not np.all(np.isfinite(val_estimate)) or np.any(val_estimate <= 0.0):
            raise ValueError("GMM per-sample mode-variance estimate (val) is non-positive or non-finite.")

    if not return_diagnostics:
        return train_estimate, val_estimate

    diagnostics: dict[str, Any] = {
        "n_components": int(gmm.n_components),
        "bic_by_k": bic_by_k,
        "component_weights": weights.tolist(),
        "component_scale": component_scales.tolist(),
        "covariance_type": covariance_type,
        "train_mean_responsibility": resp_train.mean(axis=0).tolist(),
    }
    if covariance_type == "block":
        diagnostics["block_size"] = int(block_size)
        diagnostics["n_blocks"] = len(blocks)
    return train_estimate, val_estimate, diagnostics
