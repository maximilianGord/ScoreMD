from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    import scoremd.diffusion.classic.sde as sdes


def _optimal_tsm_lambda(sigma_sq: jnp.ndarray, sigma_data_sq: jnp.ndarray) -> jnp.ndarray:
    return (sigma_sq * sigma_data_sq) / jnp.maximum(sigma_sq + sigma_data_sq, 1e-12)


def tsm_weight(
    t_normalized: jnp.ndarray,
    sigma_t: jnp.ndarray,
    tsm_type: str,
    tsm_lambda: float,
    tsm_t0: float,
    tsm_sigma_max: float,
) -> jnp.ndarray:
    """Return gamma(t) for target score matching."""
    gamma0 = jnp.asarray(tsm_lambda, dtype=t_normalized.dtype)
    t0 = jnp.asarray(tsm_t0, dtype=t_normalized.dtype)
    sigma_max = jnp.asarray(tsm_sigma_max, dtype=t_normalized.dtype)

    if tsm_type == "constant":
        return jnp.full_like(t_normalized, gamma0)
    if tsm_type == "hard_cutoff":
        return jnp.where(t_normalized <= t0, gamma0, 0.0)
    if tsm_type == "smooth_decay":
        return gamma0 * jnp.exp(-jnp.square(t_normalized / t0))
    if tsm_type in ("linear", "linear_decay"):
        return gamma0 * jnp.maximum(1.0 - t_normalized / t0, 0.0)
    if tsm_type == "noise_cutoff":
        return jnp.where(sigma_t <= sigma_max, gamma0, 0.0)
    raise ValueError(
        f"Unknown tsm_type: {tsm_type}. Use 'constant', 'hard_cutoff', "
        "'noise_cutoff', 'smooth_decay', 'linear', or 'mode_mixture'."
    )


def _as_batch_weights(value: ArrayLike, batch_size: int, dtype, name: str) -> jnp.ndarray:
    """Convert a scalar or per-sample value into a batch-shaped vector."""
    value = jnp.asarray(value, dtype=dtype).reshape(-1)
    if value.size not in (1, batch_size):
        raise ValueError(f"{name} must be a scalar or have one value per sample; got shape {value.shape}.")
    return jnp.broadcast_to(value, (batch_size,))


# should be called from utils.py
def tsm_loss(
    key: jax.random.PRNGKey,
    sde: sdes.VP,
    score: ArrayLike,
    x: ArrayLike,
    force: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    time_weighting: Callable[[ArrayLike], ArrayLike],
    tsm_type: str,
    tsm_lambda: float,
    tsm_t0: float,
    tsm_sigma_max: float,
    sigma_data: float = 1.0,
    sigma_mode_sq: Optional[float] = None,
    kbT: float = 1.0,
    reduce: Callable[[ArrayLike], ArrayLike] = jnp.nanmean,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute a time-weighted target score matching loss.

    ``x`` is the already perturbed sample ``x_t`` and ``force`` is the
    physical force evaluated at the corresponding data sample.  The force is
    converted to a score target as ``force / kbT``.  Sampling ``t`` and
    constructing the companion DSM term are deliberately left to the caller.

    ``sigma_data`` is the data standard deviation used by the mode-mixture
    schedule.  ``sigma_mode_sq`` optionally overrides its square when a local
    mode variance is known.

    For regular schedules, this returns ``(loss, gamma_t)`` where ``gamma_t``
    is the configured TSM weight.  For ``mode_mixture``, it returns
    ``(loss, kappa_t)``.  The returned loss contains the TSM contribution
    ``lambda_t * (1 - kappa_t) * TSM``; a caller combining DSM and TSM should
    weight its DSM contribution by ``lambda_t * kappa_t``.
    """
    del key  # The caller supplies the already-perturbed samples and times.

    x = jnp.asarray(x)
    force = jnp.asarray(force, dtype=x.dtype)
    t = jnp.asarray(t, dtype=x.dtype).reshape(-1)

    if x.ndim < 1:
        raise ValueError(f"x must include a batch dimension; got shape {x.shape}.")
    if force.shape != x.shape:
        raise ValueError(f"force must have the same shape as x; got {force.shape} and {x.shape}.")
    if t.size not in (1, x.shape[0]):
        raise ValueError(f"t must be scalar or have one value per sample; got shape {t.shape}.")
    t = jnp.broadcast_to(t, (x.shape[0],))

    if kbT <= 0.0:
        raise ValueError(f"kbT must be positive; got {kbT}.")

    score = jnp.asarray(score, dtype=x.dtype)
    if score.shape != x.shape:
        raise ValueError(f"score must return shape {x.shape}; got {score.shape}.")

    target_score = force / jnp.asarray(kbT, dtype=x.dtype)
    squared_error = jnp.square(score - target_score).reshape((x.shape[0], -1))
    loss_per_sample = jnp.mean(squared_error, axis=-1)
    time_weights = _as_batch_weights(time_weighting(t), x.shape[0], x.dtype, "time_weighting(t)")

    if tsm_type == "mode_mixture":
        sigma_t = _as_batch_weights(sde.std(t), x.shape[0], x.dtype, "sde.std(t)")
        alpha_t = jnp.exp(sde.log_mean_coeff(t))
        sigma_sq = jnp.square(sigma_t)
        if sigma_mode_sq is not None:
            if sigma_mode_sq <= 0.0:
                raise ValueError(f"sigma_mode_sq must be positive; got {sigma_mode_sq}.")
            sigma_data_sq = jnp.asarray(sigma_mode_sq, dtype=x.dtype)
        else:
            if sigma_data <= 0.0:
                raise ValueError(f"sigma_data must be positive; got {sigma_data}.")
            sigma_data_sq = jnp.square(jnp.asarray(sigma_data, dtype=x.dtype))

        kappa_t = sigma_sq / jnp.maximum(sigma_sq + jnp.square(alpha_t) * sigma_data_sq, 1e-12)
        lambda_t = _optimal_tsm_lambda(sigma_sq, sigma_data_sq)
        weighted_loss = time_weights * lambda_t  * loss_per_sample
        return weighted_loss, kappa_t, lambda_t

    sigma_t = _as_batch_weights(sde.std(t), x.shape[0], x.dtype, "sde.std(t)")
    gamma_t = tsm_weight(t, sigma_t, tsm_type, tsm_lambda, tsm_t0, tsm_sigma_max)
    weighted_loss = time_weights * gamma_t * loss_per_sample
    return weighted_loss, None, None



__all__ = ["tsm_loss"]
