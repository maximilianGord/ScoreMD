from evaluate import EvaluationSettings, evaluate
from scoremd.utils import slurm
from scoremd.utils.file import (
    get_persistent_storage,
)  # this should be the very first line so that the .env file is loaded
import hashlib
import json
import logging
import os
import functools
from typing import Callable, List, Optional
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp
from hydra.conf import HydraConf, RunDir
from hydra.core.hydra_config import HydraConfig
from hydra_zen import builds, zen, store
from hydra_zen.typing import Builds
from scoremd.data.preprocess import CenterMolecule
import scoremd.models as diffusion_models
import wandb as wandb_lib
from scoremd.data.dataset import ToyDataset, Dataset, MuellerBrownSimulation
from scoremd.data.dataset.base import Datapoints
from scoremd.data.dataset.utils import compute_cg_local_sigma_mode, compute_full_atom_sigma_mode
from scoremd.models.mixture import MixtureOfModels
from scoremd.training import load_and_train, TrainingSchedule
from scoremd.training.train_state import EmaTrainState
from scoremd.training.weighting import (
    plot_weighting_function,
)
from scoremd.utils.hydra import SlurmInitializer, WandbInitializer
from scoremd.utils.ranges import assert_range_continuity
from stores import create_dataset_store, create_trainig_schedule_store, create_weighting_store, create_optimizer_store

log = logging.getLogger(__name__)


def _loss_options(ranged_loss) -> dict:
    """Return the configured keyword arguments of a Hydra-Zen loss partial."""
    return dict(getattr(ranged_loss.loss, "keywords", {}) or {})


def _set_runtime_loss_options(ranged_loss, **options) -> None:
    """Bind run-specific values without mutating another ranged loss's partial."""
    loss_factory = ranged_loss.loss
    if not isinstance(loss_factory, functools.partial):
        raise TypeError("TSM runtime options require a functools.partial loss factory.")
    keywords = dict(loss_factory.keywords or {})
    keywords.update(options)
    ranged_loss.loss = functools.partial(loss_factory.func, *loss_factory.args, **keywords)


def _force_positions(datapoints: Datapoints) -> jnp.ndarray:
    """Positions at which the force targets are evaluated (biased samples under enhanced sampling)."""
    return datapoints.data if datapoints.force_data is None else datapoints.force_data


def _forces_cache_path(dataset: Dataset, datapoints: Datapoints) -> Optional[str]:
    """Deterministic on-disk cache location for a (dataset, datapoints) pair's physical forces.

    Physical forces are a pure function of the sample positions, so once computed for a given
    dataset config and set of samples they never change; caching avoids repeating a potentially
    slow (e.g. per-sample OpenMM) computation across every run that reuses the same data.
    """
    try:
        sha256 = hashlib.sha256()
        sha256.update(repr(dataset).encode("utf-8"))
        sha256.update(np.asarray(_force_positions(datapoints)).tobytes())
        digest = sha256.hexdigest()
    except Exception as e:
        log.warning("Could not compute a forces cache key; forces will not be cached: %s", e)
        return None
    directory = os.path.join(get_persistent_storage(), getattr(dataset, "name", type(dataset).__name__), "forces")
    return os.path.join(directory, f"{digest}.npy")


def _precompute_forces(dataset: Dataset, datapoints: Optional[Datapoints]) -> Optional[Datapoints]:
    """Attach physical forces to datapoints before entering JAX training."""
    if datapoints is None or datapoints.forces is not None:
        return datapoints
    if not hasattr(dataset, "force") or not callable(dataset.force):
        raise TypeError("loss_type='tsm', 'sc' or 'fm' requires a dataset with a callable force(frame) method.")

    sample_shape = tuple(dataset.sample_shape)
    if datapoints.data.shape[1] != int(np.prod(sample_shape)):
        raise ValueError("Datapoint coordinate dimension does not match dataset.sample_shape.")

    cache_path = _forces_cache_path(dataset, datapoints)
    if cache_path is not None and os.path.exists(cache_path):
        try:
            cached_forces = np.load(cache_path)
            if cached_forces.shape == datapoints.data.shape:
                log.info("Loaded cached physical forces from %s", cache_path)
                if hasattr(dataset, "release_force_coordinates"):
                    dataset.release_force_coordinates(datapoints)
                return datapoints.replace(forces=jnp.asarray(cached_forces))
            log.warning(
                "Cached forces at %s have shape %s, expected %s; recomputing.",
                cache_path,
                cached_forces.shape,
                datapoints.data.shape,
            )
        except Exception as e:
            log.warning("Failed to load cached forces from %s: %s", cache_path, e)

    target_frames = np.asarray(_force_positions(datapoints)).reshape((-1, *sample_shape))
    force_coordinates = (
        dataset.force_coordinates_for(datapoints) if hasattr(dataset, "force_coordinates_for") else None
    )
    force_frames = (
        np.asarray(force_coordinates).reshape((len(datapoints), *np.asarray(force_coordinates).shape[1:]))
        if force_coordinates is not None
        else target_frames
    )
    forces = np.empty_like(target_frames)
    log.info("Precomputing physical forces for %d TSM/SC samples.", len(target_frames))
    if hasattr(dataset, "batched_forces"):
        # Analytic potentials: one jitted, vectorized call instead of a Python loop over frames.
        forces[:] = np.asarray(dataset.batched_forces(jnp.asarray(force_frames)), dtype=target_frames.dtype).reshape(
            target_frames.shape
        )
        force_frames = []
    for index, (target_frame, force_frame) in enumerate(zip(target_frames, force_frames)):
        force = np.asarray(dataset.force(jnp.asarray(force_frame)), dtype=target_frames.dtype)
        if force.shape != target_frame.shape:
            if not hasattr(dataset, "project_forces"):
                raise ValueError(f"dataset.force returned {force.shape}; expected {target_frame.shape}.")
            force = np.asarray(dataset.project_forces(force), dtype=target_frames.dtype).reshape(target_frame.shape)
        forces[index] = force
    if hasattr(dataset, "release_force_coordinates"):
        dataset.release_force_coordinates(datapoints)
    forces = forces.reshape(datapoints.data.shape)
    if cache_path is not None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, forces)
            log.info("Cached physical forces to %s", cache_path)
        except Exception as e:
            log.warning("Failed to cache forces to %s: %s", cache_path, e)
    return datapoints.replace(forces=jnp.asarray(forces))


def _attach_tsm_bank(train_data: Datapoints, force_losses) -> Datapoints:
    """Attach a precomputed TSM bank (precompute_tsm_mean_force.py) to the training data, if a loss names one.

    The bank stores, per training frame, M noisy samples (t, eps) in the TSM window and their force targets; the force
    term then uses those instead of the DSM sample and the per-frame force (see get_loss, tsm_mean_force_bank). The
    force term is the TSM term (loss_type='tsm', tsm_type='noise_cutoff', window tsm_sigma_max) or the force anchor
    of the sc switch (loss_type='sc', sc_switch=True, window force_sigma_max; the semigroup term above it needs no
    force targets).
    """
    banks = [loss for loss in force_losses if _loss_options(loss).get("tsm_mean_force_bank")]
    if not banks:
        return train_data
    if len(banks) > 1 or len(force_losses) > 1:
        raise ValueError("tsm_mean_force_bank supports a single force loss.")
    ranged_loss = banks[0]
    options = _loss_options(ranged_loss)
    loss_type = options.get("loss_type")
    if loss_type == "tsm" and options.get("tsm_type") == "noise_cutoff":
        window_key, window = "tsm_sigma_max", options.get("tsm_sigma_max", 0.01)
    elif loss_type == "sc" and options.get("sc_switch", False) and options.get("sg_type") != "mode_mixture":
        window_key, window = "force_sigma_max", options.get("force_sigma_max", 0.01)
    else:
        raise ValueError("tsm_mean_force_bank requires loss_type='tsm' with tsm_type='noise_cutoff', or loss_type='sc' "
                         "with sc_switch=True and a non-mode_mixture sg_type.")
    path = options["tsm_mean_force_bank"]
    bank = np.load(path)
    meta = json.loads(str(bank["meta"]))
    # Same frames in the same order? The bank's per-frame forces F(x0) must match this run's precomputed forces
    # (a hash of the frames is not enough: CPU and GPU give frames that differ by ~1e-6 nm after alignment).
    if meta["frames"] != train_data.data.shape[0]:
        raise ValueError(f"{path} has {meta['frames']} frames, this run has {train_data.data.shape[0]}.")
    f_run, f_bank = np.asarray(train_data.forces, dtype=np.float64), np.asarray(bank["force_x0"], dtype=np.float64)
    mismatch = np.sqrt(np.mean((f_run - f_bank) ** 2) / np.mean(f_run**2))
    if not mismatch < 1e-3:
        raise ValueError(f"{path} was computed for other training frames: relative rms difference of the per-frame "
                         f"forces {mismatch:.2e} (expected < 1e-3).")
    if not np.isclose(meta["sigma_max"], window):
        raise ValueError(f"bank sigma_max={meta['sigma_max']} differs from {window_key}={window}; precompute a bank "
                         f"with precompute_tsm_mean_force.py --sigma-max {window}.")
    target = options.get("tsm_bank_target", "mean_force")
    if target == "mean_force":
        tsm_forces = bank["mean_force"]
    elif target == "force_x0":
        tsm_forces = np.repeat(bank["force_x0"][:, None], bank["t"].shape[1], axis=1)
    else:
        raise ValueError(f"Unknown tsm_bank_target={target!r}; use 'mean_force' or 'force_x0'.")
    # fraction of training times ts ~ U(range) that fall into the bank's TSM window
    t1, t0 = ranged_loss.range
    weight = (meta["t_max"] - meta["t_min"]) / (t1 - t0)
    _set_runtime_loss_options(ranged_loss, tsm_bank_weight=float(weight))
    log.info(
        "TSM bank %s: %d frames x %d draws, sigma <= %g (t in [%g, %.5f]), target %s, TSM weight x %.4f",
        path, meta["frames"], bank["t"].shape[1], meta["sigma_max"], meta["t_min"], meta["t_max"], target, weight,
    )
    return train_data.replace(
        tsm_t=jnp.asarray(bank["t"], dtype=jnp.float32),
        tsm_noise=jnp.asarray(bank["noise"], dtype=jnp.float32),
        tsm_forces=jnp.asarray(tsm_forces, dtype=jnp.float32),
    )


def _prepare_tsm_inputs(
    dataset: Dataset,
    training_schedule,
    train_data: Datapoints,
    val_data: Optional[Datapoints],
    norm_factor: jnp.ndarray,
) -> tuple[Datapoints, Optional[Datapoints]]:
    """Prepare static TSM/SC inputs once, before the JIT-compiled training loop."""
    force_losses = [
        loss
        for loss in training_schedule.losses
        if _loss_options(loss).get("loss_type", "dsm") in ("tsm", "sc", "fm")
    ]
    if not force_losses:
        return train_data, val_data

    mode_mixture_losses = [
        loss
        for loss in force_losses
        if (
            _loss_options(loss).get("loss_type") == "tsm"
            and _loss_options(loss).get("tsm_type") == "mode_mixture"
        )
        or (
            _loss_options(loss).get("loss_type") == "sc"
            and _loss_options(loss).get("sg_type", "constant") == "mode_mixture"
        )
    ]
    if mode_mixture_losses:
        if np.asarray(norm_factor).size != 1:
            raise ValueError("Mode-mixture TSM requires a scalar coordinate norm_factor.")
        # Must run before _precompute_forces releases ALDP's retained full frames.
        mode_var_computation = getattr(dataset, "mode_var_computation", "data_hessian")
        per_sample_mode_var = mode_var_computation == "gmm_per_sample"
        if mode_var_computation == "potential":
            if not isinstance(dataset, MuellerBrownSimulation):
                raise ValueError("dataset.mode_var_computation=potential is only supported for Müller-Brown.")
            sigma_mode_sq = dataset.sigma_mode_sq_from_potential()
            diagnostics = {"source": "potential", "mode": "global_minimum"}
        elif mode_var_computation == "data_hessian":
            sigma_mode_sq, diagnostics = compute_full_atom_sigma_mode(dataset, return_diagnostics=True)
        elif mode_var_computation == "schur_hessian":
            from scoremd.data.dataset.utils import compute_schur_hessian_sigma_mode

            sigma_mode_sq, diagnostics = compute_schur_hessian_sigma_mode(dataset)
        elif mode_var_computation == "data_empirical":
            from scoremd.data.dataset.utils import compute_empirical_sigma_mode

            sigma_mode_sq = compute_empirical_sigma_mode(dataset)
            diagnostics = {"source": "data", "estimator": "centered_empirical_variance"}
        elif mode_var_computation == "cg_local_covariance":
            sigma_mode_sq, diagnostics = compute_cg_local_sigma_mode(train_data.data)
            diagnostics = {"source": "cg_data", "estimator": "local_covariance", **diagnostics}
        elif mode_var_computation == "gmm":
            from scoremd.data.dataset.utils import compute_gmm_sigma_mode

            sigma_mode_sq, diagnostics = compute_gmm_sigma_mode(
                train_data.data,
                n_components=getattr(dataset, "gmm_n_components", None),
                covariance_type=getattr(dataset, "gmm_covariance_type", "full"),
                block_size=getattr(dataset, "gmm_block_size", None),
                return_diagnostics=True,
            )
            diagnostics = {"source": "data", "estimator": "gaussian_mixture", **diagnostics}
        elif mode_var_computation == "gmm_per_sample":
            from scoremd.data.dataset.utils import compute_gmm_sigma_mode_per_sample

            train_sigma_mode_sq, val_sigma_mode_sq, diagnostics = compute_gmm_sigma_mode_per_sample(
                train_data.data,
                val_data=val_data.data if val_data is not None else None,
                n_components=getattr(dataset, "gmm_n_components", None),
                covariance_type=getattr(dataset, "gmm_covariance_type", "full"),
                block_size=getattr(dataset, "gmm_block_size", None),
                return_diagnostics=True,
            )
            diagnostics = {"source": "data", "estimator": "gaussian_mixture_per_sample", **diagnostics}
        else:
            raise ValueError(
                "dataset.mode_var_computation must be 'potential', 'data_hessian', 'schur_hessian', 'data_empirical', "
                "'cg_local_covariance', 'gmm', or 'gmm_per_sample'; "
                f"got {mode_var_computation!r}."
            )

        if per_sample_mode_var:
            norm_factor_sq = float(np.asarray(norm_factor) ** 2)
            train_sigma_mode_sq_normalized = norm_factor_sq * np.asarray(train_sigma_mode_sq)
            train_data = train_data.replace(sigma_mode_sq=jnp.asarray(train_sigma_mode_sq_normalized))
            if val_data is not None and val_sigma_mode_sq is not None:
                val_sigma_mode_sq_normalized = norm_factor_sq * np.asarray(val_sigma_mode_sq)
                val_data = val_data.replace(sigma_mode_sq=jnp.asarray(val_sigma_mode_sq_normalized))
            log.info(
                "Computed physical sigma_mode_sq=%g (per-sample mean, std=%g); "
                "normalized sigma_mode_sq=%g (per-sample mean, std=%g); "
                "mode_var_computation=%s; diagnostics=%s",
                float(np.mean(train_sigma_mode_sq)),
                float(np.std(train_sigma_mode_sq)),
                float(np.mean(train_sigma_mode_sq_normalized)),
                float(np.std(train_sigma_mode_sq_normalized)),
                mode_var_computation,
                diagnostics,
            )
            for ranged_loss in mode_mixture_losses:
                _set_runtime_loss_options(ranged_loss, kbT=float(dataset.kbT))
        else:
            sigma_mode_sq_normalized = float(np.asarray(norm_factor) ** 2 * sigma_mode_sq)
            log.info(
                "Computed physical sigma_mode_sq=%g; normalized sigma_mode_sq=%g; "
                "mode_var_computation=%s; diagnostics=%s",
                sigma_mode_sq,
                sigma_mode_sq_normalized,
                mode_var_computation,
                diagnostics,
            )
            for ranged_loss in mode_mixture_losses:
                _set_runtime_loss_options(ranged_loss, sigma_mode_sq=sigma_mode_sq_normalized, kbT=float(dataset.kbT))

    train_data = _precompute_forces(dataset, train_data)
    val_data = _precompute_forces(dataset, val_data)
    train_data = _attach_tsm_bank(train_data, force_losses)
    for ranged_loss in force_losses:
        _set_runtime_loss_options(ranged_loss, kbT=float(dataset.kbT))
        if _loss_options(ranged_loss).get("tsm_project_bonds", False):
            if not hasattr(dataset, "bond_pairs"):
                raise ValueError("tsm_project_bonds=True requires a dataset with bond_pairs().")
            pairs = dataset.bond_pairs()
            log.info("Projecting %d bond-stretch directions out of the TSM residual.", len(pairs))
            _set_runtime_loss_options(ranged_loss, bond_pairs=jnp.asarray(pairs))
    return train_data, val_data


def training_routine(
    dataset: Builds[Dataset],
    optimizer: Callable[[int], optax.GradientTransformation],
    ranged_models: List[Builds[diffusion_models.RangedModel]],
    training_schedule: Builds[TrainingSchedule],
    num_devices: Optional[int],
    weighting_function: Callable[
        [List[diffusion_models.RangedModel]], Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ],
    checkpoint_options: ocp.CheckpointManagerOptions,
    load_dir: Optional[str],
    continue_from: Optional[str],
    load_epoch: Optional[int],
    evaluation: EvaluationSettings,
    no_evaluation: bool,
    seed: int,
    wandb: dict,
    out_subdir: str = "out",
):
    assert len(ranged_models) > 0, "At least one model is required"
    assert continue_from is None or load_dir is None, "Please specify either load_dir or continue_from, not both."

    base_dir = HydraConfig.get().runtime.output_dir

    if continue_from is None:
        log.info(f"The unique output directory for this folder is {base_dir}.")
        model_load_dir = os.path.join(load_dir, "model") if load_dir is not None else None
    else:
        continue_from = os.path.abspath(os.path.expanduser(continue_from))
        if not os.path.exists(continue_from):
            log.warning(
                f"Directory {continue_from} does not exist, we will start a new run, but use the specified output directory {continue_from}."
            )
        log.info(
            f"The unique run directory for this folder is {base_dir} (i.e., hydra config will be stored there), however we will continue from {continue_from} and add our results to it."
        )
        base_dir = continue_from
        model_load_dir = os.path.join(continue_from, "model")

    if model_load_dir is not None:
        if not os.path.exists(model_load_dir):
            log.warning(
                f"Model directory {model_load_dir} does not exist, we will start a new run, not loading any models."
            )
            model_load_dir = None

    out_dir = os.path.join(base_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    log.info(
        f"Working directory: {os.getcwd()}, output directory: {out_dir}, tmpdir: {os.getenv('TMPDIR')}, persistent storage: {get_persistent_storage()}"
    )
    log.info(f"Loading dataset '{dataset.name}'")

    train_data, val_data = dataset.train, dataset.val
    log.info(
        f"Loaded dataset '{dataset.name}' with data shape: {train_data.data.shape} and features shape: {train_data.features.shape if train_data.features is not None else None}"
    )
    if val_data is not None:
        log.info(
            f"Validation data shape: {val_data.data.shape} and features shape: {val_data.features.shape if val_data.features is not None else None}"
        )

    BS = training_schedule.BS

    ranged_models = sorted(ranged_models, key=lambda x: x.range[0], reverse=True)
    log.info(f"Specified {len(ranged_models)} model(s) with range(s) {' -> '.join([str(m) for m in ranged_models])}")
    assert_range_continuity([m.range for m in ranged_models])

    # create the weighting function with the given models
    weighting_function = weighting_function(ranged_models)

    if len(ranged_models) > 1:
        plt.figure(clear=True)
        plot_weighting_function(weighting_function, x=None)
        plt.savefig(f"{out_dir}/model_weights.png", bbox_inches="tight")
        plt.close()

        if wandb["enabled"]:
            wandb_lib.save(f"{out_dir}/model_weights.png", base_path=out_dir)

    if dataset.sample_shape[-1] == 3:
        norm_factor = 1.0 / dataset.std
    else:
        # We don't normalize toy data because the distribution is not really hard
        # also, not everything is perfectly centered
        norm_factor = 1.0
    log.info(f"Normalization factor: {norm_factor}")

    train_data, val_data = _prepare_tsm_inputs(dataset, training_schedule, train_data, val_data, norm_factor)

    unified_model = MixtureOfModels(
        [m.build(dataset, norm_factor) for m in ranged_models], weighting_function, CenterMolecule(dataset)
    )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    if num_devices is None:
        if isinstance(dataset, ToyDataset) or isinstance(dataset, MuellerBrownSimulation):
            num_devices = 1
            log.info("num_devices not specified, using 1 device since we are using a simple dataset")
        else:
            num_devices = jax.device_count()
            log.info(f"num_devices not specified, using all {num_devices} devices")
    elif num_devices > jax.device_count():
        raise ValueError(
            f"Number of devices ({num_devices}) must be less than or equal to the number of available devices ({jax.device_count()})"
        )

    if BS > train_data.data.shape[0]:
        # Replicate coordinates, features, and precomputed projected forces together.
        log.warning(
            f"Batch size is larger than the dataset size. "
            f"To prevent errors, we increase the training data from {train_data.data.shape[0]} to {BS}."
        )
        new_train_data = jnp.concatenate([train_data.data] * BS, axis=0)[:BS]
        new_train_features = (
            None
            if train_data.features is None
            else jnp.concatenate([train_data.features] * BS, axis=0, dtype=train_data.features.dtype)[:BS]
        )
        new_train_forces = (
            None
            if train_data.forces is None
            else jnp.concatenate([train_data.forces] * BS, axis=0, dtype=train_data.forces.dtype)[:BS]
        )
        new_train_force_data = (
            None
            if train_data.force_data is None
            else jnp.concatenate([train_data.force_data] * BS, axis=0)[:BS]
        )
        train_data = train_data.replace(
            data=new_train_data,
            features=new_train_features,
            forces=new_train_forces,
            force_data=new_train_force_data,
        )
    if training_schedule.BS % num_devices != 0:
        raise ValueError(f"Batch size ({training_schedule.BS}) must be divisible by number of devices ({num_devices})")

    log.info(
        f"Using {num_devices} devices with batch size {training_schedule.BS} ({training_schedule.BS // num_devices} per device)"
    )

    init_params = unified_model.init(
        init_key,
        jnp.ones([BS // num_devices, train_data.data.shape[1]]),
        jnp.ones([BS // num_devices, *train_data.features.shape[1:]], dtype=train_data.features.dtype)
        if train_data.features is not None
        else None,
        jnp.ones([BS // num_devices, 1]),
        training=False,
    )
    # for ema weight see https://arxiv.org/pdf/2011.13456 pages 26, we use same as in two for one 0.995
    state = EmaTrainState.create(apply_fn=unified_model.apply, params=init_params, tx=optimizer(1), ema_weight=0.995)

    log.debug("Initialized model with params: %s", state.params)
    log.info(f"Initialized model {unified_model} with {sum(x.size for x in jax.tree.leaves(state.params))} parameters.")

    if wandb["enabled"]:
        wandb_lib.define_metric("train/*", step_metric="train/epoch")
        wandb_lib.define_metric("val/*", step_metric="val/epoch")

    log.info(
        f"Training for {training_schedule.total()} epochs, which is about {train_data.data.shape[0] // BS * training_schedule.total()} steps"
    )

    state, train_losses, val_losses = load_and_train(
        training_schedule,
        unified_model,
        state,
        optimizer,
        train_data,
        val_data,
        num_devices,
        norm_factor,
        key,
        checkpoint_options,
        os.path.join(base_dir, "model"),
        model_load_dir,
        continue_from is not None and model_load_dir is not None,
        load_epoch,
        wandb["enabled"],
    )
    log.info("Finished training. Evaluating model...")

    for losses, prefix, filename in zip([train_losses, val_losses], ["Training", "Validation"], ["loss", "val_loss"]):
        if losses is None or len(losses) == 0:
            continue

        # Persist the numerical histories as well as the rendered figures. Keeping
        # one file per component makes it possible to restyle/replot a completed
        # run without training again: loss0.npy, ..., lossN.npy (and validation
        # equivalents) each contain one value per epoch.
        losses_np = np.asarray(losses)
        np.save(f"{out_dir}/{filename}.npy", losses_np)
        np.save(f"{out_dir}/{filename}_total.npy", losses_np.sum(axis=-1))

        plt.figure(clear=True)
        plt.title(f"{prefix} - Loss")
        plt.plot(losses_np.sum(axis=-1))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{out_dir}/{filename}.png", bbox_inches="tight")
        plt.close()

        for i, (loss, title) in enumerate(
            zip(
                losses_np.T,
                ["Diffusion Loss", "Vector FP Loss", "Scalar FP Loss", "TSM Loss", "SC Loss"],
            )
        ):
            np.save(f"{out_dir}/{filename}{i}.npy", loss)
            if jnp.any(jnp.abs(loss) > 1e-6):
                plt.figure(clear=True)
                plt.title(f"{prefix} - {title}")
                plt.plot(loss)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(f"{out_dir}/{filename}{i}.png", bbox_inches="tight")
                plt.close()

    if no_evaluation:
        log.info("Skipping evaluation")
        return

    if evaluation.seed is None:
        evaluation.seed = seed
    tsm_force_contributions = {
        _loss_options(loss).get("tsm_force_contribution", "absolute")
        for loss in training_schedule.losses
        if _loss_options(loss).get("loss_type") == "tsm"
    }
    if len(tsm_force_contributions) > 1:
        raise ValueError("Evaluation cannot combine absolute and relative TSM force contributions.")
    log.info(f"Evaluating with {'EMA' if evaluation.use_ema else 'raw (non-EMA)'} parameters")
    evaluate(
        unified_model,
        state.ema_params if evaluation.use_ema else state.params,
        dataset,
        evaluation,
        BS // num_devices,  # This uses only a single GPU for evaluation
        norm_factor,
        wandb["enabled"],
        out_dir,
        tsm_force_contribution=next(iter(tsm_force_contributions), "absolute"),
    )


if __name__ == "__main__":
    create_dataset_store(store)
    create_weighting_store(store)
    create_optimizer_store(store)
    create_trainig_schedule_store(store)

    TrainConfig = builds(
        training_routine,
        checkpoint_options=builds(
            ocp.CheckpointManagerOptions,
            save_interval_steps=100,
            max_to_keep=3,
            create=True,
            cleanup_tmp_directories=True,
            enable_background_delete=True,
        ),
        load_dir=None,
        continue_from=None,
        load_epoch=None,
        num_devices=None,
        no_evaluation=False,
        evaluation=builds(EvaluationSettings, populate_full_signature=True),
        seed=1,
        wandb={"enabled": False, "project": "scoremd"},
        out_subdir="out",
        zen_meta={"model": None},
        populate_full_signature=True,
        hydra_defaults=[
            "_self_",
            {"dataset": "???"},
            {"training_schedule": "vp_standard"},
            {"optimizer": "cosine"},
            {"weighting_function": "ranged"},
            {"override /hydra/job_logging": "colorlog"},
            {"override /hydra/hydra_logging": "colorlog"},
        ],
    )

    output_root = os.environ.get("OUTPUT_ROOT", "outputs")
    run_dir_path = output_root + "/${dataset.name}/${now:%Y-%m-%d/%H-%M-%S}"

    slurm_job_id = slurm.get_slurm_job_id()
    if slurm_job_id is not None:
        run_dir_path += f"-{slurm_job_id}"

    store(TrainConfig, name="train")
    store(
        HydraConf(
            callbacks={
                "wandb": builds(WandbInitializer),
                "slurm": builds(SlurmInitializer),
            },
            run=RunDir(run_dir_path),
        )
    )

    store.add_to_hydra_store(overwrite_ok=True)

    zen(training_routine).hydra_main(
        config_name="train",
        version_base="1.3",
        config_path="config",
    )
