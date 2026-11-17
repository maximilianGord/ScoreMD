import jax.numpy as jnp
import wandb as wandb_lib
from scoremd.data.dataset import ALDPDataset, CGMinipeptideDataset, CoarseGrainingLevel
import matplotlib.pyplot as plt
from scoremd.data.dataset.protein import SingleProteinDataset
from scoremd.utils.evaluation import js_divergence, pairwise_distances, phi_psi_metrics
import nglview as nv
import logging
from typing import Callable, Optional
import jax
from scoremd.evaluate.utils import once_on_every_and_all, empty_if_none
from scoremd.rmsd import kabsch_align_many
from tqdm import tqdm
import numpy as onp
from scoremd.simulation import simulate
from scoremd.utils.plots import plot_fes_angles
from scoremd.evaluate.structure import key_angle_summary, plot_bond_density, structure_summary

# Free-energy plot settings: 400-point grid, Gaussian filter against sampling-noise jaggedness (phi 0.1 rad,
# psi 0.15 rad), plus a stronger 0.25 rad filter only above 20 (phi) / 15 (psi) against spikes in sparse regions;
# psi: values above 20 compressed smoothly (c + w tanh((F - c) / w)), staying below 24;
# phi: unchanged up to 40, above it compressed logarithmically (40 + 2 log(1 + (F - 40) / 2)), staying below 48.
FES_PHI = dict(n_grid=400, fes_smooth_sigma=0.1, smooth_above_fes=40, cap_mode="log", smooth_width=2.0, max_above=8.0,
               high_fes_smooth_sigma=0.25, high_fes_from=20.0)
FES_PSI = dict(n_grid=400, fes_smooth_sigma=0.15, smooth_above_fes=20, smooth_width=4,
               high_fes_smooth_sigma=0.25, high_fes_from=15.0)
import gc

log = logging.getLogger(__name__)


def plot_biased_free_energies(dataset, target_phi, target_psi, sampled_phi, sampled_psi, path_prefix: str, title=None):
    """With enhanced sampling (dataset.enhanced = metadynamics npz): phi/psi free energies of the target, the biased
    training data, and the samples, saved as <path_prefix>_{phi,psi}_free_energy_biased.png. The biased reference is
    the phi/psi of all metadynamics frames (backbone dihedrals, identical for every CG mapping that keeps the
    backbone). Never lets a plotting problem break the evaluation; returns the written paths."""
    enhanced = getattr(dataset, "enhanced", None)
    if enhanced is None:
        return []
    try:
        biased = onp.load(enhanced)
        biased_phi, biased_psi = biased["phi"], biased["psi"]
        paths = []
        for name, target, bias, sampled, fes, ylim, xlabel in (
            ("phi", target_phi, biased_phi, sampled_phi, FES_PHI, dict(bottom=-5, top=50), r"$\varphi$"),
            ("psi", target_psi, biased_psi, sampled_psi, FES_PSI, dict(top=25), r"$\psi$"),
        ):
            plt.figure(clear=True)
            if title:
                plt.title(title)
            plot_fes_angles(jnp.asarray(target), dataset.kbT, label="Target", **fes)
            plot_fes_angles(jnp.asarray(bias), dataset.kbT, label="Biased (MetaD data)", linestyle=":", color="0.4", **fes)
            plot_fes_angles(jnp.asarray(sampled), dataset.kbT, label="Sampled", linestyle="--", **fes)
            plt.xlabel(xlabel)
            plt.ylim(**ylim)
            plt.legend()
            path = f"{path_prefix}_{name}_free_energy_biased.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            paths.append(path)
        return paths
    except Exception as e:  # noqa: BLE001
        log.warning(f"Biased free-energy plots failed: {e}")
        plt.close("all")
        return []


def _structure_metrics(dataset, target, sampled, out_path: str, title: str, key_prefix: str) -> dict:
    """Pooled bond-length / bond-angle summary plot; never lets a plotting problem break the evaluation."""
    try:
        metrics = structure_summary(dataset, target, sampled, out_path, title)
        metrics |= key_angle_summary(dataset, target, sampled, out_path.replace("_structure.png", "_key_angles.png"), title)
        return {f"eval/{key_prefix}_{k}": v for k, v in metrics.items()}
    except Exception as e:  # noqa: BLE001
        log.warning(f"Structure summary failed: {e}")
        return {}


def evaluate_iid_samples(
    dataset: ALDPDataset | CGMinipeptideDataset | SingleProteinDataset,
    ground_truth_samples: jnp.ndarray,
    q_samples: jnp.ndarray,
    write_animation: Optional[Callable[[jnp.ndarray, str], None]],
    wandb: bool,
    prefix: str,
    out_dir: str,
    only_store_results: bool = False,
) -> dict:
    log.info(f"Evaluating iid samples for {prefix}")
    plt.figure(clear=True)
    dataset.plot_2d(q_samples, title=f"Histogram {prefix} (iid)")
    plt.savefig(f"{out_dir}/{prefix}_iid_phi_psi_histogram.png", bbox_inches="tight")
    plt.close()

    target_phi, target_psi = dataset.get_2d_features(ground_truth_samples)
    sampled_phi, sampled_psi = dataset.get_2d_features(q_samples)

    plt.figure(clear=True)
    plot_fes_angles(target_phi, dataset.kbT, label="Target", **FES_PHI)
    plot_fes_angles(sampled_phi, dataset.kbT, label="Sampled", linestyle="--", **FES_PHI)
    plt.xlabel(r"$\varphi$")
    plt.ylim(bottom=-5, top=50)
    plt.legend()
    plt.savefig(f"{out_dir}/{prefix}_iid_phi_free_energy.png", bbox_inches="tight")
    plt.close()

    plt.figure(clear=True)
    plot_fes_angles(target_psi, dataset.kbT, label="Target", **FES_PSI)
    plot_fes_angles(sampled_psi, dataset.kbT, label="Sampled", linestyle="--", **FES_PSI)
    plt.xlabel(r"$\psi$")
    plt.ylim(top=25)
    plt.legend()
    plt.savefig(f"{out_dir}/{prefix}_iid_psi_free_energy.png", bbox_inches="tight")
    plt.close()
    plot_biased_free_energies(dataset, target_phi, target_psi, sampled_phi, sampled_psi, f"{out_dir}/{prefix}_iid")

    if isinstance(dataset, SingleProteinDataset):
        plt.figure(clear=True)
        dataset.plot_contact_map(ground_truth_samples)
        plt.savefig(f"{out_dir}/{prefix}_contact_map_target.png", bbox_inches="tight")
        plt.close()

        plt.figure(clear=True)
        dataset.plot_contact_map(q_samples)
        plt.savefig(f"{out_dir}/{prefix}_contact_map_iid.png", bbox_inches="tight")
        plt.close()

    if wandb:
        wandb_lib.save(f"{out_dir}/{prefix}_target_phi_psi.png", base_path=out_dir)
        wandb_lib.log(
            {f"eval/{prefix}_iid_phi_psi_histogram": wandb_lib.Image(f"{out_dir}/{prefix}_iid_phi_psi_histogram.png")}
        )
        wandb_lib.log(
            {f"eval/{prefix}_iid_phi_free_energy": wandb_lib.Image(f"{out_dir}/{prefix}_iid_phi_free_energy.png")}
        )
        wandb_lib.log(
            {f"eval/{prefix}_iid_psi_free_energy": wandb_lib.Image(f"{out_dir}/{prefix}_iid_psi_free_energy.png")}
        )

        if isinstance(dataset, SingleProteinDataset):
            wandb_lib.log({"eval/contact_map_iid": wandb_lib.Image(f"{out_dir}/{prefix}_contact_map_iid.png")})
            wandb_lib.log({"eval/contact_map_target": wandb_lib.Image(f"{out_dir}/{prefix}_contact_map_target.png")})

    try:
        if write_animation is not None:
            write_animation(q_samples[:64], f"{out_dir}/{prefix}_iid_samples.pdb")

            view = nv.show_file(f"{out_dir}/{prefix}_iid_samples.pdb")
            html_file_path = f"{out_dir}/{prefix}_iid_samples.html"
            nv.write_html(html_file_path, view)

            if wandb:
                wandb_lib.save(f"{out_dir}/{prefix}_iid_samples.pdb", base_path=out_dir)
                wandb_lib.save(f"{out_dir}/{prefix}_iid_samples.html", base_path=out_dir)
    except Exception as e:
        log.warning(f"Failed to write animation: {e}")

    target_phi_psi = jnp.stack([target_phi, target_psi], axis=1)
    phi_psi_samples = jnp.stack([sampled_phi, sampled_psi], axis=1)

    rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(target_phi, target_psi, sampled_phi, sampled_psi, n_bins=64)
    structure_metrics = _structure_metrics(
        dataset, ground_truth_samples, q_samples, f"{out_dir}/{prefix}_iid_structure.png", f"{prefix} (iid)", f"{prefix}_iid"
    )

    return structure_metrics | {
        f"eval/{prefix}_iid_js_divergence": js_divergence(target_phi_psi, phi_psi_samples, bins=64),
        f"eval/{prefix}_iid_rms_fe_sq_error": rms_fe_sq_error,
        f"eval/{prefix}_iid_rms_mjs_error": rms_mjs_error,
    }


def simulate_molecule(
    dataset: ALDPDataset | CGMinipeptideDataset | SingleProteinDataset,
    force: Callable[[jnp.ndarray], jnp.ndarray],
    initial_positions: jnp.ndarray,
    features: Optional[jnp.ndarray],
    n_samples: int,
    n_steps: int,
    langevin_dt: Optional[float],
    seed: int,
):
    key = jax.random.PRNGKey(seed)
    key, velocity_key = jax.random.split(key)

    n_points = initial_positions.shape[0]
    initial_positions = initial_positions.reshape(n_points, *dataset.sample_shape)
    initial_velocities = jnp.sqrt(dataset.kbT / dataset.mass)[None, ...] * jax.random.normal(
        velocity_key, (n_points, *dataset.sample_shape)
    )

    def adjusted_force(x, features, point_idx):
        # flatten the input and convert back to (n_atoms, 3)
        return force(x.reshape(1, -1), features, point_idx=point_idx).reshape(*dataset.sample_shape)

    @jax.jit
    def multi_step(x, v, key):
        # step each trajectory individually
        trajectory_step = jax.vmap(dataset.langevin_step_function(n_steps, force=adjusted_force, dt=langevin_dt))

        # we need to take care of the keys
        keys = jax.random.split(key, n_points)
        return trajectory_step(x, v, keys, features=features, point_idx=jnp.arange(n_points))

    log.info(
        f"Simulating {n_samples} MD steps with {n_steps} intermediate steps. We use a timestep of {langevin_dt} ps and start from {n_points} starting points."
    )

    @jax.jit
    def _simulate_reshaped(initial_positions, initial_velocities, key):
        # output shape is (n_points, n_samples, -1)
        trajectories, velocities = simulate(initial_positions, initial_velocities, multi_step, n_samples, key)

        # trajectories now has shape (n_samples, n_points, *dataset.sample_shape), we change that to (n_points, n_samples, -1)
        trajectories = jnp.swapaxes(trajectories, 0, 1).reshape(n_points, n_samples, -1)
        velocities = jnp.swapaxes(velocities, 0, 1).reshape(n_points, n_samples, -1)

        return trajectories, velocities

    gc.collect()  # simulation may need a lot of memory, so we collect garbage
    trajectories, velocities = _simulate_reshaped(initial_positions, initial_velocities, key)

    log.info("Finished simulation")

    # filter out nan values
    nan_entries = jnp.isnan(trajectories).any(axis=(1, 2)).flatten()

    if nan_entries.sum() > 0:
        log.warning(f"Found {nan_entries.sum()} NaN entries. Filtering them out...")
        trajectories, velocities = trajectories[~nan_entries, ...], velocities[~nan_entries, ...]

    return trajectories, velocities


def evaluate_langevin_samples(
    dataset: ALDPDataset | CGMinipeptideDataset | SingleProteinDataset,
    baseline: jnp.ndarray,
    trajectories: jnp.ndarray,
    velocities: jnp.ndarray,
    max_num_openmm_evaluations: int,
    write_animation: Optional[Callable[[jnp.ndarray, str], None]],
    wandb: bool,
    prefix: str,
    out_dir: str,
    only_store_results: bool = False,
    save_trajectories: bool = True,
) -> dict:
    # store the trajectories and velocities
    if save_trajectories:
        onp.save(f"{out_dir}/{prefix}_langevin_trajectories.npy", trajectories)
        onp.save(f"{out_dir}/{prefix}_langevin_velocities.npy", velocities)
        log.info(
            f"Saved langevin trajectories and velocities to {out_dir}/{prefix}_langevin_trajectories.npy and {out_dir}/{prefix}_langevin_velocities.npy"
        )

    if only_store_results:
        return {}

    target_phi, target_psi = dataset.get_2d_features(baseline)

    initial_positions_phi, initial_positions_psi = dataset.get_2d_features(trajectories[:, 0])
    plt.figure(clear=True)
    plt.title("Starting positions")
    for i, (phi, psi) in enumerate(zip(initial_positions_phi, initial_positions_psi)):
        plt.plot(phi, psi, markersize=8, marker="*", label=f"Trajectory {prefix} {i}")
    if trajectories.shape[0] < 10:
        plt.legend()
    plt.xlim(-jnp.pi, jnp.pi)
    plt.ylim(-jnp.pi, jnp.pi)
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\psi$")
    plt.gca().set_box_aspect(1)
    plt.savefig(f"{out_dir}/{prefix}_langevin_start_positions.png", bbox_inches="tight")
    plt.close()

    if wandb:
        wandb_lib.log(
            {
                f"eval/{prefix}_langevin_start_positions": wandb_lib.Image(
                    f"{out_dir}/{prefix}_langevin_start_positions.png"
                )
            }
        )

    base_frame = trajectories[0, 0]

    @jax.jit
    def align_trajectory(trajectory):
        return kabsch_align_many(trajectory, base_frame)[0]

    # align each trajectory to the base frame, one at a time and kept on the host: aligning all
    # trajectories at once runs out of GPU memory for long simulations (e.g. 100 x 120k frames)
    trajectories = onp.stack([onp.asarray(align_trajectory(t)) for t in trajectories])

    def _plot_2d_histogram(trajectories: jnp.ndarray, i: Optional[int]):
        # plot the phi psi histogram of all trajectories individually and together
        plt.figure(clear=True)
        current_phi, current_psi = dataset.plot_2d(
            trajectories.reshape(-1, *dataset.sample_shape),
            title=f"Histogram {prefix} (simulated){empty_if_none(i, f' - Trajectory {i}')}",
            highlight=None if i is None else [0, -1],
        )
        plt.savefig(f"{out_dir}/{prefix}_langevin_phi_psi{empty_if_none(i, f'_{i}')}.png", bbox_inches="tight")
        plt.close()

        if wandb:
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_phi_psi_histogram{empty_if_none(i, f'_{i}')}": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_phi_psi{empty_if_none(i, f'_{i}')}.png"
                    )
                }
            )

        # now we plot the marginals
        plt.figure(clear=True)
        plt.title(f"$\\varphi$ Histogram {prefix} (simulated) {empty_if_none(i, f' - Trajectory {i}')}")
        plot_fes_angles(target_phi, dataset.kbT, label="Target", **FES_PHI)
        plot_fes_angles(current_phi, dataset.kbT, label="Sampled", linestyle="--", **FES_PHI)
        plt.xlabel(r"$\varphi$")
        plt.ylim(bottom=-5, top=50)
        plt.legend()
        plt.savefig(f"{out_dir}/{prefix}_langevin_phi_free_energy{empty_if_none(i, f'_{i}')}.png", bbox_inches="tight")
        plt.close()

        if wandb and i is None:  # only log full trajectory
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_phi_free_energy": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_phi_free_energy.png"
                    )
                }
            )

        # now we do psi
        plt.figure(clear=True)
        plt.title(f"$\\psi$ Histogram {prefix} (simulated) {empty_if_none(i, f' - Trajectory {i}')}")
        plot_fes_angles(target_psi, dataset.kbT, label="Target", **FES_PSI)
        plot_fes_angles(current_psi, dataset.kbT, label="Sampled", linestyle="--", **FES_PSI)
        plt.xlabel(r"$\psi$")
        plt.ylim(top=25)
        plt.legend()
        plt.savefig(f"{out_dir}/{prefix}_langevin_psi_free_energy{empty_if_none(i, f'_{i}')}.png", bbox_inches="tight")
        plt.close()
        if i is None:  # all trajectories together only
            plot_biased_free_energies(dataset, target_phi, target_psi, current_phi, current_psi,
                                      f"{out_dir}/{prefix}_langevin", title=f"Free energy {prefix} (simulated)")

        if wandb and i is None:  # only log full trajectory
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_psi_free_energy": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_psi_free_energy.png"
                    )
                }
            )

    once_on_every_and_all(trajectories, _plot_2d_histogram, desc="Plotting phi psi histogram")

    # see how many transitions are happening
    for i, t in enumerate(trajectories):
        phi, psi = dataset.get_2d_features(t)
        plt.figure(clear=True)
        plt.title(f"$\\varphi$ over time {prefix} {i}")
        plt.scatter(range(len(phi)), phi, label=f"Trajectory {prefix} {i}", c=psi, s=10)
        plt.xlabel("Step")
        plt.ylabel(r"$\varphi$")
        plt.ylim(-jnp.pi, jnp.pi)
        plt.savefig(f"{out_dir}/{prefix}_langevin_phi_over_t_{i}.png", bbox_inches="tight")
        plt.close()

        if wandb:
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_phi_over_t_{i}": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_phi_over_t_{i}.png"
                    )
                }
            )

    target_distances = pairwise_distances(dataset.train.data)

    def _plot_bond_lengths(trajectories: jnp.ndarray, i: Optional[int]):
        # only the bonded pairs: all pairwise distances of every frame do not fit in memory for long simulations
        positions = onp.asarray(trajectories).reshape(trajectories.shape[0], -1, 3)

        for a0, a1 in dataset.bonds:
            a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
            bond_lengths = onp.linalg.norm(positions[:, a0] - positions[:, a1], axis=-1)
            # plot the histogram of the distances
            plt.figure(clear=True)
            which = "Langevin, all trajectories" if i is None else f"Langevin trajectory {i}"
            plt.title(f"Histogram of distances between {a0_name} and {a1_name} ({prefix}, {which})")
            plt.hist(target_distances[:, a0, a1], bins=100, density=True, label="Target")
            plt.hist(bond_lengths, bins=100, density=True, label="Sampled", alpha=0.7)
            plt.xlim(target_distances[:, a0, a1].min(), target_distances[:, a0, a1].max())
            plt.legend()
            plt.savefig(
                f"{out_dir}/{prefix}_langevin_bond_lengths_{a0}_{a1}{empty_if_none(i, f'_{i}')}.png",
                bbox_inches="tight",
            )
            plt.close()

            if wandb:
                wandb_lib.log(
                    {
                        f"eval/{prefix}_langevin_bond_lengths_{a0}_{a1}{empty_if_none(i, f'_{i}')}": wandb_lib.Image(
                            f"{out_dir}/{prefix}_langevin_bond_lengths_{a0}_{a1}{empty_if_none(i, f'_{i}')}.png"
                        )
                    }
                )

    once_on_every_and_all(
        trajectories,
        _plot_bond_lengths,
        desc="Plotting bond lengths",
        call_on_all=not isinstance(dataset, CGMinipeptideDataset),  # we might not have enough memory for this
    )
    # paper-style per-bond densities (all trajectories together, in Angstrom)
    try:
        all_distances = pairwise_distances(trajectories.reshape(-1, *dataset.sample_shape))
        for a0, a1 in dataset.bonds:
            plot_bond_density(
                target_distances[:, a0, a1],
                all_distances[:, a0, a1],
                f"{dataset.atom_names[a0]}-{dataset.atom_names[a1]}",
                f"{out_dir}/{prefix}_langevin_bond_density_{a0}_{a1}.png",
                f"{prefix}, Langevin (all trajectories)",
            )
        del all_distances
    except Exception as e:  # noqa: BLE001
        log.warning(f"Bond density plots failed: {e}")

    structure_metrics = _structure_metrics(
        dataset, dataset.train.data, trajectories, f"{out_dir}/{prefix}_langevin_structure.png",
        f"{prefix} (Langevin)", f"{prefix}_langevin",
    )

    if hasattr(dataset, "coarse_graining_level") and dataset.coarse_graining_level == CoarseGrainingLevel.NONE:
        ground_truth_energies = []
        for i, t in enumerate(trajectories):
            ground_truth_energies.append(
                jnp.array(
                    [
                        dataset.energy(x)
                        for x in tqdm(
                            t[:max_num_openmm_evaluations],
                            desc=f"Computing ground truth energies with openmm {i + 1}/{len(trajectories)}",
                        )
                    ]
                )
            )

        if wandb:
            for i in range(len(ground_truth_energies)):
                wandb_lib.define_metric(f"eval/{prefix}_energy_{i}", step_metric=f"eval/{prefix}_langevin_steps")

            for j in range(len(ground_truth_energies[0])):
                logging_dict = {"eval/{prefix}_langevin_steps": j}
                for i in range(len(ground_truth_energies)):
                    logging_dict |= {f"eval/{prefix}_energy_{i}": ground_truth_energies[i][j]}
                wandb_lib.log(logging_dict)

        for i, ground_truth_energy in enumerate(ground_truth_energies):
            plt.figure(clear=True)
            plt.plot(ground_truth_energy)
            plt.xlabel("Step")
            plt.ylabel("Energy")
            plt.savefig(f"{out_dir}/{prefix}_langevin_energy_{i}.png", bbox_inches="tight")
            plt.close()

    for i, t in enumerate(trajectories):
        try:
            if write_animation is not None:
                write_animation(t[:1000], f"{out_dir}/{prefix}_langevin_samples_{i}.pdb")

            if wandb:
                wandb_lib.save(f"{out_dir}/{prefix}_langevin_samples_{i}.pdb", base_path=out_dir)
        except Exception as e:
            log.warning(f"Failed to write {i}-th animation: {e}")

    # contact map for all trajectories in total
    if isinstance(dataset, SingleProteinDataset):
        plt.figure(clear=True)
        dataset.plot_contact_map(trajectories.reshape(-1, *dataset.sample_shape))
        plt.savefig(f"{out_dir}/{prefix}_contact_map_langevin.png", bbox_inches="tight")
        plt.close()

        if wandb:
            wandb_lib.log(
                {"eval/contact_map_langevin": wandb_lib.Image(f"{out_dir}/{prefix}_contact_map_langevin.png")}
            )

    target_phi_psi = jnp.stack([*dataset.get_2d_features(dataset.train.data)], axis=1)
    sampled_phi_psi = jnp.stack(
        [*dataset.get_2d_features(trajectories.reshape(-1, *dataset.sample_shape))], axis=1
    )

    rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(
        target_phi, target_psi, sampled_phi_psi[:, 0], sampled_phi_psi[:, 1], n_bins=64
    )

    return structure_metrics | {
        f"eval/{prefix}_langevin_js_divergence": js_divergence(target_phi_psi, sampled_phi_psi, bins=64),
        f"eval/{prefix}_langevin_rms_fe_sq_error": rms_fe_sq_error,
        f"eval/{prefix}_langevin_rms_mjs_error": rms_mjs_error,
    }
