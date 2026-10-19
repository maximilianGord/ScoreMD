"""Per-region torque error of learned ALDP force fields (atomistic), compared with the true OpenMM forces.

For each frame, the force along a dihedral (generalized torque, kJ/mol/rad) is the force projected onto the
dihedral's gradient. This script computes it for the true OpenMM force and for each model's learned force at
t = eval_t (the force Langevin simulation uses) and reports, per conformational region (REGIONS below) and per
dihedral (phi, psi, omega1, omega2):
  * mean and rms of the true torque (the mean sets the basin populations, the rms is mostly bond/angle vibration),
  * bias b = mean error (systematic, shifts basin populations) with its standard error,
  * scatter sigma_e = std of the error (frame-to-frame part), rms error = sqrt(b^2 + sigma_e^2),
  * rms error relative to the rms true torque, correlation.
Plus the relative rms error of the full Cartesian force (all directions, incl. bonds and angles) and maps of the
psi-torque bias and rms error per (phi, psi) bin.

--frames-from data (default) evaluates all models on the same training-data frames. --frames-from langevin evaluates
each model on frames of its own Langevin trajectories (out/aldp_langevin_trajectories.npy, after discarding the first
--burn-in fraction of every trajectory), i.e. where the simulation actually goes; regions are then those of each
model's own frames. Outputs of the langevin mode carry the suffix _langevin.

Coarse-grained runs (e.g. heavy atoms only) work as well: the dihedrals are remapped to the kept atoms and, on data
frames, the true force is the OpenMM force of the paired full-atom frame projected onto the kept atoms (the TSM label;
for CG it contains the fluctuations of the integrated-out atoms, so the rms error includes irreducible label noise,
while the mean error is still the bias with respect to the mean force). CG Langevin frames have no hydrogens, so
OpenMM cannot label them: with --reference binned (forced for CG runs), the reference torque of a frame is the mean
true torque of the data frames in the same (phi, psi) bin (all data frames, cached forces from
aldp_force_label_variance.py). This gives only the bias (conditioned on phi/psi), no rms error. Its outputs carry the
suffix _langevin_binned.

Only reads the run folders; writes to --out (default evaluation/torque_error/):
  torque_error<suffix>.md    human-readable report: settings, region definitions and all result tables
  torque_error<suffix>.json  the same numbers, machine-readable
  psi_torque_error_maps<suffix>.png

Usage:
    python evaluation/torque_error.py --runs DIR:LABEL [DIR:LABEL ...] [--frames 20000] [--frames-from langevin]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmm as mm
import openmm.unit as unit
import orbax.checkpoint as ocp
from hydra_zen import instantiate
from omegaconf import OmegaConf

from scoremd.data.preprocess import CenterMolecule
from scoremd.models.mixture import MixtureOfModels
from scoremd.utils.file import get_persistent_storage

# Atom indices of the full 22-atom system: C4, N6, CA8, C14, N16 (phi = C-N-CA-C, psi = N-CA-C-N).
DIHEDRALS = {"phi": [4, 6, 8, 14], "psi": [6, 8, 14, 16], "omega1": [0, 4, 6, 8], "omega2": [8, 14, 16, 18]}

# Conformational regions as boxes in (phi, psi), degrees, half-open [lo, hi). Frames in no box are "other".
# The boxes are a coarse, partly arbitrary partition of the Ramachandran plane, not fitted basin boundaries;
# each is named after the metastable state it contains.
REGIONS = {
    "C5": {"phi": (-180, -100), "psi": (100, 180), "role": "C5 (extended)"},
    "PII": {"phi": (-100, -40), "psi": (100, 180), "role": "polyproline II (extended)"},
    "alpha_R": {"phi": (-100, -40), "psi": (-100, 0), "role": "αR (right-handed helix)"},
    "alpha_L": {"phi": (0, 100), "psi": (0, 100), "role": "αL (left-handed helix, rare)"},
    "C7ax": {"phi": (0, 100), "psi": (-100, 0), "role": "C7ax (compact, rare)"},
}
OTHER = "other"
REGION_NAMES = ["all", *REGIONS, OTHER]


def dihedral_and_grad(p):
    """Dihedral (deg) and its gradient wrt the 4 atoms (Blondel-Karplus); p: (n, 4, 3) in nm."""
    r1, r2, r3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 3] - p[:, 2]
    m, n = np.cross(r1, r2), np.cross(r2, r3)
    l2 = np.linalg.norm(r2, axis=-1, keepdims=True)
    ang = np.arctan2(np.sum(np.cross(m, n) * r2, -1) / l2[:, 0], np.sum(m * n, -1))
    g1 = -l2 * m / np.sum(m * m, -1, keepdims=True)
    g4 = l2 * n / np.sum(n * n, -1, keepdims=True)
    a = np.sum(r1 * r2, -1, keepdims=True) / l2**2
    b = np.sum(r3 * r2, -1, keepdims=True) / l2**2
    g2 = (a - 1) * g1 - b * g4
    g3 = (b - 1) * g4 - a * g1
    return np.degrees(ang), np.stack([g1, g2, g3, g4], 1)


def torques(x, f, dihedrals=DIHEDRALS):
    """Generalized torque along each dihedral: least-squares projection of the force on the dihedral gradient."""
    out = {}
    for name, idx in dihedrals.items():
        ang, g = dihedral_and_grad(x[:, idx])
        out[name] = (ang, np.sum(f[:, idx] * g, axis=(1, 2)) / np.sum(g**2, axis=(1, 2)))
    return out


def load_force_fn(run_dir, eval_t, batch=256):
    """Rebuild the model exactly as train.py does, restore the EMA parameters, return physical force(x) in kJ/mol/nm."""
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    dataset = instantiate(cfg.dataset)
    ranged = [instantiate(m) for m in cfg.ranged_models]  # per element: instantiating the list returns configs
    weighting = instantiate(cfg.weighting_function)(ranged)
    norm = 1.0 / dataset.std
    model = MixtureOfModels([m.build(dataset, norm) for m in ranged], weighting, CenterMolecule(dataset))
    dim = int(np.prod(dataset.sample_shape))
    init = model.init(jax.random.PRNGKey(0), jnp.ones([batch, dim]), None, jnp.ones([batch, 1]), training=False)
    path = os.path.abspath(os.path.join(run_dir, "model"))
    with ocp.CheckpointManager(path) as mgr:
        step = mgr.latest_step()
        try:
            params = mgr.restore(step).ema_params
        except Exception:  # different hardware than at save time
            restored = mgr.restore(step, args=ocp.args.Composite(
                params=ocp.args.StandardRestore(init), ema_params=ocp.args.StandardRestore(init)))
            params = restored.ema_params
    kbT = float(dataset.kbT)

    @jax.jit
    def f(x):
        return kbT * model.apply(params, x * norm, None, eval_t, training=False, method=MixtureOfModels.force) * norm

    def force(x):
        flat = np.asarray(x).reshape(len(x), -1)  # the model expects (frames, 3 * atoms)
        out = [np.asarray(f(jnp.asarray(flat[i:i + batch]))) for i in range(0, len(flat), batch)]
        return np.concatenate(out).reshape(len(x), -1, 3)

    return force, dataset, step


def true_forces(system, x, platform="CPU"):
    """OpenMM forces in kJ/mol/nm for frames x (n, 22, 3) in nm."""
    props = {"Precision": "mixed"} if platform == "CUDA" else {}
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform), props)
    f = np.empty_like(x)
    for i, xi in enumerate(x):
        ctx.setPositions(xi)
        f[i] = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)
    return f


def langevin_frames(run_dir, n, burn_in, rng):
    """n random frames (n, 22, 3) of the run's Langevin trajectories, without the first burn_in fraction of each."""
    traj = np.load(os.path.join(run_dir, "out", "aldp_langevin_trajectories.npy"), mmap_mode="r")
    traj = traj.reshape(-1, traj.shape[-2], traj.shape[-1]) if traj.ndim == 3 else traj[None]
    start = int(burn_in * traj.shape[1])
    i = rng.integers(0, traj.shape[0], n)
    j = rng.integers(start, traj.shape[1], n)
    return np.asarray(traj[i, j], dtype=np.float64).reshape(n, -1, 3)


def kept_dihedrals(dataset):
    """DIHEDRALS with the atom indices remapped to the atoms kept by the dataset (identity for the full system)."""
    index = {a: k for k, a in enumerate(dataset._atoms_to_keep)}
    return {name: [index[a] for a in idx] for name, idx in DIHEDRALS.items()}


def binned_reference(bins, min_count=20):
    """Mean true torque per (phi, psi) bin and dihedral over all data frames, from the cached all-frame forces."""
    from bgmol.datasets import AImplicitUnconstrained

    root = get_persistent_storage()
    cache = os.path.join(root, "aldp_atomistic_forces.npy")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} missing: run evaluation/aldp_force_label_variance.py once to create it.")
    xyz = np.asarray(AImplicitUnconstrained(root=root, download=False, read=True).xyz)
    forces = np.load(cache, mmap_mode="r")
    sums, count = {name: 0.0 for name in DIHEDRALS}, 0.0
    for s in range(0, len(xyz), 200_000):
        t = torques(xyz[s:s + 200_000].astype(np.float64), np.asarray(forces[s:s + 200_000], dtype=np.float64))
        sample = [t["phi"][0], t["psi"][0]]
        count = count + np.histogramdd(sample, bins=[bins, bins])[0]
        for name in DIHEDRALS:
            sums[name] = sums[name] + np.histogramdd(sample, bins=[bins, bins], weights=t[name][1])[0]
    ok = count >= min_count
    return {name: np.where(ok, sums[name] / np.maximum(count, 1), np.nan) for name in DIHEDRALS}, count


def regions(phi, psi):
    """Boolean mask per region (REGION_NAMES order, incl. "all" and "other") for angles in degrees."""
    masks = {"all": np.ones(len(phi), bool)}
    for name, box in REGIONS.items():
        (p0, p1), (q0, q1) = box["phi"], box["psi"]
        masks[name] = (phi >= p0) & (phi < p1) & (psi >= q0) & (psi < q1)
    masks[OTHER] = ~np.any([masks[name] for name in REGIONS], axis=0)
    return masks


def error_stats(true, model):
    """Error statistics of model vs. true torques of one region (kJ/mol/rad); rms_error^2 = bias^2 + scatter^2."""
    n = len(true)
    if n < 2:
        return {"n": n}
    e = model - true
    bias, scatter, rms = float(e.mean()), float(e.std()), float(np.sqrt(np.mean(e**2)))
    rms_true = float(np.sqrt(np.mean(true**2)))
    return {
        "n": n,
        "mean_true": float(true.mean()),
        "rms_true": rms_true,
        "mean_error": bias,
        "mean_error_se": float(e.std(ddof=1) / np.sqrt(n)),
        "scatter": scatter,
        "rms_error": rms,
        "rel_rms_error": rms / rms_true,
        "bias_share": bias**2 / rms**2,
        "corr": float(np.corrcoef(true, model)[0, 1]),
    }


def data_region_shares(dataset):
    """Share of all training frames per region (the target the Langevin shares should match)."""
    data = np.asarray(dataset.train.data, dtype=np.float64).reshape(len(dataset.train.data), -1, 3)
    d = kept_dihedrals(dataset)
    phi, _ = dihedral_and_grad(data[:, d["phi"]])
    psi, _ = dihedral_and_grad(data[:, d["psi"]])
    return {name: float(m.mean()) for name, m in regions(phi, psi).items()}, len(data)


def git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "evaluation/torque_error.py"], cwd=root,
                                        text=True).strip()
        return sha + (" (torque_error.py not committed)" if dirty else "")
    except Exception:
        return "unknown"


def fmt(v, spec):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(path, meta, results, data_shares, binned):
    """All results as Markdown tables: settings, region definitions, frames per region, force and torque errors."""
    out = [f"# Torque error of learned ALDP force fields ({meta['frames_from']} frames)", ""]
    out += ["## Settings", ""]
    out += [markdown_table(["setting", "value"], [
        ["date", meta["date"]],
        ["command", f"`{meta['command']}`"],
        ["git commit", meta["git_commit"]],
        ["frames", f"{meta['frames']} per model, "
                   + ("the same training-data frames for every model" if meta["frames_from"] == "data"
                      else f"from each model's own Langevin trajectories, first {meta['burn_in']:.0%} of each "
                           "trajectory discarded")],
        ["random seed", meta["seed"]],
        ["model force", f"k_BT · s_θ(x, t = {meta['eval_t']:g}), EMA parameters"],
        ["reference force", "mean true torque of the data frames in the same 10° (φ, ψ) bin" if binned
         else "OpenMM force (Amber ff99SB-ILDN + OBC implicit solvent) on the same frame"],
    ]), ""]
    out += [markdown_table(["model", "run directory", "checkpoint step"],
                           [[label, f"`{run}`", results[label]["checkpoint_step"]]
                            for label, run in meta["runs"].items()]), ""]

    out += ["## Regions", "",
            "Boxes in (φ, ψ) in degrees, half-open [lo, hi). φ = (C4, N6, CA8, C14), ψ = (N6, CA8, C14, N16), "
            "range (−180°, 180°]. A frame's region is set by the dihedrals of the frame itself. The boxes are a "
            "coarse, partly arbitrary partition of the Ramachandran plane, not fitted basin boundaries; each is "
            "named after the metastable state it contains. \"other\" is everything outside the boxes; for this "
            "dataset that is mostly the area between αR and the extended states (φ < −40°, 0° ≤ ψ < 100°) and the "
            "edges of the αR and C5 basins, not only high-energy states.", ""]
    rows = [[name, f"[{b['phi'][0]}, {b['phi'][1]})", f"[{b['psi'][0]}, {b['psi'][1]})", b["role"]]
            for name, b in REGIONS.items()]
    rows.append([OTHER, "all remaining (φ, ψ)", "", "outside the boxes (between and around the states)"])
    out += [markdown_table(["region", "φ range", "ψ range", "contains"], rows), ""]

    out += ["## Frames per region", "",
            f"Share of frames per region. The `data` row uses all {meta['n_data']} training frames.", ""]
    rows = [["data"] + [f"{data_shares[r]:.3f}" for r in REGION_NAMES[1:]]]
    for label, res in results.items():
        n_all = res["psi/all"]["n"]
        rows.append([label] + [f"{res[f'psi/{r}']['n'] / n_all:.3f} ({res[f'psi/{r}']['n']})" for r in REGION_NAMES[1:]])
    out += [markdown_table(["model", *REGION_NAMES[1:]], rows), ""]

    if binned:
        out += ["## Torque bias with respect to the data mean torque", "",
                "Bias = mean of (learned torque − data mean torque of the frame's (φ, ψ) bin), in kJ/mol/rad, "
                "± its standard error. Frames in bins with fewer than 20 data frames are left out.", ""]
        out += [markdown_table(["model", "frames in bins with data"],
                               [[label, f"{res['reference_coverage']:.1%}"] for label, res in results.items()]), ""]
        for name in DIHEDRALS:
            out += [f"### {name}", ""]
            rows = []
            for label, res in results.items():
                for r in REGION_NAMES:
                    s = res[f"{name}/{r}"]
                    if s["n"] < 2:
                        continue
                    rows.append([label, r, s["n"], fmt(s["mean_true"], "+.1f"),
                                 f"{s['mean_error']:+.1f} ± {s['mean_error_se']:.1f}"])
            out += [markdown_table(["model", "region", "n", "data mean torque", "bias ± SE"], rows), ""]
    else:
        out += ["## Cartesian force error", "",
                "Relative rms error √(Σ|F_model − F_true|² / Σ|F_true|²) over all atoms and directions, "
                "incl. bonds and angles.", ""]
        rows = [[label] + [fmt(res[f"cartesian/{r}"].get("rel_rms_error"), ".3f") for r in REGION_NAMES]
                for label, res in results.items()]
        out += [markdown_table(["model", *REGION_NAMES], rows), ""]

        out += ["## Torque error per dihedral and region", "",
                "All torques in kJ/mol/rad. Per region with n frames and per-frame error e = τ_model − τ_true:",
                "",
                "* **mean τ_true**: the true mean torque, the force that sets the populations along this dihedral.",
                "* **rms τ_true**: √⟨τ_true²⟩, mostly bond/angle vibration projected onto the dihedral.",
                "* **bias b** = ⟨e⟩ ± its standard error: systematic error; compare it with mean τ_true.",
                "* **scatter σ_e** = std(e): frame-to-frame error. rms error² = b² + σ_e².",
                "* **rel rms** = rms error / rms τ_true; **b²/rms²**: share of the squared error due to the bias.",
                ""]
        for name in DIHEDRALS:
            out += [f"### {name}", ""]
            rows = []
            for label, res in results.items():
                for r in REGION_NAMES:
                    s = res[f"{name}/{r}"]
                    if s["n"] < 2:
                        rows.append([label, r, s["n"]] + ["–"] * 8)
                        continue
                    rows.append([label, r, s["n"], f"{s['mean_true']:+.1f}", f"{s['rms_true']:.1f}",
                                 f"{s['mean_error']:+.1f} ± {s['mean_error_se']:.1f}", f"{s['scatter']:.1f}",
                                 f"{s['rms_error']:.1f}", f"{s['rel_rms_error']:.2f}", f"{s['bias_share']:.2f}",
                                 f"{s['corr']:.2f}"])
            out += [markdown_table(["model", "region", "n", "mean τ_true", "rms τ_true", "bias b ± SE", "scatter σ_e",
                                    "rms error", "rel rms", "b²/rms²", "corr"], rows), ""]

    out += ["## Reproduce", "", "```bash", "cd <repository root>",
            f"CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda pixi run {meta['command']}", "```", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def plot_maps(path, learned, truth, bins, binned, frames_from):
    """Maps of the psi-torque bias and rms error (binned reference: frame count) per (phi, psi) bin."""
    labels = list(learned)
    fig, ax = plt.subplots(2, len(labels), figsize=(3.6 * len(labels), 6.6), squeeze=False)
    for j, label in enumerate(labels):
        t_true = truth[label]
        phi, psi = t_true["phi"][0], t_true["psi"][0]
        cnt, _, _ = np.histogram2d(phi, psi, bins=[bins, bins])
        e = learned[label]["psi"][1] - t_true["psi"][1]
        valid = ~np.isnan(e)
        s1, _, _ = np.histogram2d(phi[valid], psi[valid], bins=[bins, bins], weights=e[valid])
        s2, _, _ = np.histogram2d(phi[valid], psi[valid], bins=[bins, bins], weights=e[valid] ** 2)
        ok = cnt >= 10
        mean = np.where(ok, s1 / np.maximum(cnt, 1), np.nan)
        rms = np.where(ok, np.sqrt(s2 / np.maximum(cnt, 1)), np.nan)
        im = ax[0, j].pcolormesh(bins, bins, mean.T, cmap="RdBu_r", vmin=-40, vmax=40)
        ax[0, j].set_title(f"{label}: psi-torque bias", fontsize=9)
        plt.colorbar(im, ax=ax[0, j])
        if binned:
            im = ax[1, j].pcolormesh(bins, bins, np.where(cnt > 0, cnt, np.nan).T, cmap="viridis",
                                     norm=matplotlib.colors.LogNorm())
            ax[1, j].set_title(f"{label}: Langevin frames per bin", fontsize=9)
        else:
            im = ax[1, j].pcolormesh(bins, bins, rms.T, cmap="magma", vmin=0, vmax=150)
            ax[1, j].set_title(f"{label}: psi-torque rms error", fontsize=9)
        plt.colorbar(im, ax=ax[1, j])
        for a in ax[:, j]:
            for box in REGIONS.values():  # region outlines
                (p0, p1), (q0, q1) = box["phi"], box["psi"]
                a.add_patch(matplotlib.patches.Rectangle((p0, q0), p1 - p0, q1 - q0, fill=False, lw=0.8, ls="--",
                                                         edgecolor="black"))
            a.set_xlabel("phi / deg")
            a.set_ylabel("psi / deg")
    reference = "data mean torque of the (phi, psi) bin" if binned else "true torque"
    fig.suptitle(f"learned minus {reference} along psi (kJ/mol/rad) on {frames_from} frames, bins with >= 10 frames")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="DIR:LABEL")
    p.add_argument("--frames", type=int, default=20_000)
    p.add_argument("--eval-t", type=float, default=1e-5)
    p.add_argument("--frames-from", choices=["data", "langevin"], default="data")
    p.add_argument("--burn-in", type=float, default=0.1, help="langevin mode: discarded fraction of each trajectory")
    p.add_argument("--platform", default="CPU", help="OpenMM platform for the true forces: CPU or CUDA")
    p.add_argument("--reference", choices=["exact", "binned"], default="exact",
                   help="langevin mode: OpenMM force on the frame (exact, full-atom runs only) or the data mean torque "
                        "of the frame's (phi, psi) bin (binned)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "torque_error"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [r.rsplit(":", 1) for r in args.runs]
    _, dataset, _ = load_force_fn(runs[0][0], args.eval_t)
    system = dataset._dataset.system.system
    keep = list(dataset._atoms_to_keep)
    dihedrals = kept_dihedrals(dataset)
    coarse = len(keep) < system.getNumParticles()
    binned = args.frames_from == "langevin" and (args.reference == "binned" or coarse)
    if args.frames_from == "langevin" and coarse and args.reference == "exact":
        print("coarse-grained runs: Langevin frames have no hydrogens, using --reference binned")
    suffix = "" if args.frames_from == "data" else ("_langevin_binned" if binned else "_langevin")
    bins = np.linspace(-180, 180, 37)
    rng = np.random.default_rng(args.seed)
    if args.frames_from == "data":
        data = np.asarray(dataset.train.data).reshape(len(dataset.train.data), -1, 3)
        full = np.asarray(dataset.force_coordinates_for(dataset.train)).reshape(len(data), -1, 3)
        idx = rng.choice(len(data), min(args.frames, len(data)), replace=False)
        x_data = data[idx].astype(np.float64)
        full = full[idx].astype(np.float64)
        assert np.allclose(full[:, keep], x_data, atol=1e-4), "full-atom frames do not match the dataset frames"
        f_data = true_forces(system, full, args.platform)[:, keep]
    if binned:
        ref, ref_count = binned_reference(bins)

    results, learned, truth = {}, {}, {}
    for run_dir, label in runs:
        if args.frames_from == "data":
            x, f_true = x_data, f_data
        else:
            x = langevin_frames(run_dir, args.frames, args.burn_in, rng)
            f_true = None if binned else true_forces(system, x, args.platform)
        force, _, step = load_force_fn(run_dir, args.eval_t)
        f_model = force(x.astype(np.float32))
        t_model = torques(x, f_model, dihedrals)
        if binned:  # reference torque: data mean of the frame's (phi, psi) bin
            ip = np.clip(np.digitize(t_model["phi"][0], bins) - 1, 0, len(bins) - 2)
            iq = np.clip(np.digitize(t_model["psi"][0], bins) - 1, 0, len(bins) - 2)
            t_true = {name: (t_model[name][0], ref[name][ip, iq]) for name in DIHEDRALS}
        else:
            t_true = torques(x, f_true, dihedrals)
        reg = regions(t_true["phi"][0], t_true["psi"][0])
        learned[label], truth[label] = t_model, t_true
        res = {"checkpoint_step": int(step)}
        if binned:
            covered = ~np.isnan(t_true["phi"][1])
            res["reference_coverage"] = float(covered.mean())
            for name in DIHEDRALS:
                for rname, msk in reg.items():
                    m = msk & covered
                    res[f"{name}/{rname}"] = error_stats(t_true[name][1][m], t_model[name][1][m])
                    res[f"{name}/{rname}"]["n_uncovered"] = int((msk & ~covered).sum())
        else:
            for rname, msk in reg.items():
                sq_true = np.sum(f_true[msk] ** 2, axis=(1, 2))
                sq_err = np.sum((f_model[msk] - f_true[msk]) ** 2, axis=(1, 2))
                res[f"cartesian/{rname}"] = {"n": int(msk.sum())}
                if msk.any():
                    res[f"cartesian/{rname}"] |= {
                        "rms_true": float(np.sqrt(np.mean(sq_true))),
                        "rms_error": float(np.sqrt(np.mean(sq_err))),
                        "rel_rms_error": float(np.sqrt(np.mean(sq_err) / np.mean(sq_true))),
                    }
            for name in DIHEDRALS:
                for rname, msk in reg.items():
                    res[f"{name}/{rname}"] = error_stats(t_true[name][1][msk], t_model[name][1][msk])
        results[label] = res
        print(f"done {label} (checkpoint {step})", flush=True)

    data_shares, n_data = data_region_shares(dataset)
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        "git_commit": git_commit(),
        "frames": len(x), "frames_from": args.frames_from, "reference": "binned" if binned else "exact",
        "coarse_grained": coarse, "burn_in": args.burn_in, "eval_t": args.eval_t, "seed": args.seed,
        "n_data": n_data, "runs": {label: run_dir for run_dir, label in runs},
    }
    with open(os.path.join(args.out, f"torque_error{suffix}.json"), "w") as fh:
        json.dump({**meta, "regions": REGIONS, "data_region_shares": data_shares, "results": results}, fh, indent=2)
    report = os.path.join(args.out, f"torque_error{suffix}.md")
    write_report(report, meta, results, data_shares, binned)
    plot_maps(os.path.join(args.out, f"psi_torque_error_maps{suffix}.png"), learned, truth, bins, binned,
              args.frames_from)
    print(open(report).read())
    print(f"Saved {report}, torque_error{suffix}.json and psi_torque_error_maps{suffix}.png in {args.out}")


if __name__ == "__main__":
    main()
