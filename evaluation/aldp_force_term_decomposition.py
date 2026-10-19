"""Split the atomistic ALDP force labels into the terms of the force field that produces them.

The true force is F = -grad U with U = sum_k U_k, the energy terms of the OpenMM System that generated the data
(bgmol's AImplicitUnconstrained: Amber ff99SB-ILDN + OBC implicit solvent, 22 atoms, no constraints). Every OpenMM
Force object is put into its own force group, so the force of term k alone, F_k, can be evaluated and F = sum_k F_k
holds exactly. Over --frames random frames of the dataset this script reports, per term k:
  * <|F_k|^2>: mean squared force per atom, (1 / (N * 22)) sum_i sum_a |F_k,a(x_i)|^2, in (kJ/mol/nm)^2,
    and its square root, the rms force per atom in kJ/mol/nm,
  * share s_k = <|F_k|^2> / sum_l <|F_l|^2> (normalized by the sum of the terms, not by <|F|^2>),
  * corr(F_k, F) = sum F_k . F / sqrt(sum |F_k|^2 sum |F|^2), sums over all frames, atoms and x/y/z,
and the cross terms <|F|^2> - sum_k <|F_k|^2> = sum_{k != l} <F_k . F_l>, also per pair of terms.

Writes to --out (default evaluation/force_term_decomposition/):
  force_term_decomposition.md    human-readable report: settings, force-field terms, result tables, command
  force_term_decomposition.json  the same numbers, machine-readable

Usage (from the repository root, so the dataset is found in ./storage):
    python evaluation/aldp_force_term_decomposition.py [--frames 5000] [--seed 0] [--platform CPU]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import openmm as mm
import openmm.unit as unit
from bgmol.datasets import AImplicitUnconstrained

from scoremd.utils.file import get_persistent_storage

DESCRIPTION = {  # what each OpenMM Force object of the system contains
    "HarmonicBondForce": "bond stretches",
    "HarmonicAngleForce": "bond-angle bends",
    "PeriodicTorsionForce": "dihedral terms (φ, ψ, ω, side-chain/methyl rotors, impropers)",
    "NonbondedForce": "Lennard-Jones + Coulomb",
    "GBSAOBCForce": "implicit solvent (OBC generalized Born)",
    "CMMotionRemover": "removes centre-of-mass motion, no force",
}


def num_terms(force):
    for method in ("getNumBonds", "getNumAngles", "getNumTorsions", "getNumParticles"):
        if hasattr(force, method):
            return getattr(force, method)()
    return 0


def git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        script = os.path.relpath(os.path.abspath(__file__), root)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", script], cwd=root, text=True).strip()
        return sha + (f" ({script} not committed)" if dirty else "")
    except Exception:
        return "unknown"


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", default="CPU", help="OpenMM platform: CPU or CUDA (CUDA uses mixed precision)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_term_decomposition"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. the OpenMM System that generated the data, one force group per Force object
    d = AImplicitUnconstrained(root=get_persistent_storage(), download=False, read=True)
    system = d.system.system
    n_atoms = system.getNumParticles()
    forces = system.getForces()
    names = [type(f).__name__ for f in forces]
    for k, f in enumerate(forces):
        f.setForceGroup(k)
    props = {"Precision": "mixed"} if args.platform == "CUDA" else {}
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(args.platform), props)

    # 2. random frames of the whole dataset (nm), total force and the force of every group (kJ/mol/nm)
    idx = np.random.default_rng(args.seed).choice(len(d.xyz), args.frames, replace=False)
    units = unit.kilojoule_per_mole / unit.nanometer
    total = np.empty((args.frames, n_atoms, 3))
    per_term = np.empty((len(names),) + total.shape)
    for i, k in enumerate(idx):
        ctx.setPositions(d.xyz[k])
        total[i] = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(units)
        for g in range(len(names)):
            per_term[g, i] = ctx.getState(getForces=True, groups={g}).getForces(asNumpy=True).value_in_unit(units)
    assert np.allclose(per_term.sum(0), total, rtol=1e-4, atol=1e-2), "force groups do not add up to the total force"

    # 3. statistics; terms without force (CMMotionRemover) are left out
    def msq(f):  # mean squared force per atom: |F_a|^2 summed over x/y/z, averaged over frames and atoms
        return float((f**2).sum(-1).mean())

    keep = [g for g in range(len(names)) if msq(per_term[g]) > 0]
    msq_total = msq(total)
    msq_terms = {names[g]: msq(per_term[g]) for g in keep}
    sum_terms = sum(msq_terms.values())
    terms = {}
    for g in keep:
        f = per_term[g]
        terms[names[g]] = {
            "description": DESCRIPTION.get(names[g], ""),
            "num_terms": num_terms(forces[g]),
            "mean_sq_force_per_atom": msq_terms[names[g]],
            "rms_force_per_atom": float(np.sqrt(msq_terms[names[g]])),
            "share": msq_terms[names[g]] / sum_terms,
            "corr_with_total": float((f * total).sum() / np.sqrt((f**2).sum() * (total**2).sum())),
        }
    cross = {}  # 2 <F_k . F_l> per atom for every pair
    for a, g in enumerate(keep):
        for h in keep[a + 1:]:
            cross[f"{names[g]} x {names[h]}"] = float(2 * (per_term[g] * per_term[h]).sum(-1).mean())

    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        "git_commit": git_commit(),
        "frames": args.frames, "seed": args.seed, "platform": args.platform, "dataset_frames": len(d.xyz),
        "atoms": n_atoms,
    }
    results = {"terms": terms, "mean_sq_force_per_atom_total": msq_total, "sum_of_terms": sum_terms,
               "cross_terms_total": msq_total - sum_terms, "cross_terms_per_pair": cross}
    with open(os.path.join(args.out, "force_term_decomposition.json"), "w") as fh:
        json.dump({**meta, **results}, fh, indent=2)

    # 4. report
    out = ["# Force-field decomposition of the atomistic ALDP force labels", "", "## Settings", ""]
    out += [markdown_table(["setting", "value"], [
        ["date", meta["date"]],
        ["command", f"`{meta['command']}`"],
        ["git commit", meta["git_commit"]],
        ["force field", "OpenMM System of bgmol's `AImplicitUnconstrained` (the system that generated the data): "
                        "Amber ff99SB-ILDN + OBC implicit solvent, no constraints"],
        ["frames", f"{args.frames} drawn without replacement (seed {args.seed}) from all {len(d.xyz)} dataset frames"],
        ["atoms", n_atoms],
        ["OpenMM platform", args.platform + (" (mixed precision)" if args.platform == "CUDA" else "")],
    ]), ""]
    out += ["## Force-field terms", "",
            "Every OpenMM Force object is its own force group; the forces of all groups add up to the total force "
            "(checked).", ""]
    out += [markdown_table(["OpenMM force", "contains", "number of terms"],
                           [[f"`{n}`", DESCRIPTION.get(n, ""), num_terms(f)] for n, f in zip(names, forces)]), ""]
    out += ["## Size of each term", "",
            "⟨|F_k|²⟩ = (1/(N·atoms)) Σ_i Σ_a |F_k,a(x_i)|² (mean squared force per atom), rms = √⟨|F_k|²⟩, "
            "share s_k = ⟨|F_k|²⟩ / Σ_l ⟨|F_l|²⟩, corr(F_k, F) = Σ F_k·F / √(Σ|F_k|² Σ|F|²) over all frames, atoms "
            "and x/y/z.", ""]
    rows = [[f"`{n}`", f"{t['mean_sq_force_per_atom']:.4g}", f"{t['rms_force_per_atom']:.0f}", f"{t['share']:.1%}",
             f"{t['corr_with_total']:+.3f}"] for n, t in sorted(terms.items(), key=lambda kv: -kv[1]["share"])]
    rows.append(["sum of terms", f"{sum_terms:.4g}", "", "100%", ""])
    rows.append(["total force F", f"{msq_total:.4g}", f"{np.sqrt(msq_total):.0f}", "", ""])
    out += [markdown_table(["term", "⟨\\|F_k\\|²⟩, (kJ/mol/nm)²", "rms per atom, kJ/mol/nm", "share s_k",
                            "corr(F_k, F)"], rows), ""]
    out += ["## Cross terms", "",
            f"⟨|F|²⟩ − Σ_k ⟨|F_k|²⟩ = {msq_total - sum_terms:.4g} (kJ/mol/nm)². Per pair, 2⟨F_k·F_l⟩ per atom "
            "(negative: the two terms partly cancel):", ""]
    out += [markdown_table(["pair", "2⟨F_k·F_l⟩, (kJ/mol/nm)²"],
                           [[k, f"{v:+.3g}"] for k, v in sorted(cross.items(), key=lambda kv: kv[1])]), ""]
    out += ["## Reproduce", "", "```bash", "cd <repository root>", f"pixi run {meta['command']}", "```", ""]
    report = os.path.join(args.out, "force_term_decomposition.md")
    with open(report, "w") as fh:
        fh.write("\n".join(out))
    print("\n".join(out))
    print(f"Saved {report} and force_term_decomposition.json")


if __name__ == "__main__":
    main()
