"""Enhanced-sampling data for ALDP: well-tempered metadynamics on (phi, psi), then production with the bias frozen.

Same system and integrator as the unbiased training data (bgmol AImplicitUnconstrained: implicit solvent, no
constraints, Langevin 300 K, friction 1/ps, 1 fs).

  build    well-tempered metadynamics (openmm.app.Metadynamics) on phi = (4, 6, 8, 14), psi = (6, 8, 14, 16);
           saves the final bias grid V(phi, psi) (kJ/mol) to <out>/bias.npz.
  produce  plain Langevin MD with the *frozen* bias V as an extra force (force group 1). The frames are samples of
           p_b(x) ∝ exp(-(U(x) + V(phi, psi)) / kT), with V known and fixed. Saves per frame: full-atom positions,
           V, phi, psi and the unbiased potential energy U to <out>/walker<k>.npz. No forces are stored: the training
           code computes the (unbiased) forces from the positions with the unbiased system.
  combine  concatenates the walkers into <out>/aldp_metad.npz (xyz, bias, phi, psi, energy, walker) and prints a
           sanity check: the reweighted (w ∝ exp(V/kT)) phi/psi distribution vs. the unbiased 1 µs trajectory.

Usage:
  python generate_aldp_metad.py build   --ns 10
  python generate_aldp_metad.py produce --walker 0 --ns 25
  python generate_aldp_metad.py combine
"""

import argparse
import json
import os
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

PHI, PSI = (4, 6, 8, 14), (6, 8, 14, 16)
BIAS_GROUP = 1
DEFAULT_OUT = "/home/go57nuk/ScoreMD/storage/aldp_metad"


def load_system():
    from bgmol.datasets import AImplicitUnconstrained
    from scoremd.utils.file import get_persistent_storage

    d = AImplicitUnconstrained(root=get_persistent_storage(), download=False, read=False)
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(d.system.system))  # private copy
    for f in system.getForces():
        if f.getForceGroup() != 0:
            raise RuntimeError("expected all ALDP forces in group 0")
    integ = d.integrator
    return system, d.system.topology, np.asarray(d.system.positions), (
        integ.getTemperature(), integ.getFriction(), integ.getStepSize())


def torsion(atoms):
    f = mm.CustomTorsionForce("theta")
    f.addTorsion(*atoms)
    return f


def frozen_bias_force(grid):
    """CustomCVForce identical to openmm.app.Metadynamics' bias force, with a fixed table."""
    g = np.load(grid)
    force = mm.CustomCVForce("table(cv0, cv1)")
    force.addCollectiveVariable("cv0", torsion(PHI))
    force.addCollectiveVariable("cv1", torsion(PSI))
    widths, limits = [int(w) for w in g["widths"]], [float(v) for v in g["limits"]]
    force.addTabulatedFunction("table", mm.Continuous2DFunction(*widths, g["bias"].flatten(), *limits, True))
    force.setForceGroup(BIAS_GROUP)
    return force


def platform_props(name):
    plat = mm.Platform.getPlatformByName(name)
    return plat, ({"Precision": "mixed"} if name == "CUDA" else {"Threads": "1"} if name == "CPU" else {})


def cvs(context):
    pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return dihedral(pos, PHI), dihedral(pos, PSI)


def dihedral(x, idx):
    p0, p1, p2, p3 = (x[i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def build(args):
    system, top, pos, (temp, fric, dt) = load_system()
    phi = app.BiasVariable(torsion(PHI), -np.pi, np.pi, args.width, periodic=True, gridWidth=args.grid)
    psi = app.BiasVariable(torsion(PSI), -np.pi, np.pi, args.width, periodic=True, gridWidth=args.grid)
    meta = app.Metadynamics(system, [phi, psi], temp, args.bias_factor, args.height * unit.kilojoule_per_mole,
                            args.pace)
    plat, props = platform_props(args.platform)
    sim = app.Simulation(top, system, mm.LangevinIntegrator(temp, fric, dt), plat, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(temp, args.seed)
    sim.step(int(0.1 * unit.nanosecond / dt))  # 100 ps unbiased equilibration

    os.makedirs(args.out, exist_ok=True)
    n_steps, chunk = int(args.ns * unit.nanosecond / dt), int(args.log_ps * unit.picosecond / dt)
    traj, t0 = [], time.time()
    snapshots = []
    for k in range(n_steps // chunk):
        meta.step(sim, chunk)
        traj.append(cvs(sim.context))
        if (k + 1) % max(1, (n_steps // chunk) // 20) == 0:
            ns_done = (k + 1) * chunk * dt.value_in_unit(unit.nanosecond)
            speed = ns_done / ((time.time() - t0) / 86400)
            snapshots.append(np.array(meta._totalBias))
            print(f"  metad {ns_done:.2f}/{args.ns} ns, {speed:.0f} ns/day, max bias {meta._totalBias.max():.1f} kJ/mol",
                  flush=True)
    meta_info = dict(height=args.height, width=args.width, pace=args.pace, bias_factor=args.bias_factor,
                     ns=args.ns, grid=args.grid, temperature=temp.value_in_unit(unit.kelvin), phi=PHI, psi=PSI,
                     gamma=1 - 1 / args.bias_factor)
    np.savez(os.path.join(args.out, "bias.npz"), bias=np.array(meta._totalBias),
             widths=np.array([phi.gridWidth, psi.gridWidth]), limits=np.array([-np.pi, np.pi, -np.pi, np.pi]),
             snapshots=np.array(snapshots), cv_traj=np.array(traj), meta=json.dumps(meta_info),
             final_positions=sim.context.getState(getPositions=True).getPositions(asNumpy=True)
             .value_in_unit(unit.nanometer))
    # check: the frozen force reproduces the metadynamics bias energy
    check = mm.Context(_bias_only_system(system.getNumParticles(), os.path.join(args.out, "bias.npz")),
                       mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    state = sim.context.getState(getPositions=True)
    check.setPositions(state.getPositions())
    e_meta = sim.context.getState(getEnergy=True, groups={meta._force.getForceGroup()}).getPotentialEnergy()
    e_frozen = check.getState(getEnergy=True).getPotentialEnergy()
    print(f"bias at final frame: metadynamics {e_meta}, frozen {e_frozen}", flush=True)
    print(f"saved {os.path.join(args.out, 'bias.npz')}", flush=True)


def _bias_only_system(n, grid):
    s = mm.System()
    for _ in range(n):
        s.addParticle(1.0)
    s.addForce(frozen_bias_force(grid))
    return s


def produce(args):
    system, top, pos, (temp, fric, dt) = load_system()
    grid = os.path.join(args.out, "bias.npz")
    system.addForce(frozen_bias_force(grid))
    start = np.load(grid)["final_positions"]
    plat, props = platform_props(args.platform)
    integ = mm.LangevinIntegrator(temp, fric, dt)
    integ.setRandomNumberSeed(1000 + args.walker)
    sim = app.Simulation(top, system, integ, plat, props)
    sim.context.setPositions(start)
    sim.context.setVelocitiesToTemperature(temp, 1000 + args.walker)
    sim.step(int(args.equil_ps * unit.picosecond / dt))  # decorrelate from the shared start

    stride = int(args.stride_ps * unit.picosecond / dt)
    n_frames = int(args.ns * 1000 / args.stride_ps)
    kj = unit.kilojoule_per_mole
    xyz = np.empty((n_frames, 22, 3), np.float32)
    bias, energy, phi, psi = (np.empty(n_frames) for _ in range(4))
    t0 = time.time()
    for k in range(n_frames):
        sim.step(stride)
        st = sim.context.getState(getPositions=True, getEnergy=True, groups={0})
        xyz[k] = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        energy[k] = st.getPotentialEnergy().value_in_unit(kj)
        bias[k] = sim.context.getState(getEnergy=True, groups={BIAS_GROUP}).getPotentialEnergy().value_in_unit(kj)
        phi[k], psi[k] = dihedral(xyz[k], PHI), dihedral(xyz[k], PSI)
        if (k + 1) % max(1, n_frames // 20) == 0:
            el = time.time() - t0
            print(f"  walker {args.walker}: {k + 1}/{n_frames} frames, {el / 60:.1f} min, "
                  f"~{el / (k + 1) * (n_frames - k - 1) / 60:.1f} min left", flush=True)
    out = os.path.join(args.out, f"walker{args.walker}.npz")
    np.savez(out + ".partial.npz", xyz=xyz, bias=bias, energy=energy, phi=phi, psi=psi,
             meta=json.dumps(dict(walker=args.walker, ns=args.ns, stride_ps=args.stride_ps, equil_ps=args.equil_ps)))
    os.replace(out + ".partial.npz", out)
    print(f"saved {out}", flush=True)


def combine(args):
    import glob

    files = sorted(glob.glob(os.path.join(args.out, "walker*.npz")))
    files = [f for f in files if not f.endswith(".partial.npz")]
    parts = [np.load(f) for f in files]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("xyz", "bias", "energy", "phi", "psi")}
    data["walker"] = np.concatenate([np.full(len(p["bias"]), i) for i, p in enumerate(parts)])
    kT = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
    meta = json.loads(str(np.load(os.path.join(args.out, "bias.npz"))["meta"]))
    meta.update(kT=kT, walkers=files, frames=int(len(data["bias"])))
    out = os.path.join(args.out, "aldp_metad.npz")
    np.savez(out + ".partial.npz", meta=json.dumps(meta), **data)
    os.replace(out + ".partial.npz", out)

    # sanity check against the unbiased trajectory (not used anywhere else)
    w = np.exp((data["bias"] - data["bias"].max()) / kT)
    ess = w.sum() ** 2 / (w**2).sum()
    print(f"saved {out}: {len(w)} frames from {len(files)} walkers; importance weights ESS {ess:.0f} "
          f"({100 * ess / len(w):.1f}%)")
    from bgmol.datasets import AImplicitUnconstrained
    from scoremd.utils.file import get_persistent_storage

    ref = np.asarray(AImplicitUnconstrained(root=get_persistent_storage(), download=False, read=True).xyz)
    ref_phi = _dihedrals(ref[::10], PHI)
    edges = np.linspace(-np.pi, np.pi, 37)
    p_ref = np.histogram(ref_phi, edges)[0] + 1e-12
    p_b = np.histogram(data["phi"], edges)[0] + 1e-12
    p_rw = np.histogram(data["phi"], edges, weights=w)[0] + 1e-12
    p_ref, p_b, p_rw = (p / p.sum() for p in (p_ref, p_b, p_rw))
    for name, p in (("biased", p_b), ("reweighted", p_rw)):
        m = 0.5 * (p + p_ref)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(p_ref * np.log(p_ref / m))
        print(f"  phi histogram JS vs. unbiased 1 µs: {name:10s} {js:.4f}; "
              f"P(phi > 0) = {p[edges[:-1] >= 0].sum():.4f} (unbiased {p_ref[edges[:-1] >= 0].sum():.4f})")


def _dihedrals(x, idx):
    p0, p1, p2, p3 = (x[:, i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "produce", "combine"])
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--ns", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    # build
    p.add_argument("--height", type=float, default=1.2, help="initial Gaussian height (kJ/mol)")
    p.add_argument("--width", type=float, default=0.35, help="Gaussian width (rad)")
    p.add_argument("--pace", type=int, default=500, help="steps between Gaussians")
    p.add_argument("--bias-factor", type=float, default=6.0)
    p.add_argument("--grid", type=int, default=72)
    p.add_argument("--log-ps", type=float, default=10.0)
    # produce
    p.add_argument("--walker", type=int, default=0)
    p.add_argument("--stride-ps", type=float, default=1.0)
    p.add_argument("--equil-ps", type=float, default=200.0)
    args = p.parse_args()
    if args.ns is None:
        args.ns = 10.0 if args.stage == "build" else 25.0
    {"build": build, "produce": produce, "combine": combine}[args.stage](args)


if __name__ == "__main__":
    main()
