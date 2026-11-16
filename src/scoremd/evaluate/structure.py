"""One-figure summary of the local structure (bond lengths and bond angles) of sampled molecules.

Every bond length r_b (and bond angle a_k) is standardized with the mean and std of the same bond in the
target data, z = (r_b - mean_target_b) / std_target_b, and the z-scores of all bonds are pooled into one
histogram. The target then follows a unit-width distribution by construction; a sampled histogram that is
wider means floppier bonds/angles, a shifted one means systematically too long/short bonds.
Bonds and angles that involve hydrogen are shown separately from heavy-atom-only ones.
"""

from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _element(name: str) -> str:
    return name[0]


def bond_list(dataset) -> list[tuple[int, int]]:
    return sorted(tuple(sorted(b)) for b in dataset.bonds)


def angle_list(bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """All bond angles (i, center, k) from the bond graph."""
    neighbours = defaultdict(list)
    for a, b in bonds:
        neighbours[a].append(b)
        neighbours[b].append(a)
    return [(n[i], c, n[k]) for c, n in neighbours.items() for i in range(len(n)) for k in range(i + 1, len(n))]


def bond_lengths(x: np.ndarray, bonds) -> np.ndarray:
    idx = np.asarray(bonds)
    return np.linalg.norm(x[:, idx[:, 0]] - x[:, idx[:, 1]], axis=-1)


def bond_angles(x: np.ndarray, angles) -> np.ndarray:
    idx = np.asarray(angles)
    u = x[:, idx[:, 0]] - x[:, idx[:, 1]]
    v = x[:, idx[:, 2]] - x[:, idx[:, 1]]
    cos = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def structure_summary(
    dataset,
    target: np.ndarray,
    sampled: np.ndarray,
    out_path: str,
    title: str = "",
    max_frames: Optional[int] = 200_000,
    seed: int = 0,
) -> dict:
    """Plot the pooled bond-length / bond-angle summary and return width ratios (sampled std / target std)."""
    rng = np.random.default_rng(seed)
    shape = (-1, *dataset.sample_shape)
    target = np.asarray(target).reshape(shape)
    sampled = np.asarray(sampled).reshape(shape)
    if max_frames is not None:
        if len(target) > max_frames:
            target = target[rng.choice(len(target), max_frames, replace=False)]
        if len(sampled) > max_frames:
            sampled = sampled[rng.choice(len(sampled), max_frames, replace=False)]

    names = dataset.atom_names
    bonds = bond_list(dataset)
    angles = angle_list(bonds)
    has_h = lambda atoms: any(_element(names[a]) == "H" for a in atoms)  # noqa: E731

    tb, sb = bond_lengths(target, bonds), bond_lengths(sampled, bonds)
    ta, sa = bond_angles(target, angles), bond_angles(sampled, angles)
    zb = (sb - tb.mean(0)) / tb.std(0)
    za = (sa - ta.mean(0)) / ta.std(0)
    zb_t = (tb - tb.mean(0)) / tb.std(0)
    za_t = (ta - ta.mean(0)) / ta.std(0)

    bond_h = np.array([has_h(b) for b in bonds])
    angle_h = np.array([has_h(a) for a in angles])
    bond_class = ["-".join(sorted(_element(names[a]) for a in b)) for b in bonds]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    bins = np.linspace(-8, 8, 161)
    metrics = {}
    for a, z, zt, is_h, label, unit in ((ax[0], zb, zb_t, bond_h, "bond lengths", "bonds"),
                                        (ax[1], za, za_t, angle_h, "bond angles", "angles")):
        a.hist(zt.ravel(), bins=bins, density=True, histtype="stepfilled", color="lightgray", label="Target (all)")
        groups = [("heavy atoms only", ~is_h, "tab:blue"), ("involving H", is_h, "tab:red")]
        for name, mask, c in groups:
            if mask.any():
                width = float(z[:, mask].std())
                shift = float(z[:, mask].mean())
                a.hist(np.clip(z[:, mask].ravel(), bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                       lw=2, color=c, label=f"Sampled, {name} (width x{width:.2f}, shift {shift:+.2f})")
                key = "bond" if "bond lengths" in label else "angle"
                metrics[f"{key}_width_{'H' if c == 'tab:red' else 'heavy'}"] = width
                metrics[f"{key}_shift_{'H' if c == 'tab:red' else 'heavy'}"] = shift
        a.set_xlabel("deviation from the target mean, in target std (z-score)")
        a.set_ylabel("density")
        a.set_title(f"All {label} pooled ({len(is_h)} {unit})")
        a.legend(fontsize=7)

    classes = sorted(set(bond_class))
    class_idx = {k: [i for i, c in enumerate(bond_class) if c == k] for k in classes}
    ratios = [float(sb[:, class_idx[k]].std(0).mean() / tb[:, class_idx[k]].std(0).mean()) for k in classes]
    counts = [bond_class.count(k) for k in classes]
    ax[2].bar(range(len(classes)), ratios, color=["tab:red" if "H" in k else "tab:blue" for k in classes])
    ax[2].axhline(1.0, color="gray", ls=":")
    ax[2].set_xticks(range(len(classes)), [f"{k}\n({n} bonds)" for k, n in zip(classes, counts)])
    ax[2].set_ylabel("bond-length width, sampled / target")
    ax[2].set_title("Width ratio per bond type (1 = target)")
    for i, r in enumerate(ratios):
        ax[2].text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
        metrics[f"bond_width_{classes[i]}"] = r
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return metrics


# Key backbone angles of alanine dipeptide, in full-atom (22-atom) indices:
# 0 CH3, 4 C, 5 O, 6 N, 8 CA, 10 CB, 14 C, 15 O, 16 N, 18 C. All heavy atoms, so they exist in CG10 as well.
KEY_BOND_ANGLES = {
    "C-N-CA": (4, 6, 8),
    "N-CA-C (tau)": (6, 8, 14),
    "CA-C-N": (8, 14, 16),
    "C-N-CH3": (14, 16, 18),
}
KEY_DIHEDRALS = {
    "omega1 (CH3-C-N-CA)": (0, 4, 6, 8),
    "omega2 (CA-C-N-CH3)": (8, 14, 16, 18),
    "chirality (N-CA-C-CB)": (6, 8, 14, 10),
}


def dihedrals(x: np.ndarray, quads) -> np.ndarray:
    idx = np.asarray(quads)
    p0, p1, p2, p3 = (x[:, idx[:, k]] for k in range(4))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1)))


def key_angle_summary(
    dataset, target: np.ndarray, sampled: np.ndarray, out_path: str, title: str = "",
    max_frames: Optional[int] = 200_000, seed: int = 0,
) -> dict:
    """Histograms and errors of the backbone bond angles, the two peptide torsions (omega) and CA chirality.

    Returns per angle the mean error (degrees), the width ratio (sampled std / target std), and for the
    chirality dihedral the fraction of mirror-image (D-alanine) frames. Only for ALDP datasets.
    """
    if not hasattr(dataset, "_atoms_to_keep"):
        return {}
    keep = {atom: k for k, atom in enumerate(dataset._atoms_to_keep)}
    rng = np.random.default_rng(seed)
    shape = (-1, *dataset.sample_shape)
    target = np.asarray(target).reshape(shape)
    sampled = np.asarray(sampled).reshape(shape)
    if max_frames is not None:
        if len(target) > max_frames:
            target = target[rng.choice(len(target), max_frames, replace=False)]
        if len(sampled) > max_frames:
            sampled = sampled[rng.choice(len(sampled), max_frames, replace=False)]

    panels = []
    for name, atoms in KEY_BOND_ANGLES.items():
        if all(a in keep for a in atoms):
            idx = [tuple(keep[a] for a in atoms)]
            panels.append((name, "angle", bond_angles(target, idx)[:, 0], bond_angles(sampled, idx)[:, 0]))
    for name, atoms in KEY_DIHEDRALS.items():
        if all(a in keep for a in atoms):
            idx = [tuple(keep[a] for a in atoms)]
            t, s = dihedrals(target, idx)[:, 0], dihedrals(sampled, idx)[:, 0]
            if name.startswith("omega"):  # trans peptide: centre on 180 deg instead of wrapping at +-180
                t, s = np.mod(t, 360.0), np.mod(s, 360.0)
            panels.append((name, "dihedral", t, s))

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    axes = axes.ravel()
    metrics, names, widths, shifts = {}, [], [], []
    for a, (name, kind, t, s) in zip(axes, panels):
        lo, hi = np.percentile(t, [0.05, 99.95])
        pad = 0.6 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 120)
        a.hist(t, bins=bins, density=True, histtype="stepfilled", color="lightgray", label="Target")
        a.hist(np.clip(s, bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=2, color="tab:orange",
               label="Sampled")
        shift = float(s.mean() - t.mean())
        width = float(s.std() / t.std())
        key = name.split(" ")[0]
        metrics[f"{key}_mean_error_deg"] = shift
        metrics[f"{key}_width_ratio"] = width
        text = f"mean error {shift:+.2f} deg\nwidth x{width:.2f}"
        if name.startswith("chirality"):
            wrong = float(np.mean(np.sign(s) != np.sign(np.median(t))))
            metrics["chirality_mirror_fraction"] = wrong
            text += f"\nmirror image (D): {100 * wrong:.2f}%"
        a.text(0.02, 0.97, text, transform=a.transAxes, va="top", fontsize=8)
        a.set_title(name, fontsize=10)
        a.set_xlabel("degrees")
        a.legend(fontsize=7, loc="upper right")
        names.append(key)
        widths.append(width)
        shifts.append(shift)

    a = axes[len(panels)]
    colors = ["tab:green" if w < 1.15 else "tab:orange" if w < 1.5 else "tab:red" for w in widths]
    a.bar(range(len(names)), widths, color=colors)
    a.axhline(1.0, color="gray", ls=":")
    a.set_ylim(0, 1.3 * max(widths + [1.0]))
    a.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    a.set_ylabel("width ratio (sampled / target)")
    a.set_title("Width ratio per key angle (1 = target)", fontsize=10)
    for i, (w, sh) in enumerate(zip(widths, shifts)):
        a.text(i, w, f"{w:.2f}\n{sh:+.1f} deg", ha="center", va="bottom", fontsize=7)
    for b in axes[len(panels) + 1:]:
        b.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return metrics



def plot_bond_density(target: np.ndarray, sampled: np.ndarray, bond_name: str, out_path: str, title: str = "",
                      max_points: int = 50_000, seed: int = 0) -> None:
    """Paper-style density of one bond length in Angstrom: smooth curves (Gaussian KDE) over the full range of
    both distributions. ``target`` and ``sampled`` are distances in nm."""
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(seed)
    t = 10.0 * np.asarray(target, dtype=float).ravel()  # nm -> Angstrom
    s = 10.0 * np.asarray(sampled, dtype=float).ravel()
    t = t if len(t) <= max_points else rng.choice(t, max_points, replace=False)
    s = s if len(s) <= max_points else rng.choice(s, max_points, replace=False)
    lo = min(np.percentile(t, 0.1), np.percentile(s, 0.1))
    hi = max(np.percentile(t, 99.9), np.percentile(s, 99.9))
    pad = 0.15 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 500)
    plt.figure(figsize=(4.2, 3.4))
    plt.plot(grid, gaussian_kde(t)(grid), color="black", lw=2.5, label="Reference")
    plt.plot(grid, gaussian_kde(s)(grid), color="tab:orange", lw=2, ls="--", label="Sampled")
    plt.xlim(grid[0], grid[-1])
    plt.ylim(bottom=0)
    plt.xlabel(f"{bond_name} bond length / Å")
    plt.ylabel("density / Å$^{-1}$")
    if title:
        plt.title(title, fontsize=9)
    plt.legend(fontsize=8)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
