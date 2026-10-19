from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from matplotlib.transforms import Bbox
import matplotlib as mpl


def full_extent(ax, pad=0.0, title=True, xlabel=True, ylabel=True, xticks=True, yticks=True):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles.
    https://stackoverflow.com/a/26432947/4417954
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = [ax]
    if title:
        items += [ax.title]
    if xlabel:
        items += [ax.xaxis.label]
    if ylabel:
        items += [ax.yaxis.label]
    if xticks:
        items += ax.get_xticklabels()
    if yticks:
        items += ax.get_yticklabels()
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)


def save_parts_of_figure(ax, filename, title=True, xlabel=True, ylabel=True, xticks=True, yticks=True, **kwargs):
    extent = full_extent(ax, title=title, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks).transformed(
        plt.gcf().dpi_scale_trans.inverted()
    )
    plt.savefig(filename, bbox_inches=extent, **kwargs)


def rasterize_contour(contour):
    """Rasterize a contour plot to avoid jagged edges."""
    for c in contour.collections:
        c.set_rasterized(True)


def plot_2d(
    samples_x,
    samples_y,
    title="Histogram",
    range=None,
    highlight=None,
    bins=60,
    vmin=None,
    vmax=None,
    cmap=None,
    free_energy_bar=False,
    fig=None,
    ax=None,
    **kwargs,
):
    assert (fig is None and ax is None) or (fig is not None and ax is not None), (
        "Both fig and ax must be either both None or both provided."
    )
    if fig is None:
        fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)
    H, _, _, _ = ax.hist2d(
        samples_x,
        samples_y,
        bins=bins,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        range=range,
        rasterized=True,
        cmap=cmap,
        **kwargs,
    )

    if vmin is None:
        vmin = H[H > 0].min() if jnp.any(H > 0) else vmin
    if vmax is None:
        vmax = H.max()

    if free_energy_bar:
        # we assume a positive energy
        e_min, e_max = 0, jnp.log(vmax / H.sum()) - jnp.log(vmin / H.sum())
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=e_min, vmax=e_max), cmap=mpl.colormaps[cmap].reversed()
        )

        # attach the colorbar to the axes
        cbar_obj = fig.colorbar(sm, ax=ax, extend="max")
        cbar_obj.set_label(r"Energy / $k_BT$")

    if range is not None:
        plt.xlim(*range[0])
        plt.ylim(*range[1])

    plt.gca().set_box_aspect(1)

    if highlight is not None:
        # plot a star for each highlight
        for h in highlight:
            plt.plot(samples_x[h], samples_y[h], "*", markersize=8, alpha=0.8)


def phi_psi(phi, psi, title="Histogram", highlight=None, bins=60, vmin=None, vmax=None, **kwargs):
    range = [-jnp.pi, jnp.pi]

    plot_2d(phi, psi, title, [range, range], highlight, bins, vmin, vmax, **kwargs)

    plt.xticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    plt.yticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )

    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\psi$")

    return phi, psi


def _gaussian_smooth(values, dx: float, sigma: float, periodic: bool):
    """Gaussian filter of a curve on a uniform grid (sigma in x units); wraps around if periodic."""
    import numpy as onp

    values = onp.asarray(values)
    half = max(1, int(onp.ceil(3 * sigma / dx)))
    kernel = onp.exp(-0.5 * (onp.arange(-half, half + 1) * dx / sigma) ** 2)
    kernel /= kernel.sum()
    if periodic:  # grid includes both endpoints of the period; filter the unique points and re-close it
        core = values[:-1]
        padded = onp.concatenate([core[-half:], core, core[:half]])
        smoothed = onp.convolve(padded, kernel, mode="valid")
        return jnp.asarray(onp.append(smoothed, smoothed[0]))
    padded = onp.pad(values, half, mode="edge")
    return jnp.asarray(onp.convolve(padded, kernel, mode="valid"))


def plot_fes(
    samples: jnp.ndarray,
    kBT: float,
    grid: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    bw_method: float = 0.05,
    linewidth: float = 3,
    smooth_above_fes: Optional[float] = None,
    smooth_width: float = 4.0,
    fes_smooth_sigma: Optional[float] = None,
    periodic: bool = False,
    cap_mode: str = "tanh",
    max_above: float = 9.0,
    high_fes_smooth_sigma: Optional[float] = None,
    high_fes_from: Optional[float] = None,
    high_fes_blend: float = 5.0,
    post_cap_smooth_sigma: Optional[float] = None,
    *args,
    **kwargs,
):
    """Plot the free energy -kBT log p of ``samples`` estimated with a Gaussian KDE.

    smooth_above_fes: values above it are compressed smoothly as c + w tanh((F - c) / w), with c the
        threshold and w = ``smooth_width``. The curve keeps slope 1 at c (no kink) and never exceeds c + w.
    high_fes_smooth_sigma / high_fes_from / high_fes_blend: optional extra Gaussian filter (sigma in x units) applied
        only where the curve is above high_fes_from, blended in smoothly over the next high_fes_blend units; removes
        the spikes few samples create in sparsely sampled regions and leaves everything below the threshold unchanged.
    fes_smooth_sigma: optional Gaussian filter of the free-energy curve (in x units, e.g. rad) that removes
        the sampling-noise jaggedness in sparsely sampled regions; applied before the compression.
    cap_mode: "tanh" (soft compression described above), "clip" (hard cut at ``smooth_above_fes``) or
        "softmin" (cut at ``smooth_above_fes`` with corners rounded over ~``smooth_width``) or "log"
        (logarithmic compression above the threshold, bounded below ``smooth_above_fes + max_above``).
    post_cap_smooth_sigma: optional Gaussian filter applied after the cap, which rounds the corners where
        the curve meets the cap (purely visual; used with cap_mode="clip").
    """
    if grid is None:
        grid = jnp.linspace(samples.min(), samples.max(), 100)

    # now we filter out values outside the range
    if (
        (samples >= grid.min()) & (samples <= grid.max())
    ).sum() > 0:  # only filter if there are values that will be still inside the range
        samples = samples[(samples >= grid.min()) & (samples <= grid.max())]
    else:
        print("Warning: No values will be inside the range after filtering, not filtering")

    density = gaussian_kde(samples, bw_method, weights).pdf(grid)
    fes = -kBT * jnp.log(jnp.clip(density, a_min=jnp.finfo(density.dtype).tiny))

    if fes_smooth_sigma is not None:
        fes = _gaussian_smooth(fes, float(grid[1] - grid[0]), fes_smooth_sigma, periodic)

    fes -= fes.min()

    if high_fes_smooth_sigma is not None and high_fes_from is not None:
        strong = _gaussian_smooth(fes, float(grid[1] - grid[0]), high_fes_smooth_sigma, periodic)
        u = jnp.clip((jnp.minimum(fes, strong) - high_fes_from) / high_fes_blend, 0.0, 1.0)
        weight = u * u * (3.0 - 2.0 * u)  # smoothstep: no kink where the blending starts or ends
        fes = (1.0 - weight) * fes + weight * strong

    if smooth_above_fes is not None:
        if cap_mode == "clip":
            fes = jnp.minimum(fes, smooth_above_fes)
        elif cap_mode == "log":
            # unchanged up to the threshold, above it c + w log(1 + (F - c) / w): slope 1 at c, keeps rising slowly
            # (rounded tops instead of flat plateaus); a final tanh bound keeps it below c + max_above
            excess = jnp.clip(fes - smooth_above_fes, a_min=0.0)
            compressed = smooth_width * jnp.log1p(excess / smooth_width)
            compressed = max_above * jnp.tanh(compressed / max_above)
            fes = jnp.where(fes > smooth_above_fes, smooth_above_fes + compressed, fes)
        elif cap_mode == "softmin":
            # smooth minimum of F and the cap: equals the cap on plateaus, follows F well below it (error
            # k * exp(-(cap - F) / k)), rounded corners in between; k = smooth_width
            k = smooth_width
            fes = -k * jnp.logaddexp(-fes / k, -smooth_above_fes / k)
        elif cap_mode == "tanh":
            excess = jnp.clip(fes - smooth_above_fes, a_min=0.0)
            fes = jnp.where(fes > smooth_above_fes, smooth_above_fes + smooth_width * jnp.tanh(excess / smooth_width), fes)
        else:
            raise ValueError(f"Unknown cap_mode={cap_mode!r}; use 'tanh', 'clip', 'softmin' or 'log'.")

    if post_cap_smooth_sigma is not None:
        # Round the corners only around the capped stretches: the weight is the capped-region mask blurred along x,
        # so it varies smoothly in x and is ~0 a few sigma away from the cap (basins keep their computed values).
        dx = float(grid[1] - grid[0])
        smoothed = _gaussian_smooth(fes, dx, post_cap_smooth_sigma, periodic)
        top = smooth_above_fes if smooth_above_fes is not None else float(fes.max())
        capped = (fes >= top - 1e-6).astype(fes.dtype)
        weight = jnp.clip(2.0 * _gaussian_smooth(capped, dx, post_cap_smooth_sigma, periodic), 0.0, 1.0)
        fes = (1.0 - weight) * fes + weight * smoothed

    plt.plot(grid, fes, linewidth=linewidth, *args, **kwargs)
    plt.xlim(grid.min(), grid.max())
    plt.ylabel(r"Energy / $k_BT$")

    return grid, fes


def plot_fes_angles(
    angles: jnp.ndarray,
    kBT: float,
    weights: Optional[jnp.ndarray] = None,
    bw_method: float = 0.05,
    n_grid: int = 100,
    *args,
    **kwargs,
):
    grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    # we add the periodic boundary conditions so that the plot is continuous
    extended_angles = jnp.concatenate([angles, angles + 2 * jnp.pi, angles - 2 * jnp.pi])
    # extended_angles = angles
    grid, fes = plot_fes(extended_angles, kBT, grid, weights, bw_method, *args, periodic=True, **kwargs)
    plt.xticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    return grid, fes


def plot_potential_1d(potential, potential_net, x_min, x_max):
    xx = jnp.linspace(x_min, x_max, 1000)
    xx = xx.reshape(-1, 1)
    ground_truth_potential = potential(xx)
    plt.title("Potential")
    ax = plt.plot(xx, ground_truth_potential, label="Ground Truth")
    if potential_net is not None:
        model_potential = potential_net(xx)
        plt.plot(xx, model_potential - model_potential.min() + ground_truth_potential.min(), label="Model")
    plt.xlabel(r"$x$")
    plt.legend()

    return ax


def plot_potential_2d(potential, density, potential_net, xlim, ylim):
    n = 100j

    (x_min, x_max), (y_min, y_max) = xlim, ylim
    xx, yy = jnp.mgrid[x_min:x_max:n, y_min:y_max:n]
    positions = jnp.vstack([xx.ravel(), yy.ravel()]).T

    ground_truth_density = density(positions)
    ground_truth_potential = potential(positions)

    # Ground Truth
    ax0 = plt.subplot(2, 2, 1)
    plt.title(r"Density$\propto\exp (-U(x))$ (Truth)")
    contour = plt.contourf(xx, yy, ground_truth_density.reshape(xx.shape), 100)
    rasterize_contour(contour)
    plt.colorbar()

    ax1 = plt.subplot(2, 2, 3 if potential_net is not None else 2)
    plt.title(r"Potential $\propto U(x)$ (Truth)")
    contour = plt.contourf(xx, yy, ground_truth_potential.reshape(xx.shape), 100)
    rasterize_contour(contour)
    plt.colorbar()

    if potential_net is not None:
        model_potential = potential_net(positions)
        model_potential = model_potential - model_potential.min() + ground_truth_potential.min()
        model_density = jnp.exp(-model_potential)

        # Model
        ax3 = plt.subplot(2, 2, 2)
        plt.title(r"Density$\propto\exp (-U(x))$ (Model)")
        contour = plt.contourf(xx, yy, model_density.reshape(xx.shape), 100)
        rasterize_contour(contour)
        plt.colorbar()

        ax4 = plt.subplot(2, 2, 4)
        plt.title(r"Potential $\propto U(x)$ (Model)")
        contour = plt.contourf(xx, yy, model_potential.reshape(xx.shape), 100)
        rasterize_contour(contour)
        plt.colorbar()
    else:
        return ax0, ax1, None, None

    return ax0, ax1, ax3, ax4


def plot_force_2d(force_net, forces, xlim, ylim, equal_scaling=True, n=100j, every=5):
    """
    Args:
        equal_scaling (bool, optional): If we use equal scaling, the arrows will show comparable lengths. Defaults to True.

    Returns:
        _type_: _description_
    """
    (x_min, x_max), (y_min, y_max) = xlim, ylim

    def get_scale(xx, yy, uu, vv):
        """Get autoscale value by plotting to do-nothing backend."""
        backend = plt.matplotlib.get_backend()
        plt.matplotlib.use("template")
        Q = plt.quiver(xx, yy, uu, vv, scale=None)
        plt.matplotlib.use(backend)
        Q._init()
        return Q.scale

    xx, yy = jnp.mgrid[x_min:x_max:n, y_min:y_max:n]
    positions = jnp.vstack([xx.ravel(), yy.ravel()]).T
    u, v = forces(positions).T

    if equal_scaling:
        scale = get_scale(
            xx[::every, ::every],
            yy[::every, ::every],
            u.reshape(xx.shape)[::every, ::every],
            v.reshape(yy.shape)[::every, ::every],
        )
    else:
        scale = None

    plt.quiver(
        xx[::every, ::every],
        yy[::every, ::every],
        u.reshape(xx.shape)[::every, ::every],
        v.reshape(yy.shape)[::every, ::every],
        label="Ground Truth",
        color="blue",
        scale=scale,
    )

    u, v = force_net(positions).T
    plt.quiver(
        xx[::every, ::every],
        yy[::every, ::every],
        u.reshape(xx.shape)[::every, ::every],
        v.reshape(yy.shape)[::every, ::every],
        label="Model",
        color="orange",
        scale=scale,
    )
    plt.legend()
