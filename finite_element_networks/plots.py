from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as pp
import numpy as np
import PIL
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation
from torchtyping import TensorType

from .data import STBatch
from .fen import FEN, FENDomainInfo, FreeFormTerm, SystemState, TransportTerm


def render_figure(fig: pp.Figure) -> PIL.Image:
    """Render a matplotlib figure into a Pillow image."""
    buf = BytesIO()
    fig.savefig(buf, **{"format": "rgba"})
    return PIL.Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), buf.getbuffer(), "raw", "RGBA", 0, 1
    )


def render_worker(args):
    render_frame, frame = args
    fig = render_frame(*frame)
    try:
        return render_figure(fig)
    finally:
        pp.close(fig)


def render_animation(render_frame, frames, *, duration: int = 200) -> bytes:
    """Create an animation by rendering each frame in parallel.

    The usual matplotlib.animation style of creating animations renders them sequentially
    which can be intolerably slow if the individual frames take significant rendering
    time, e.g. meshes or stream plots.

    Arguments
    ---------
    duration
        How long to show each frame in milliseconds

    Returns
    -------
    The rendered animation bytes that you can render or write to a file.
    """

    with ProcessPoolExecutor() as pool:
        imgs = list(pool.map(render_worker, zip([render_frame] * len(frames), frames)))

    with BytesIO() as buffer:
        imgs[0].save(
            buffer,
            format="WebP",
            append_images=imgs[1:],
            save_all=True,
            loop=0,
            duration=duration,
            # Trade-off between speed and quality
            method=3,
            quality=95,
        )

        return buffer.getvalue()


def plot_data_and_prediction(u, u_hat, t, mesh, vmin, vmax):
    w, h = mesh.p.ptp(axis=1)
    aspect_ratio = w / h

    n_features = u.shape[-1]
    fig = pp.figure(
        constrained_layout=True, figsize=(9, 1 + 4 / aspect_ratio * n_features)
    )
    gs = fig.add_gridspec(
        n_features, 3, width_ratios=[4, 4, 0.5], height_ratios=[1] * n_features
    )

    ax_gt = [fig.add_subplot(gs[j, 0]) for j in range(n_features)]
    ax_pd = [fig.add_subplot(gs[j, 1]) for j in range(n_features)]
    cax = [fig.add_subplot(gs[j, 2]) for j in range(n_features)]

    for ax in ax_gt + ax_pd:
        ax.set_aspect(1.0)
        ax.set_axis_off()

    title = fig.suptitle(f"$t$ = {t:.2f}")
    ax_gt[0].set_title("GT")
    ax_pd[0].set_title("Prediction")

    tri = Triangulation(*mesh.p, mesh.t.T)
    for j, ax in enumerate(ax_gt):
        ax.set_ylabel(f"Feature {j}")
        field_gt = ax.tripcolor(tri, u[:, j], vmin=vmin[j], vmax=vmax[j])
        fig.colorbar(field_gt, cax=cax[j])
    for j, ax in enumerate(ax_pd):
        ax.tripcolor(tri, u_hat[:, j], vmin=vmin[j], vmax=vmax[j])

    return fig


def animate_data_and_predictions(
    data: STBatch,
    u_hat: TensorType["batch", "time", "node", "feature"],
    interval: int = 200,
):
    t = data.target_t.cpu().detach().numpy()
    u = data.target_u.cpu().detach().numpy()
    u_hat = u_hat.cpu().detach().numpy()

    t = t[0]
    u = u[0]
    u_hat = u_hat[0]

    assert len(t) > 0
    assert u.ndim == 3
    assert u_hat.ndim == 3

    # Compute feature scale here so that it is constant across frames
    vmin = u.min(axis=(0, 1))
    vmax = u.max(axis=(0, 1))
    mesh = data.domain.basis.mesh

    frames = [(u[i], u_hat[i], t[i], mesh, vmin, vmax) for i in range(len(t))]
    return render_animation(plot_data_and_prediction, frames, duration=interval)


def plot_flow_fields(u_hat, cell_ff, t, mesh, normalize: bool, vmin, vmax):
    w, h = mesh.p.ptp(axis=1)
    aspect_ratio = w / h

    n_features = cell_ff.shape[-2]
    fig = pp.figure(
        constrained_layout=True, figsize=(4, 1 + 4 / aspect_ratio * n_features)
    )
    gs = fig.add_gridspec(
        n_features, 1, width_ratios=[4], height_ratios=[1] * n_features
    )

    axes = [fig.add_subplot(gs[j, 0]) for j in range(n_features)]

    for ax in axes:
        ax.set_aspect(1.0)
        ax.set_axis_off()

    title = fig.suptitle(f"$t$ = {t:.2f}")
    if normalize:
        axes[0].set_title("Normalized Flow Fields")
    else:
        axes[0].set_title("Flow Fields")

    scale = None if normalize else 1.0
    x, y = mesh.p[:, mesh.t].mean(axis=1)
    tri = Triangulation(*mesh.p, mesh.t.T)
    for j, ax in enumerate(axes):
        ax.set_ylabel(f"Feature {j}")
        u, v = cell_ff[:, j].T
        ax.quiver(
            x,
            y,
            u,
            v,
            angles="xy",
            scale=scale,
            scale_units="xy",
            zorder=2,
            width=0.007,
            linewidth=0.5,
            edgecolors="1.0",
        )
        ax.tripcolor(tri, u_hat[:, j], zorder=1, vmin=vmin[j], vmax=vmax[j])

    return fig


@torch.no_grad()
def animate_flow_fields(
    data: STBatch,
    u_hat: TensorType["batch", "time", "node", "feature"],
    model: FEN,
    *,
    interval: int = 200,
    normalize: bool = True,
    feature_idx: Optional[int] = None,
):
    tt = model.dynamics.transport_terms[0]

    batch_idx = 0
    t = data.time_encoder.encode(data.target_t.T)
    n_steps = t.shape[0]
    ffs = []
    for step in range(n_steps):
        state = SystemState(data.domain_info, t[step], u_hat[:, step])
        ffs.append(tt.estimate_flow_field(state).cpu().numpy()[batch_idx])
    ffs = np.stack(ffs)

    all_u = torch.flatten(u_hat, end_dim=-2)
    vmin, vmax = torch.amin(all_u, dim=0), torch.amax(all_u, dim=0)

    if feature_idx is not None:
        u_hat = u_hat[..., feature_idx : feature_idx + 1]
        vmin = vmin[feature_idx : feature_idx + 1]
        vmax = vmax[feature_idx : feature_idx + 1]
        ffs = ffs[..., feature_idx : feature_idx + 1, :]

    u_hat = u_hat.cpu().numpy()
    vmin = vmin.cpu().numpy()
    vmax = vmax.cpu().numpy()
    mesh = data.domain.basis.mesh
    target_t = data.target_t[batch_idx].cpu().numpy()
    frames = [
        (u_hat[batch_idx, i], ffs[i], target_t[i], mesh, normalize, vmin, vmax)
        for i in range(n_steps)
    ]
    return render_animation(plot_flow_fields, frames, duration=interval)


def plot_disentanglement(labels, activations, t, mesh, vmin, vmax):
    w, h = mesh.p.ptp(axis=1)
    aspect_ratio = w / h

    n_terms = len(labels)
    n_features = len(vmin)
    fig = pp.figure(
        constrained_layout=True,
        figsize=(1 + 4 * n_terms, 1 + 4 / aspect_ratio * n_features),
    )
    gs = fig.add_gridspec(
        n_features,
        1 + n_terms,
        width_ratios=[4] * n_terms + [0.5],
        height_ratios=[1] * n_features,
    )

    axes = [
        [fig.add_subplot(gs[j, i]) for j in range(n_features)] for i in range(n_terms)
    ]
    cax = [fig.add_subplot(gs[j, n_terms]) for j in range(n_features)]

    title = fig.suptitle(f"$t$ = {t:.2f}")
    for label, term_axes in zip(labels, axes):
        term_axes[0].set_title(label)
        for ax in term_axes:
            ax.set_aspect(1.0)
            ax.set_axis_off()

    tri = Triangulation(*mesh.p, mesh.t.T)
    for j in range(n_features):
        norm = TwoSlopeNorm(0.0, vmin=min(vmin[j], -1e-8), vmax=max(vmax[j], 1e-8))
        for i in range(n_terms):
            ax = axes[i][j]
            ax.tripcolor(tri, activations[i][:, j], cmap="coolwarm", norm=norm)
            field = ax.get_children()[0]
            if i == 0:
                ax.set_ylabel(f"Feature {j}")
                fig.colorbar(field, cax=cax[j])

    return fig


@torch.no_grad()
def animate_disentanglement(
    data: STBatch,
    u_hat: TensorType["batch", "time", "node", "feature"],
    model: FEN,
    interval: int = 200,
    feature_idx: Optional[int] = None,
):
    dynamics = model.dynamics
    t = data.time_encoder.encode(data.target_t.T)
    n_steps = t.shape[0]
    activations = []
    for term in dynamics.terms:
        step_activations = [
            dynamics._send_msgs(
                term(SystemState(data.domain_info, t[step], u_hat[:, step])),
                data.domain_info,
            )
            for step in range(n_steps)
        ]
        activations.append(torch.stack(step_activations, dim=1)[0])

    all_activations = torch.cat(
        [torch.flatten(act, end_dim=-2) for act in activations], dim=0
    )
    vmin, vmax = torch.amin(all_activations, dim=0), torch.amax(all_activations, dim=0)

    if feature_idx is not None:
        vmin = vmin[feature_idx : feature_idx + 1]
        vmax = vmax[feature_idx : feature_idx + 1]
        activations = [act[..., feature_idx : feature_idx + 1] for act in activations]

    mesh = data.domain.basis.mesh
    labels = {FreeFormTerm: "Free-Form", TransportTerm: "Transport"}
    frames = [
        (
            [labels[type(term)] for term in dynamics.terms],
            [act[i].cpu().numpy() for act in activations],
            data.target_t[0, i].cpu().numpy(),
            mesh,
            vmin.cpu().numpy(),
            vmax.cpu().numpy(),
        )
        for i in range(n_steps)
    ]
    return render_animation(plot_disentanglement, frames, duration=interval)
