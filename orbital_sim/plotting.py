"""Matplotlib visualisation: 3D orbits, ground tracks, element drift, attitude."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .constants import R_EARTH
from .groundtrack import ground_track

COLORS = ["#2a6fdb", "#e8590c", "#2b8a3e", "#ae3ec9", "#c92a2a", "#1098ad"]


def _earth(ax, radius=R_EARTH, alpha=0.25):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(radius * np.cos(u) * np.sin(v), radius * np.sin(u) * np.sin(v),
                    radius * np.cos(v), color="#4a90d9", alpha=alpha, linewidth=0, zorder=0)


def _equal_3d(ax, pts):
    m = np.max(np.abs(pts)) * 1.05
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))


def plot_orbits_3d(trajectories: Sequence, labels: Sequence[str] | None = None,
                   burns: Sequence[np.ndarray] = (), title="Orbits (ECI frame)", ax=None):
    """Plot one or more trajectories (objects with .r) around a 3D Earth."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
    _earth(ax)
    pts = [np.zeros((1, 3)) + R_EARTH]
    for k, tr in enumerate(trajectories):
        lab = labels[k] if labels else getattr(tr, "label", None)
        ax.plot(*tr.r.T, color=COLORS[k % len(COLORS)], lw=1.4, label=lab)
        pts.append(tr.r)
    for b in burns:
        ax.scatter(*b, color="k", marker="*", s=120, zorder=5)
    if burns:
        ax.scatter([], [], color="k", marker="*", s=80, label="burn")
    _equal_3d(ax, np.vstack(pts))
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(title)
    if labels or any(getattr(t, "label", "") for t in trajectories):
        ax.legend(loc="upper left")
    return ax


def plot_ground_track(traj, title="Ground track", ax=None, label=None, color=None):
    """Latitude/longitude of the sub-satellite point on a plain map grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.5))
    lat, lon, _ = ground_track(traj.r, traj.t)
    lon_plot = lon.copy()
    jumps = np.nonzero(np.abs(np.diff(lon)) > 180)[0]
    lon_plot = np.insert(lon_plot, jumps + 1, np.nan)
    lat_plot = np.insert(lat, jumps + 1, np.nan)
    ax.plot(lon_plot, lat_plot, lw=1.2, color=color or COLORS[0], label=label)
    ax.scatter(lon[0], lat[0], color="#2b8a3e", zorder=5, label="start" if label is None else None)
    ax.scatter(lon[-1], lat[-1], color="#c92a2a", zorder=5, label="end" if label is None else None)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-90, 91, 30))
    ax.grid(alpha=0.3)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return ax


def plot_ground_track_3d(traj, title="Ground track on rotating Earth (ECEF)", ax=None):
    """Sub-satellite point drawn on the surface of a 3D Earth (Earth-fixed frame)."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
    _earth(ax, alpha=0.35)
    _, _, r_ecef = ground_track(traj.r, traj.t)
    surface = r_ecef / np.linalg.norm(r_ecef, axis=1)[:, None] * R_EARTH * 1.005
    ax.plot(*surface.T, color=COLORS[1], lw=1.2, label="ground track")
    ax.plot(*r_ecef.T, color=COLORS[0], lw=0.8, alpha=0.6, label="orbit (ECEF)")
    _equal_3d(ax, r_ecef)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def plot_attitude(res, title="Reaction-wheel PID attitude response", settle_band_deg=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    e = res.euler_error_deg()
    for k, name in enumerate(["roll", "pitch", "yaw"]):
        axs[0, 0].plot(res.t, e[:, k], color=COLORS[k], label=name)
        axs[0, 1].plot(res.t, np.degrees(res.w[:, k]), color=COLORS[k], label=name)
        axs[1, 0].plot(res.t, res.torque[:, k] * 1000, color=COLORS[k], label=name)
        axs[1, 1].plot(res.t, res.h_wheel[:, k], color=COLORS[k], label=name)
    ts = res.settling_time(settle_band_deg)
    for ax in axs.flat:
        ax.axvline(ts, color="k", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend()
    axs[0, 0].set_title(f"Attitude error [deg]  (2% settling: {ts:.2f} s)")
    axs[0, 1].set_title("Body rate [deg/s]")
    axs[1, 0].set_title("Control torque [mN m]")
    axs[1, 1].set_title("Wheel momentum [N m s]")
    axs[1, 0].set_xlabel("time [s]"); axs[1, 1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save(fig_or_ax, path, dpi=130):
    fig = fig_or_ax if isinstance(fig_or_ax, plt.Figure) else fig_or_ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
