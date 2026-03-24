import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from typing import Optional, Tuple
from scipy.signal import medfilt


def plot(
    spots: pd.DataFrame,
    edges: pd.DataFrame,
    conversion_factors: Tuple[float],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    img: np.ndarray = None,
    img_id: int = 0,
    draw_mean_hline: bool = False,
    figsize: tuple[int, int] = (7, 6),
    legend: bool = True,
    line_tracks_kwargs: dict = {"lw": 0.9, "alpha": 1},
    scatter_spots_kwargs: dict = {"alpha": 0.8, "s": 0.7, "marker": ","},
    tracks: Optional[list[int]] = [],
    xlim: Optional[list[int]] = [],
    ylim: Optional[list[int]] = [],
    # speed coloring (lines)
    color_code: str = "speed",
    cmap: str = "brg",
    logscale: bool = False,
    speed_clip_percent: tuple[int, int] = (1, 99),
    equal_axes: bool = True,
    invert_y: bool = False,
    colorbar: bool = True,
    speed_units: str = "m/s",
    ax_units: str = "mm",
    set_rasterized: bool = True,
    # quiver (arrow) overlay options
    arrows: bool = True,
    arrows_every: int = 4,
    arrows_scale_units: str = "xy",
    arrows_scale: float = 0.5,  # larger -> shorter arrows
    arrows_width: float = 0.004,
    arrows_headwidth: float = 4.0,
    arrows_headlength: float = 5.0,
    arrows_headaxislength: float = 3.5,
    arrows_per_track_norm: bool = False,
    arrow_alpha: float = 1.0,
    arrows_equal_size: bool = True,
    arrows_rel_len: float = 0.05,
    arrows_clip_percent: tuple[int, int] = (5, 95),
    arrows_colorbar: bool = False,
    # q/m (charge-to-mass) coloring parameters
    gap: float = 1.7e-2,  # [m] plate separation
    phi: float = 2000.0,  # [V] applied potential (mesh-substrate)
    gravity: float = 9.81,  # [m/s^2] gravity
    lines_raw: bool = True,
    clip_speed:bool = False,
) -> Tuple[plt.Figure, plt.Axes]:

    if len(tracks) > 0:
        edges = edges[edges.TRACK_ID.isin(tracks)]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if img is not None:
        _, x_um_per_px, y_um_per_px = conversion_factors
        ny, nx = img.shape[:2]

        ax.set_aspect("auto")

        ny, nx = img.shape[:2]

        im = ax.imshow(
            img,
            origin="upper",
            extent=[0, nx * x_um_per_px, 0, ny * y_um_per_px],
            aspect="auto",  # <-- important
            zorder=0,
            cmap="grey",
        )

    if xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.autoscale(False)  # or: ax.set_autoscale_on(False)
        ax.set_xmargin(0)
        ax.set_ymargin(0)

        im.sticky_edges.x[:] = []
        im.sticky_edges.y[:] = []
    #     # Spots
    # g = spots.sort_values("t").groupby("TRACK_ID", sort=False)

    # idx = g["t"].idxmin()
    # out = spots.loc[idx, ["TRACK_ID", "x", "y", "t"]].reset_index(drop=True)
    # out = out[out.TRACK_ID.isin(tracks)]
    # ax.scatter(
    # out["x"].to_numpy(),
    # out["y"].to_numpy(),
    # rasterized=True,
    # **scatter_spots_kwargs,
    # )

    # ---------- Build segments + speeds (prefers vx,vy if present) ----------
    all_segs = []
    all_speed = []
    per_track_data = []  # each: {"mid": (M,2), "vec": (M,2), "speed": (M,)}

    if lines_raw:
        cols = ["x_raw", "y_raw", "t"]
    else:
        cols = ["x", "y", "t"]

    # Ensure time order if t exists
    edges = edges.sort_values(["TRACK_ID", "t"])
    all_qm = []
    for _, g in edges.groupby("TRACK_ID", sort=False):
        arr = g[cols].to_numpy(dtype=float)
        xy = arr[:, :2]

        # --- q/m per *point* (then averaged to segments midpoints) ---
        # Need acceleration ay and direction angle theta from velocity.
        # 1) velocity (prefer columns; else finite diff)

        vx = g["vx"].to_numpy(dtype=float)
        vy = g["vy"].to_numpy(dtype=float)

        ay = g["ay"].to_numpy(dtype=float) * 1e-3

        # 3) theta from velocity direction (rad); use arctan2 for proper quadrant
        theta = np.arctan2(vy, vx)

        # 4) Ey field (uniform): Ey = -Phi/h
        Ey = -phi / gap  # [V/m] = [N/C]

        # 5) q/m from your relation: q/m = (ay + g*cos(theta)) / Ey
        qm_point = (ay + gravity * np.cos(theta)) / Ey  # [C/kg]

        # match segments (N-1) by midpoint averaging
        qm_seg = 0.5 * (qm_point[:-1] + qm_point[1:])

        if xy.shape[0] < 2:
            continue

        segs = np.stack([xy[:-1], xy[1:]], axis=1)  # (N-1, 2, 2)
        mids = 0.5 * (segs[:, 0, :] + segs[:, 1, :])  # (N-1, 2)

        vxy = g[["vx", "vy"]].to_numpy(dtype=float) / 1e3  # m/s

        sp = np.hypot(vxy[:, 0], vxy[:, 1])  # (N,)
        seg_speed = 0.5 * (sp[:-1] + sp[1:])  # (N-1,)
        dvec = 0.5 * (vxy[:-1, :] + vxy[1:, :])  # (N-1, 2)

        valid = np.isfinite(seg_speed) & np.all(np.isfinite(dvec), axis=1)

        segs_v = segs[valid]
        mids_v = mids[valid]
        speed_v = seg_speed[valid]
        vec_v = dvec[valid]

        if speed_v.size == 0:
            continue
        qm_seg_v = qm_seg[valid]
        all_qm.append(qm_seg_v)

        all_segs.append(segs_v)
        all_speed.append(speed_v)
        per_track_data.append(
            {"mid": mids_v, "vec": vec_v, "speed": speed_v, "qm": qm_seg[valid]}
        )

    if not all_segs:
        segs_plain = []
        for _, g in edges.groupby("TRACK_ID", sort=False):
            xy = g[["x_raw", "y_raw"]].to_numpy()
            if xy.shape[0] >= 2:
                segs_plain.append(xy)
        if segs_plain:
            lc_plain = LineCollection(
                segs_plain, antialiased=True, **line_tracks_kwargs
            )
            lc.set_capstyle("round")
            lc.set_joinstyle("round")
            if set_rasterized:
                lc_plain.set_rasterized(True)
            else:
                lc.set_rasterized(False)
                ax.set_rasterization_zorder(-1)
            ax.add_collection(lc_plain)
        _finalize_axes(
            ax, legend, draw_mean_hline, spots, equal_axes, invert_y, ax_units
        )
        return (fig, ax)

    segs_concat = np.concatenate(all_segs, axis=0)  # (M, 2, 2)
    speed_concat = np.concatenate(all_speed, axis=0)  # (M,)
    qm_concat = np.concatenate(all_qm, axis=0) * 1e3 if all_qm else None
    # qm_concat = medfilt(qm_concat, kernel_size=3)

    # ---------- Lines: color by speed (global norm) ----------
    # ---------- Choose scalar to color by + normalization ----------
    scalar_vals = None
    cbar_label = ""
    norm = None

    if color_code == "speed":
        scalar_vals = speed_concat
        if clip_speed:
            lo, hi = np.percentile(scalar_vals, speed_clip_percent)
        else:
            lo = np.min(scalar_vals)
            hi = np.max(scalar_vals)
        if hi <= lo:
            hi = lo * 1.0001

        norm = LogNorm(vmin=lo, vmax=hi) if logscale else Normalize(vmin=lo, vmax=hi)
        cbar_label = f"Speed ({speed_units})"

    elif color_code in {"qm", "charge", "q_over_m"}:
        if qm_concat is None or qm_concat.size == 0:
            raise ValueError("q/m requested but could not compute from provided data.")
        scalar_vals = qm_concat
        norm = Normalize(vmin=min(scalar_vals), vmax=max(scalar_vals))
        cbar_label = r"$q/m$ (mC/kg)"

    else:
        raise ValueError(f"Unknown color_code='{color_code}'. Use 'speed' or 'qm'.")

    lc = LineCollection(
        segs_concat,
        cmap=cmap,
        norm=norm,
        antialiased=True,
        **line_tracks_kwargs,
    )
    lc.set_array(scalar_vals)

    if set_rasterized:
        lc.set_rasterized(True)
    else:
        lc.set_rasterized(False)
        ax.set_rasterization_zorder(-1)
    lc.set_capstyle("round")
    lc.set_joinstyle("round")
    ax.add_collection(lc)

    if arrows and per_track_data:
        mids = np.concatenate([td["mid"] for td in per_track_data], axis=0)
        vecs = np.concatenate([td["vec"] for td in per_track_data], axis=0)
        spds = np.concatenate([td["speed"] for td in per_track_data], axis=0)
        qms = np.concatenate([td["qm"] for td in per_track_data], axis=0) * 1e3
        # qms = medfilt(qms, kernel_size=3)

        if arrows_every > 1:
            idx = np.arange(mids.shape[0])[::arrows_every]
            mids, vecs, spds, qms = mids[idx], vecs[idx], spds[idx], qms[idx]

        # === choose the scalar for arrows to match the line ===
        if color_code in {"qm", "charge", "q_over_m"}:
            cvals = qms
        else:  # "speed"
            cvals = spds

        # use the EXACT SAME norm & cmap as the LineCollection
        norm_a = norm

        # === force equal-length arrows (data units) if requested ===
        if arrows_equal_size:
            mag = np.linalg.norm(vecs, axis=1, keepdims=True)
            mag[mag == 0] = 1.0
            udir = vecs / mag

            # ensure limits exist before using get_xlim()
            fig.canvas.draw_idle()
            xr = ax.get_xlim()
            xspan = xr[1] - xr[0]
            if not np.isfinite(xspan) or xspan <= 0:
                xspan = (mids[:, 0].max() - mids[:, 0].min()) or 1.0

            L = arrows_rel_len * xspan
            U, V = udir[:, 0] * L, udir[:, 1] * L
            scale_units = "xy"
            scale_val = 1.0
        else:
            U, V = vecs[:, 0], vecs[:, 1]
            scale_units = arrows_scale_units
            scale_val = arrows_scale

        q = ax.quiver(
            mids[:, 0],
            mids[:, 1],
            U,
            V,
            cvals,  # <-- same scalar as the line
            cmap=cmap,
            norm=norm_a,  # <-- same norm as the line
            angles="xy",
            scale_units=scale_units,
            scale=scale_val,
            width=arrows_width,
            headwidth=arrows_headwidth,
            headlength=arrows_headlength,
            headaxislength=arrows_headaxislength,
            zorder=3,
            alpha=arrow_alpha,
        )

    # ---------- Cosmetics ----------
    if draw_mean_hline:
        ax.axhline(
            np.median(spots["y"]),
            label="y-Median of tracks",
            alpha=0.6,
            color="tab:orange",
        )

    _finalize_axes(ax, legend, draw_mean_hline, spots, equal_axes, invert_y, ax_units)

    if colorbar:
        cbar = plt.colorbar(lc, ax=ax, pad=0.01)
        cbar.set_label(cbar_label)

    return (fig, ax)


def _finalize_axes(ax, legend, draw_mean_hline, spots, equal_axes, invert_y, ax_units):
    # ax.autoscale()
    # if equal_axes:
    # ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(rf"x ({ax_units})")
    ax.set_ylabel(rf"y ({ax_units})")
    if legend and draw_mean_hline:
        ax.legend(loc="upper right", handlelength=0.3)


def add_balanced_axis_arrow(
    ax,
    a=0.10,
    direction="up",
    side="left",
    text_pos="middle",
    text="E",
    m=0.03,          # now: absolute x-offset in AXES coords (good defaults ~0.02-0.05)
    y_text=None,
    **kw,
):
    """
    Vertical arrow in AXES coords (0..1) with equal margins:
      bottom = top = a, and side margin (left or right) = a.

    Text placement:
      - x is in AXES coords, placed next to the arrow by an absolute offset m.
      - y is in DATA coords:
          y_text is a fraction of the current y-range (0.10 -> 10% above ymin)
        If text_pos == "middle" and y_text is None: use 50%.

    """
    if not (0 <= a < 0.5):
        raise ValueError("a must be in [0, 0.5).")

    s = side.lower()
    if s not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    # arrow x position (axes coords)
    x = a if s == "left" else (1 - a)

    # arrow y span (axes coords)
    up = direction.lower().startswith("u")
    y0, dy = (a, 1 - 2 * a) if up else (1 - a, -(1 - 2 * a))

    # --- text x next to arrow (axes coords) ---
    # left side: label to the left of the arrow; right side: label to the right
    x_text = x + m if s == "left" else x - m

    # --- text y in DATA coords as fraction of y-range ---
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if yr == 0:
        y_data = ymin
    else:
        if text_pos == "middle" and y_text is None:
            frac = 0.5
        else:
            frac = 0.1 if y_text is None else float(y_text)
        y_data = ymin + frac * yr

    text_transform = blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(
        x_text,
        y_data,
        text,
        transform=text_transform,
        rotation=0,
        color=kw.get("color", "k"),
        ha=kw.get("ha", "center"),
        va=kw.get("va", "center"),
        zorder=kw.get("zorder", 1000),
        clip_on=False,
    )

    arr = mpatches.FancyArrow(
        x, y0, 0.0, dy,
        width=0.0,
        length_includes_head=True,
        head_width=kw.get("head_width", 0.02),
        head_length=kw.get("head_length", 0.03),
        transform=ax.transAxes,
        fc=kw.get("color", "k"),
        ec=kw.get("color", "k"),
        lw=kw.get("lw", 2),
        zorder=kw.get("zorder", 1000),
        clip_on=False,
    )
    ax.add_patch(arr)
    return arr
