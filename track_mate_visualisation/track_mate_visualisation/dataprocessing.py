import pandas as pd
import numpy as np
from typing import Tuple, Optional, Callable, TypeVar
from . import analysis

A = TypeVar("A", bound=np.typing.NDArray[np.floating])


def _read_edges(path: str, skiprows: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).iloc[skiprows:]
    df = df[
        [
            "EDGE_TIME",
            "EDGE_X_LOCATION",
            "EDGE_Y_LOCATION",
            "TRACK_ID",
        ]
    ].astype(float)
    df.rename(
        columns={
            "EDGE_X_LOCATION": "x",
            "EDGE_Y_LOCATION": "y",
            "EDGE_TIME": "t",
        },
        inplace=True,
    )
    return df


def _read_spots(path: str, skiprows: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).iloc[skiprows:]
    df = df[
        [
            "POSITION_X",
            "POSITION_Y",
            "POSITION_T",
            "TRACK_ID",
        ]
    ].astype(float)
    df.rename(
        columns={"POSITION_X": "x", "POSITION_Y": "y", "POSITION_T": "t"},
        inplace=True,
    )
    return df


def combine_csv(
    *,
    paths_edges: list[str],
    paths_spots: list[str],
    batch_size: int = 2000,
) -> Tuple[pd.DataFrame]:
    """Combine split TrackMate CSV exports into continuous edge/spot tables.

    Each file is treated as a batch and receives:
    - a time offset of `batch_index * batch_size`
    - a cumulative TRACK_ID offset so IDs are unique globally
    """

    skiprows = 3

    batch_edges = []
    next_id_offset = 0
    # Combine edges with cumulative TRACK_ID offsets.
    for i, p in enumerate(paths_edges):
        _ = _read_edges(p, skiprows=skiprows)
        _["TRACK_ID"] = _["TRACK_ID"] + next_id_offset
        _["t"] = _["t"] + i * batch_size
        batch_edges.append(_)
        if not _.empty:
            next_id_offset = int(_["TRACK_ID"].max()) + 1

    batch_spots = []
    next_id_offset = 0
    # Combine spots with cumulative TRACK_ID offsets.
    for i, p in enumerate(paths_spots):
        _ = _read_spots(p, skiprows=skiprows)
        _["TRACK_ID"] = _["TRACK_ID"] + next_id_offset
        _["t"] = _["t"] + i * batch_size
        batch_spots.append(_)
        if not _.empty:
            next_id_offset = int(_["TRACK_ID"].max()) + 1

    edges = pd.concat(batch_edges)
    spots = pd.concat(batch_spots)

    return dict(edges=edges, spots=spots)


def _apply_per_track(g: pd.DataFrame, fn: Callable[[A], A]) -> pd.DataFrame:
    """Apply a 1D filter to x and y for a single TRACK_ID group, preserving length."""
    g = g.sort_values("t").copy()
    x_in = g["x"].to_numpy()
    y_in = g["y"].to_numpy()
    if len(x_in) < 3 or len(y_in) < 3:
        return g
    x_out = fn(x_in)
    y_out = fn(y_in)
    if x_out.shape != x_in.shape or y_out.shape != y_in.shape:
        raise ValueError("apply_filter must return arrays of the same length as input.")
    g["x"] = x_out
    g["y"] = y_out
    return g


def filter_df(
    *,
    df_edges: pd.DataFrame,
    df_spots: pd.DataFrame,
    conversion_factors: Tuple[float],
    min_y: float | str = 1.0,
    std_min_x: float = 0.0,
    std_min_y: float = 1.0,
    min_max_y: float = 1.0,
    min_num_points: int = 10,
    max_sign_changes_x: Optional[int] = None,
    max_sign_changes_y: Optional[int] = None,
    min_max_x: Optional[int] = None,
    min_length: Optional[int] = None,
    apply_filter: Optional[Callable[[A], A]] = None,  # 1D per-axis
    apply_track_filter: Optional[
        Callable[[pd.DataFrame], pd.DataFrame]
    ] = None,  # per-track DF
    track_filter_params: Optional[ Tuple[float] ] = None,
    flip_y_to_cartesian: bool = True,
    image_shape: Optional[Tuple[int, int]] = (720, 1280),  # (ny, nx) in pixels
):
    """Convert, filter, and optionally smooth trajectories.

    The pipeline is:
    1) optional image-coordinate to Cartesian flip on y
    2) unit conversion (px/frame -> physical units)
    3) coarse per-track filtering on simple summary statistics
    4) geometric/path filtering using `analysis.path_analytics`
    5) optional per-axis / per-track smoothing
    """

    # unpack conversion
    fps, sx, sy = conversion_factors

    # Copy before mutation.
    edges_phys = df_edges.copy()
    spots_phys = df_spots.copy()

    # 1) Optional pixel-space y-flip before converting to physical units.
    if flip_y_to_cartesian:
        if image_shape is None:
            raise ValueError("image_shape is required when flip_y_to_cartesian=True.")
        ny, _ = image_shape  # pixels
        edges_phys["y"] = (ny - 1) - edges_phys["y"]
        spots_phys["y"] = (ny - 1) - spots_phys["y"]

    # 2) Convert position/time to physical units.
    edges_phys["t"] /= fps
    spots_phys["t"] /= fps
    edges_phys["x"] *= sx
    edges_phys["y"] *= sy
    spots_phys["x"] *= sx
    spots_phys["y"] *= sy

    # Keep the converted-but-unsmoothed trajectory around for plotting/debugging.
    edges_phys["x_raw"] = edges_phys.x
    edges_phys["y_raw"] = edges_phys.y

    # If velocity/acceleration columns exist, convert them into physical units too.
    for comp, scale in (("x", sx), ("y", sy)):
        vcol, acol = f"v{comp}", f"a{comp}"
        if vcol in edges_phys.columns:
            edges_phys[vcol] = (
                df_edges.loc[edges_phys.index, vcol].to_numpy() * scale * fps
            )
        if acol in edges_phys.columns:
            edges_phys[acol] = (
                df_edges.loc[edges_phys.index, acol].to_numpy() * scale * (fps**2)
            )

    # 3) Compute coarse filtering stats in physical units.
    if min_y == "median":
        min_y = float(np.median(edges_phys["y"]))

    g = edges_phys.groupby("TRACK_ID", sort=False)
    stats = g.agg(
        y_min=("y", "min"),
        y_max=("y", "max"),
        x_std=("x", "std"),
        y_std=("y", "std"),
        n=("y", "size"),
    )

    ids_any = stats.index[
        (stats["y_min"] >= min_y)
        & (stats["y_std"] > std_min_y)
        & (stats["x_std"] > std_min_x)
        & (stats["n"] > min_num_points)
        & ((stats["y_max"] - stats["y_min"]) >= min_max_y)
    ]
    if len(ids_any) == 0:
        return {"edges": edges_phys.iloc[0:0], "spots": spots_phys.iloc[0:0]}

    edges_phys_red = edges_phys[edges_phys["TRACK_ID"].isin(ids_any)]

    pa = analysis.path_analytics(edges_phys_red)
    if min_max_x is not None:
        pa = pa[pa.min_max_x >= min_max_x]
    if min_length is not None:
        pa = pa[pa.length >= min_length]
    if len(pa) == 0:
        return {"edges": edges_phys.iloc[0:0], "spots": spots_phys.iloc[0:0]}

    if max_sign_changes_x is not None:
        pa = pa[pa.sign_changes_x <= max_sign_changes_x]
    if max_sign_changes_y is not None:
        pa = pa[pa.sign_changes_y <= max_sign_changes_y]
    if len(pa) == 0:
        return {"edges": edges_phys.iloc[0:0], "spots": spots_phys.iloc[0:0]}

    edges_phys_red = edges_phys_red[edges_phys_red["TRACK_ID"].isin(pa.index)]

    # 4) Apply smoothing only to tracks that passed all filters.
    if apply_filter is not None:
        edges_phys_red = edges_phys_red.groupby("TRACK_ID", group_keys=False).apply(
            _apply_per_track, fn=apply_filter, include_groups=False
        )

    if apply_track_filter is not None:
        edges_phys_red = edges_phys_red.groupby("TRACK_ID", group_keys=False).apply(
            apply_track_filter, include_groups=False
        )

    reduced_tracks = edges_phys_red["TRACK_ID"]
    spots_phys_red = spots_phys[spots_phys["TRACK_ID"].isin(reduced_tracks)]

    return {"edges": edges_phys_red, "spots": spots_phys_red}

def _apply_mask(df: pd.DataFrame, mask: pd.core.indexes.base.Index):
    return df[df["TRACK_ID"].isin(mask)].copy().sort_values(by="t")


def flip_to_cartesian_yup(
    edges: pd.DataFrame,
    spots: Optional[pd.DataFrame] = None,
    image_shape: Optional[Tuple[int, int]] = None,  # (ny, nx) in pixels
    sy: float = 1.0,  # y scale [physical units / px]
    inplace: bool = False,
):
    """
    Make dataset Cartesian (y up):
      - Flip signs of vy, ay so that positive is 'up'
      - If image height is known (image_shape + sy), also reflect y positions: y <- H - y

    Parameters
    ----------
    edges : DataFrame
        Must contain columns ['y'] and optionally ['vy','ay'] in the same physical units as the rest of your pipeline.
    spots : DataFrame, optional
        If provided, its 'y' will be reflected too (positions only).
    image_shape : (ny, nx), optional
        Image size in pixels. If given (with `sy`), positions are reflected about y=H,
        where H = ny * sy (same units as your 'y' column).
    sy : float
        Physical units per pixel along y (e.g., mm/px). Used only when image_shape is given.
    inplace : bool
        Modify inputs in place if True.

    Returns
    -------
    edges_out[, spots_out]
    """
    df_e = edges if inplace else edges.copy()
    df_s = None if spots is None else (spots if inplace else spots.copy())

    # 1) Reflect positions if we know the physical image height H
    if image_shape is not None:
        ny, _ = image_shape
        H = ny * sy
        if "y" in df_e.columns:
            df_e["y"] = H - df_e["y"]
        if df_s is not None and "y" in df_s.columns:
            df_s["y"] = H - df_s["y"]

    # 2) Flip velocities/accelerations so +vy/+ay means upward
    if "vy" in df_e.columns:
        df_e["vy"] = -df_e["vy"]
    if "ay" in df_e.columns:
        df_e["ay"] = -df_e["ay"]

    return (df_e, df_s) if df_s is not None else df_e
