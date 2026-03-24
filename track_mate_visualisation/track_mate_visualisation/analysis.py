import numpy as np
from scipy.signal import find_peaks
from numpy.typing import NDArray
import pandas as pd


def path_analytics(df: pd.DataFrame):
    rows = {}
    for tid, g in df.groupby("TRACK_ID"):

        x = g["x"].to_numpy()
        y = g["y"].to_numpy()

        rows[tid] = {
            "length": path_length(x, y),
            "sign_changes_x": dir_changes(x),
            "sign_changes_y": dir_changes(y),
            "min_max_x": float(np.nanmax(x) - np.nanmin(x)),
            "min_max_y": float(np.nanmax(y) - np.nanmin(y)),
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "TRACK_ID"
    return out.sort_index()


def dir_changes(
    arr: NDArray[np.floating],
    *,
    eps: float | None = None,  # deadband in data units (e.g. px). None => robust auto
    k: float = 2.5,  # scale for auto-deadband: eps = k * 1.4826 * MAD(diffs)
    min_run: int = 3,  # require at least this many consecutive steps per run
    min_disp: float = 1.0,  # require at least this total displacement per run
) -> int:
    """Count robust direction reversals (left↔right or up↔down) in a 1D trajectory."""
    n = arr.size
    if n < 3:
        return 0

    # Finite differences (velocity proxy)
    dx = np.diff(arr)

    # Deadband threshold
    if eps is None:
        mad = np.median(np.abs(dx - np.median(dx)))
        eps = k * 1.4826 * mad  # robust scale ~ std for normal data

    step = dx.copy()
    step[np.abs(step) < max(0.0, float(eps))] = 0.0
    signs = np.sign(step).astype(int)

    # Keep only non-zero steps (zeros are "no decision")
    nz_idx = np.flatnonzero(signs)
    if nz_idx.size == 0:
        return 0

    # Run-length encode non-zero sign segments; enforce min_run & min_disp
    runs: list[int] = []
    start = nz_idx[0]
    cur_sign = signs[start]
    cum_disp = step[start]
    length = 1

    for i in nz_idx[1:]:
        if signs[i] == cur_sign and i == start + length:
            cum_disp += step[i]
            length += 1
        else:
            if length >= min_run and abs(cum_disp) >= min_disp:
                runs.append(cur_sign)
            # start a new run
            start = i
            cur_sign = signs[i]
            cum_disp = step[i]
            length = 1

    # finalize last run
    if length >= min_run and abs(cum_disp) >= min_disp:
        runs.append(cur_sign)

    if len(runs) < 2:
        return 0
    # Count sign flips between accepted runs (+1→-1 or -1→+1)
    return int(np.sum(np.diff(runs) != 0))


def path_length(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Return the length of the trajectory"""
    return np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
