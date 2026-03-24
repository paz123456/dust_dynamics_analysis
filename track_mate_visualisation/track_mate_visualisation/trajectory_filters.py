from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import median_filter  # has mode='nearest' (no zero padding)
from dataclasses import dataclass
import pandas as pd
from numpy.typing import NDArray





def smooth_track(df_track: pd.DataFrame) -> pd.DataFrame:
    # Get track id from column if present, else from the group name
    sigma_acc: float = 6e5  # square-root of the continuous-time acceleration noise intensity 
    rx: float = 0.05   # meas. std of x (mm)
    ry: float = 0.0225  # meas. std of y (mm)
    params = [sigma_acc, rx, ry]
    tid = (
        df_track["TRACK_ID"].iloc[0]
        if "TRACK_ID" in df_track.columns and len(df_track) > 0
        else getattr(df_track, "name", None)
    )

    g = df_track.sort_values("t").copy()
    t = g["t"].to_numpy(dtype=float)
    x = g["x"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)

    if len(g) < 2:
        g[["vx", "vy", "ax", "ay"]] = 0.0
    else:
        Xs, _diagPs = _kf_ca_rts(t, x, y, params)
        g.loc[:, "x"] = Xs[:, 0]
        g.loc[:, "vx"] = Xs[:, 1]
        g.loc[:, "ax"] = Xs[:, 2]
        g.loc[:, "y"] = Xs[:, 3]
        g.loc[:, "vy"] = Xs[:, 4]
        g.loc[:, "ay"] = Xs[:, 5]

    # Ensure TRACK_ID stays a column (not just the group index)
    if "TRACK_ID" not in g.columns and tid is not None:
        g.insert(0, "TRACK_ID", tid)

    return g

def _F_Q(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """State: [x, vx, ax, y, vy, ay]^T with white-jerk process noise q."""
    dt2, dt3, dt4, dt5 = dt * dt, dt * dt * dt, dt**4, dt**5
    Fx = np.array([[1.0, dt, 0.5 * dt2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=float)
    Qx = q * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=float,
    )
    F = np.zeros((6, 6))
    F[:3, :3] = Fx
    F[3:, 3:] = Fx
    Q = np.zeros((6, 6))
    Q[:3, :3] = Qx
    Q[3:, 3:] = Qx
    return F, Q


def _kf_ca_rts(
    t: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    p : list[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kalman filter + RTS smoother for CA model. Returns (X_smooth, diagP_smooth)."""
    n = len(t)
    if n == 0:  # nothing to do
        return np.empty((0, 6)), np.empty((0, 6))
    # ensure sorted by time
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    y = y[idx]

    sigma_acc, rx, ry = p
    # Measurement matrices
    H = np.zeros((2, 6))
    H[0, 0] = 1.0
    H[1, 3] = 1.0  # observe x,y only
    R = np.diag([rx**2, ry**2])
    q = float(sigma_acc**2)

    # init state from first two samples (or zeros if only one)
    Xf = np.zeros((n, 6))
    Pf = np.zeros((n, 6, 6))
    Xp = np.zeros((n, 6))
    Pp = np.zeros((n, 6, 6))

    x0, y0 = float(x[0]), float(y[0])
    if n >= 2:
        dt0 = max(1e-9, float(t[1] - t[0]))
        vx0 = (x[1] - x[0]) / dt0
        vy0 = (y[1] - y[0]) / dt0
    else:
        vx0 = vy0 = 0.0
    X = np.array([x0, vx0, 0.0, y0, vy0, 0.0], dtype=float)
    P = np.diag([rx**2, 10.0, 1.0, ry**2, 10.0, 1.0])  # broad on v,a

    # Forward pass
    for k in range(n):
        if k == 0:
            Xp[k] = X
            Pp[k] = P
        else:
            dt = max(1e-9, float(t[k] - t[k - 1]))
            F, Q = _F_Q(dt, q)
            X = F @ X
            P = F @ P @ F.T + Q
            Xp[k] = X
            Pp[k] = P

        # Update if measurement is finite
        zk = np.array([x[k], y[k]], dtype=float)
        meas_ok = np.all(np.isfinite(zk))
        if meas_ok:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            yk = zk - (H @ X)
            X = X + K @ yk
            P = (np.eye(6) - K @ H) @ P

        Xf[k] = X
        Pf[k] = P

    # RTS smoother (backward)
    Xs = Xf.copy()
    Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        dt = max(1e-9, float(t[k + 1] - t[k]))
        F, Q = _F_Q(dt, q)
        Pk = Pf[k]
        C = Pk @ F.T @ np.linalg.inv(Pp[k + 1])
        Xs[k] = Xf[k] + C @ (Xs[k + 1] - Xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

    # Undo time sort
    inv = np.empty_like(idx)
    inv[idx] = np.arange(n)
    return Xs[inv], np.stack([np.diag(Ps[i]) for i in range(n)], axis=0)[inv]


def _odd_leq(n, max_n):
    k = min(n, max_n)
    return 1 if k < 1 else (k if k % 2 else max(1, k - 1))


def hampel(a: np.ndarray, window: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    w = max(1, window // 2)
    med = np.array([np.median(a[max(0, i - w) : i + w + 1]) for i in range(len(a))])
    mad = (
        np.array(
            [
                np.median(np.abs(a[max(0, i - w) : i + w + 1] - med[i]))
                for i in range(len(a))
            ]
        )
        + 1e-12
    )
    z = (a - med) / (1.4826 * mad)
    out = a.copy()
    out[np.abs(z) > n_sigma] = med[np.abs(z) > n_sigma]
    return out


def make_traj_filter(
    median_k: int = 5,
    hampel_w: int = 7,
    hampel_sigma: float = 3.0,
    sg_window: int = 11,
    sg_poly: int = 2,
    butter_cutoff: float | None = None,  # fraction of Nyquist (e.g. 0.15)
    butter_order: int = 2,
):
    def f(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        n = a.size
        if n == 0:
            return a

        # 1) median (length-preserving; no zero padding)
        k = _odd_leq(median_k, n)
        if k > 1:
            a = median_filter(a, size=k, mode="nearest")

        # 2) Hampel (length-preserving)
        if hampel_w > 1:
            a = hampel(a, window=hampel_w, n_sigma=hampel_sigma)

        # 3) Savitzky–Golay (length-preserving)
        sw = _odd_leq(sg_window, n)
        if sw > 1 and sg_poly < sw:
            a = savgol_filter(a, window_length=sw, polyorder=sg_poly, mode="interp")

        # 4) Optional zero-phase Butterworth (length-preserving if long enough)
        if butter_cutoff is not None and 0 < butter_cutoff < 1:
            b, aa = butter(butter_order, butter_cutoff, btype="low")
            padlen = 3 * (max(len(aa), len(b)) - 1)
            if n > padlen:  # filtfilt requirement
                a = filtfilt(b, aa, a, method="gust")

        return a

    return f
