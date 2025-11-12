# ------------------------------------------------------------
# Scatter "observé vs modélisé" avec ajustement t0* (slope=1),
# stats (RMSE/MAE/R^2), et régression libre en pointillé.
# ------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

from tsunami.speed_integrator import travel_time_seconds

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def compute_modeled_times(stations, src_lat, src_lon, depth_fn, n_samples=800, h_min=50.0, shore_trim=10):
    """
    Calcule T_i = temps de trajet modélisé (s) depuis la source (src_lat, src_lon)
    jusqu'à chaque station. Renvoie un np.array shape (Ns,) avec NaN si trajet impossible.
    """
    stations = np.asarray(stations, float)
    Ns = stations.shape[0]
    T = np.full(Ns, np.nan, float)
    for i, (lat_s, lon_s) in enumerate(stations):
        Ti = travel_time_seconds(src_lat, src_lon, lat_s, lon_s, depth_fn, n_samples=n_samples, h_min=h_min, shore_trim=shore_trim)
        T[i] = np.nan if not np.isfinite(Ti) else float(Ti)
    return T

def fit_t0_fixed_slope1(t_obs, T_model, robust=True):
    """
    Ajuste t0* dans le modèle t_obs = t0* + T_model + eps,
    en ignorant les NaN/Inf. Si robust=True, utilise la médiane des résidus (L1).
    Sinon, moyenne (moindres carrés avec pente fixée à 1).
    """
    t_obs = np.asarray(t_obs, float)
    T_model = np.asarray(T_model, float)
    mask = np.isfinite(t_obs) & np.isfinite(T_model)
    if not np.any(mask):
        return np.nan, mask
    r = t_obs[mask] - T_model[mask]
    t0 = np.nanmedian(r) if robust else np.nanmean(r)
    return float(t0), mask

def fit_affine_free(t_obs, T_model):
    """
    Régression linéaire y = a + b x (sans imposer b=1), ignorante des NaN.
    Renvoie (a, b) ou (NaN, NaN) si pas assez de points.
    """
    t_obs = np.asarray(t_obs, float)
    T_model = np.asarray(T_model, float)
    mask = np.isfinite(t_obs) & np.isfinite(T_model)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    x = T_model[mask]
    y = t_obs[mask]
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)

def _stats(y_true, y_pred):
    """
    RMSE, MAE, R^2 (coefficient de détermination), N.
    Ignore NaN/Inf via masque commun.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return np.nan, np.nan, np.nan, 0
    yt = y_true[m]
    yp = y_pred[m]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    # R^2
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2, int(np.sum(m))


def plot_obs_vs_model(t_obs_s,
                      stations,
                      src_lat, src_lon,
                      depth_fn,
                      station_names=None,
                      robust=True,
                      n_samples=800, h_min=50.0, shore_trim=10,
                      show_free_fit=True,
                      figpath="outputs/obs_vs_model.png",
                      title="Observed vs. Modelled arrival times",
                      show=True):
    """
    Produit une figure comparant t_obs (y) et T_model (x),
    avec la droite y = t0* + x (pente fixée à 1) + stats.

    Args
    ----
    t_obs_s : array (Ns,)
        Temps observés (absolus en s, par ex. POSIX normalisé).
    stations : array (Ns,2)
        (lat, lon) des stations.
    src_lat, src_lon : floats
        Source estimée.
    depth_fn : callable
        Bathy → profondeur (m), déjà prête (wrap interne).
    station_names : list[str] ou None
        Noms à annoter près des points.
    robust : bool
        t0* robuste via médiane (sinon moyenne).
    show_free_fit : bool
        Ajoute une régression libre y = a + b x en pointillé (diagnostic).
    """
    # 1) Modèle
    T_model = compute_modeled_times(stations, src_lat, src_lon, depth_fn,
                                    n_samples=n_samples, h_min=h_min, shore_trim=shore_trim)

    # 2) Ajustements
    t0_fixed, valid_mask = fit_t0_fixed_slope1(t_obs_s, T_model, robust=robust)
    y_pred_fixed = t0_fixed + T_model

    a_free, b_free = fit_affine_free(t_obs_s, T_model)
    y_pred_free = a_free + b_free * T_model if np.isfinite(a_free) and np.isfinite(b_free) else None

    # 3) Stats (sur le modèle “physique” pente=1)
    rmse, mae, r2, N = _stats(t_obs_s, y_pred_fixed)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)

    x = T_model
    y = t_obs_s
    m = np.isfinite(x) & np.isfinite(y)

    ax.scatter(x[m], y[m], s=45, facecolor="tab:blue", edgecolor="white", linewidth=0.7, label="Stations")

    # Lignes de référence
    if np.any(m):
        xgrid = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 200)
        ax.plot(xgrid, t0_fixed + xgrid, color="tab:red", lw=2.0,
                label=r"Fit (slope=1):  $t_0^{*}$ = {t0_fixed:,.0f} s")

        if show_free_fit and (y_pred_free is not None):
            ax.plot(xgrid, a_free + b_free * xgrid, color="0.25", lw=1.6, ls="--",
                    label=f"Free fit: y = {a_free:,.0f} + {b_free:.3f} x")

    # Annotations des points
    if station_names is None:
        station_names = [f"S{i+1}" for i in range(len(x))]
    for xi, yi, name in zip(x[m], y[m], np.array(station_names)[m]):
        ax.text(xi + 0.01*(ax.get_xlim()[1]-ax.get_xlim()[0]),
                yi + 0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                str(name), fontsize=8, color="k", alpha=0.9)

    # Axes + stats
    ax.set_xlabel("Modelled travel time $T_i(\\mathbf{x}_s)$ (s)")
    ax.set_ylabel("Observed arrival time $t_i^{obs}$ (s)")
    ax.set_title(title)

    ax.grid(True, linestyle=":", alpha=0.5)

    # encart de stats
    txt = (f"N = {N}\n"
           f"RMSE = {rmse:,.1f} s ({rmse/60:.1f} min)\n"
           f"MAE  = {mae:,.1f} s ({mae/60:.1f} min)\n"
           f"$R^2$ = {r2:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    ax.legend(loc="lower right")

    # 5) Save/show
    _ensure_dir(figpath)
    fig.tight_layout()
    fig.savefig(figpath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    # 6) Renvoie des éléments utiles
    residuals = y - (t0_fixed + x)
    return {
        "T_model": T_model,
        "t0_fixed": t0_fixed,
        "rmse": rmse, "mae": mae, "r2": r2, "N": N,
        "residuals": residuals,
        "a_free": a_free, "b_free": b_free
    }