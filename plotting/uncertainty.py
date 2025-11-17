# ------------------------------------------------------------
# Estimation simple de l'incertitude spatiale autour de la source
# à partir de la variation du RMSE des temps d'arrivée.
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

from tsunami.speed_integrator import travel_time_seconds
from tsunami.geo import arc_length_m

def _candidate_rmse(src_lat, src_lon, stations, t_obs_s,
                    depth_fn,
                    n_samples=800, h_min=0.0,
                    robust=True):
    """
    Calcule le RMSE des temps d'arrivée pour une source candidate.

    Modèle : t_i^obs ≈ t0* + T_i(src -> station_i)

    - On ajuste t0* (offset temporel) en imposant une pente = 1 entre T_i et t_i^obs.
    - On renvoie le RMSE sur les stations pour lesquelles le trajet est valide.
    """
    stations = np.asarray(stations, float)
    t_obs_s = np.asarray(t_obs_s, float)

    Ns = stations.shape[0]
    T = np.full(Ns, np.nan, float)

    # temps de trajet modélisés
    for i, (lat_s, lon_s) in enumerate(stations):
        Ti = travel_time_seconds(
            src_lat, src_lon,
            lat_s, lon_s,
            depth_fn,
            n_samples=n_samples,
            h_min=h_min
        )
        T[i] = np.nan if not np.isfinite(Ti) else float(Ti)

    # masque de stations valides
    m = np.isfinite(T) & np.isfinite(t_obs_s)
    if np.sum(m) < 2:
        return np.nan, np.nan, m  # pas assez de stations

    t_obs = t_obs_s[m]
    Tm = T[m]

    # ajustement de t0* (offset) : t_obs = t0* + Tm + eps
    r = t_obs - Tm
    if robust:
        t0 = float(np.median(r))
    else:
        t0 = float(np.mean(r))

    residus = t_obs - (t0 + Tm)
    rmse = float(np.sqrt(np.mean(residus**2)))
    return rmse, t0, m


def estimate_spatial_uncertainty(
    best_lat, best_lon,
    stations, t_obs_s,
    depth_fn,
    span_lat_deg=4.0,
    span_lon_deg=6.0,
    step_deg=0.25,
    rel_increase=0.10,
    n_samples=800,
    h_min=0.0,
    robust=True,
):
    """
    Estime une incertitude en latitude et longitude autour de la source.

    Idée :
        - on calcule RMSE(min) au point (best_lat, best_lon),
        - on scanne en 1D autour :
           * lat variable, lon = best_lon
           * lon variable, lat = best_lat
        - on cherche les points où RMSE dépasse (1 + rel_increase) * RMSE_min
        de chaque côté → Δφ, Δλ.

    Retourne :
        {
            "dlat_deg": Δφ (demi-largeur en degrés),
            "dlon_deg": Δλ (demi-largeur en degrés),
            "radius_km": rayon moyen en km,
            "rmse_min": RMSE au minimum,
        }
    """
    # RMSE au minimum
    rmse_min, t0_best, mask_best = _candidate_rmse(
        best_lat, best_lon,
        stations, t_obs_s,
        depth_fn,
        n_samples=n_samples,
        h_min=h_min,
        robust=robust,
    )

    if not np.isfinite(rmse_min):
        return {
            "dlat_deg": np.nan,
            "dlon_deg": np.nan,
            "radius_km": np.nan,
            "rmse_min": np.nan,
        }

    threshold = rmse_min * (1.0 + rel_increase)

    # --- scan en latitude (lon fixé) ---
    lats = np.arange(best_lat - span_lat_deg, best_lat + span_lat_deg + 1e-6, step_deg, dtype=float)

    rmse_lat = []
    for lat in lats:
        r_i, _, _ = _candidate_rmse(
            lat, best_lon,
            stations, t_obs_s,
            depth_fn,
            n_samples=n_samples,
            h_min=h_min,
            robust=robust,
        )
        rmse_lat.append(r_i)
    rmse_lat = np.asarray(rmse_lat)

    # côté nord et sud où RMSE dépasse le seuil
    dlat_up = span_lat_deg
    dlat_down = span_lat_deg

    # vers le nord (lat croissantes)
    for lat, r in zip(lats[lats >= best_lat], rmse_lat[lats >= best_lat]):
        if np.isfinite(r) and r > threshold:
            dlat_up = lat - best_lat
            break

    # vers le sud (lat décroissantes)
    for lat, r in zip(lats[lats <= best_lat][::-1], rmse_lat[lats <= best_lat][::-1]):
        if np.isfinite(r) and r > threshold:
            dlat_down = best_lat - lat
            break

    dlat_deg = 0.5 * (dlat_up + dlat_down)

    # --- scan en longitude (lat fixée) ---
    lons = np.arange(best_lon - span_lon_deg,
                     best_lon + span_lon_deg + 1e-6,
                     step_deg, dtype=float)

    rmse_lon = []
    for lon in lons:
        r_i, _, _ = _candidate_rmse(
            best_lat, lon,
            stations, t_obs_s,
            depth_fn,
            n_samples=n_samples,
            h_min=h_min,
            robust=robust,
        )
        rmse_lon.append(r_i)
    rmse_lon = np.asarray(rmse_lon)

    dlon_east = span_lon_deg
    dlon_west = span_lon_deg

    for lon, r in zip(lons[lons >= best_lon], rmse_lon[lons >= best_lon]):
        if np.isfinite(r) and r > threshold:
            dlon_east = lon - best_lon
            break

    for lon, r in zip(lons[lons <= best_lon][::-1], rmse_lon[lons <= best_lon][::-1]):
        if np.isfinite(r) and r > threshold:
            dlon_west = best_lon - lon
            break

    dlon_deg = 0.5 * (dlon_east + dlon_west)

    # --- rayon effectif en km ---
    # on prend deux directions principales : nord et est
    # (approximation de l'incertitude)
    lat_north = best_lat + dlat_deg
    lon_east = best_lon + dlon_deg

    # distance moyenne des deux directions
    d_north = arc_length_m(best_lat, best_lon, lat_north, best_lon)
    d_east  = arc_length_m(best_lat, best_lon, best_lat, lon_east)
    radius_km = 0.5 * (d_north + d_east) / 1000.0

    return {
        "dlat_deg": float(dlat_deg),
        "dlon_deg": float(dlon_deg),
        "radius_km": float(radius_km),
        "rmse_min": float(rmse_min),
    }


# ------------------------------------------------------------
# Profils 1D de misfit (RMSE) autour de la source
# ------------------------------------------------------------

def compute_misfit_profiles(
    best_lat, best_lon,
    stations, t_obs_s,
    depth_fn,
    span_lat_deg=4.0,
    span_lon_deg=6.0,
    step_deg=0.25,
    rel_increase=0.10,
    n_samples=800,
    h_min=0.0,
    robust=True,
):
    """
    Calcule les profils 1D de RMSE autour de la source :

        - RMSE(φ) pour φ dans [best_lat ± span_lat_deg], λ = best_lon
        - RMSE(λ) pour λ dans [best_lon ± span_lon_deg], φ = best_lat

    On utilise exactement le même RMSE que _candidate_rmse, avec ajustement
    de t0* pour chaque candidat.

    Retourne un dict avec tout ce qu'il faut pour tracer :

        {
            "lat_axis": latitudes testées,
            "rmse_lat": RMSE(φ),
            "lon_axis": longitudes testées,
            "rmse_lon": RMSE(λ),
            "rmse_min": RMSE au point (best_lat, best_lon),
            "threshold": RMSE_min * (1 + rel_increase),
            "dlat_deg": Δφ estimé,
            "dlon_deg": Δλ estimé,
        }
    """
    stations = np.asarray(stations, float)
    t_obs_s = np.asarray(t_obs_s, float)

    # --- RMSE au minimum (point donné par l'inversion) ---
    rmse_min, t0_best, mask_best = _candidate_rmse(
        best_lat, best_lon,
        stations, t_obs_s,
        depth_fn,
        n_samples=n_samples,
        h_min=h_min,
        robust=robust,
    )

    if not np.isfinite(rmse_min):
        raise RuntimeError("compute_misfit_profiles: rmse_min n'est pas fini (source invalide ?).")

    threshold = rmse_min * (1.0 + rel_increase)

    # --- Profil en latitude (lon fixée) ---
    lats = np.arange(
        best_lat - span_lat_deg,
        best_lat + span_lat_deg + 1e-6,
        step_deg,
        dtype=float,
    )

    rmse_lat = []
    for lat in lats:
        r_i, _, _ = _candidate_rmse(
            lat, best_lon,
            stations, t_obs_s,
            depth_fn,
            n_samples=n_samples,
            h_min=h_min,
            robust=robust,
        )
        rmse_lat.append(r_i)
    rmse_lat = np.asarray(rmse_lat)

    # Recherche des points où RMSE dépasse le seuil (comme estimate_spatial_uncertainty)
    dlat_up = span_lat_deg
    dlat_down = span_lat_deg

    # Vers le nord
    for lat, r in zip(lats[lats >= best_lat], rmse_lat[lats >= best_lat]):
        if np.isfinite(r) and r > threshold:
            dlat_up = lat - best_lat
            break
    # Vers le sud
    for lat, r in zip(lats[lats <= best_lat][::-1], rmse_lat[lats <= best_lat][::-1]):
        if np.isfinite(r) and r > threshold:
            dlat_down = best_lat - lat
            break

    dlat_deg = 0.5 * (dlat_up + dlat_down)

    # --- Profil en longitude (lat fixée) ---
    lons = np.arange(
        best_lon - span_lon_deg,
        best_lon + span_lon_deg + 1e-6,
        step_deg,
        dtype=float,
    )

    rmse_lon = []
    for lon in lons:
        r_i, _, _ = _candidate_rmse(
            best_lat, lon,
            stations, t_obs_s,
            depth_fn,
            n_samples=n_samples,
            h_min=h_min,
            robust=robust,
        )
        rmse_lon.append(r_i)
    rmse_lon = np.asarray(rmse_lon)

    dlon_east = span_lon_deg
    dlon_west = span_lon_deg

    # Vers l'est
    for lon, r in zip(lons[lons >= best_lon], rmse_lon[lons >= best_lon]):
        if np.isfinite(r) and r > threshold:
            dlon_east = lon - best_lon
            break
    # Vers l'ouest
    for lon, r in zip(lons[lons <= best_lon][::-1], rmse_lon[lons <= best_lon][::-1]):
        if np.isfinite(r) and r > threshold:
            dlon_west = best_lon - lon
            break

    dlon_deg = 0.5 * (dlon_east + dlon_west)

    return {
        "lat_axis": lats,
        "rmse_lat": rmse_lat,
        "lon_axis": lons,
        "rmse_lon": rmse_lon,
        "rmse_min": float(rmse_min),
        "threshold": float(threshold),
        "dlat_deg": float(dlat_deg),
        "dlon_deg": float(dlon_deg),
    }


def plot_misfit_profiles(
    best_lat, best_lon,
    profiles,
    title="Local RMS misfit profiles around source",
    savepath=None,
    show=True,
):
    """
    Trace deux courbes 1D :

        - RMSE(φ) vs φ (longitude fixée à best_lon)
        - RMSE(λ) vs λ (latitude fixée à best_lat)

    avec :
        - le minimum RMSE_min marqué,
        - une ligne horizontale au seuil threshold = (1+rel) RMSE_min,
        - des lignes verticales aux limites best ± Δφ, best ± Δλ.
    """
    lat_axis = profiles["lat_axis"]
    rmse_lat = profiles["rmse_lat"]
    lon_axis = profiles["lon_axis"]
    rmse_lon = profiles["rmse_lon"]
    rmse_min = profiles["rmse_min"]
    threshold = profiles["threshold"]
    dlat_deg = profiles["dlat_deg"]
    dlon_deg = profiles["dlon_deg"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120, sharey=True)
    ax_lat, ax_lon = axes

    # --- Profil latitude ---
    ax_lat.plot(lat_axis, rmse_lat, "-o", ms=3, label="RMSE(φ)")
    ax_lat.axvline(best_lat, color="k", linestyle="--", lw=1.5, label="Best φ*")
    ax_lat.axhline(rmse_min, color="tab:green", lw=1.5, label="RMSE min")
    ax_lat.axhline(threshold, color="tab:red", lw=1.2, linestyle=":", label="Seuil (1+δ)·RMSE min")

    # Lignes verticales pour les Δφ
    ax_lat.axvline(best_lat - dlat_deg, color="tab:red", linestyle="--", lw=1)
    ax_lat.axvline(best_lat + dlat_deg, color="tab:red", linestyle="--", lw=1)

    ax_lat.set_xlabel("Latitude (°N)")
    ax_lat.set_ylabel("RMS misfit (s)")
    ax_lat.set_title("Profil latitudinal (λ = {:.2f}°E)".format(best_lon))
    ax_lat.grid(True, linestyle=":", alpha=0.5)

    # --- Profil longitude ---
    ax_lon.plot(lon_axis, rmse_lon, "-o", ms=3, label="RMSE(λ)")
    ax_lon.axvline(best_lon, color="k", linestyle="--", lw=1.5, label="Best λ*")
    ax_lon.axhline(rmse_min, color="tab:green", lw=1.5, label="RMSE min")
    ax_lon.axhline(threshold, color="tab:red", lw=1.2, linestyle=":", label="Seuil (1+δ)·RMSE min")

    # Lignes verticales pour les Δλ
    ax_lon.axvline(best_lon - dlon_deg, color="tab:red", linestyle="--", lw=1)
    ax_lon.axvline(best_lon + dlon_deg, color="tab:red", linestyle="--", lw=1)

    ax_lon.set_xlabel("Longitude (°E)")
    ax_lon.set_title("Profil longitudinal (φ = {:.2f}°N)".format(best_lat))
    ax_lon.grid(True, linestyle=":", alpha=0.5)

    # Légende commune (on prend celle du 2e axe)
    ax_lon.legend(loc="upper right", frameon=True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, axes