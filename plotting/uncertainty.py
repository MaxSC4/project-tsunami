# ------------------------------------------------------------
# Estimation simple de l'incertitude spatiale autour de la source
# à partir de la variation du RMSE des temps d'arrivée.
# ------------------------------------------------------------
import numpy as np

from tsunami.speed_integrator import travel_time_seconds
from tsunami.geo import arc_length_m

def _candidate_rmse(src_lat, src_lon, stations, t_obs_s,
                    depth_fn,
                    n_samples=800, h_min=50.0, shore_trim=10,
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
            h_min=h_min,
            shore_trim=shore_trim,
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
    rel_increase=0.15,
    n_samples=800,
    h_min=50.0,
    shore_trim=10,
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
        shore_trim=shore_trim,
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
            shore_trim=shore_trim,
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
            shore_trim=shore_trim,
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
    # (approximation isotrope de l'incertitude)
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