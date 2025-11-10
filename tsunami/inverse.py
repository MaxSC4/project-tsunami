"""
Inversion robuste de la source d’un tsunami à partir des temps d’arrivée observés.

Principe :
-----------
1. On génère une grille de candidats (latitude, longitude).
2. Pour chaque candidat, on calcule les temps de trajet modèles jusqu’aux stations.
3. On estime t₀* = moyenne (ou médiane) de (t_obs - T_mod).
4. Le misfit est basé sur l’écart résiduel physique :
        misfit = somme( (t_obs - (t₀* + T_mod))² ) sur stations valides
5. On raffine la recherche autour du meilleur point jusqu’à convergence spatiale.
"""

import numpy as np
from tsunami.speed_integrator import travel_time_seconds

def triangulation_inversion(
    stations,             # (Ns, 2) lat, lon
    t_obs_s,              # (Ns,) temps observés (s)
    depth_fn,             # fonction depth(lat, lon)
    phi_min, phi_max,     # bornes latitude (°)
    lam_min, lam_max,     # bornes longitude (°)
    n=15,                 # résolution initiale
    precision_deg=0.1,    # seuil d'arrêt
    max_iter=7,           # itérations max
    min_valid_stations=3, # minimum requis
    robust=True,          # si True : écarte les outliers
    verbose=True
):
    """
    Inversion de position de la source par recherche de grille adaptative.

    Retourne :
        best_lat, best_lon, best_t0, stats
    """
    stations = np.asarray(stations, dtype=float)
    t_obs_s = np.asarray(t_obs_s, dtype=float)
    Ns = stations.shape[0]

    if Ns < 2:
        raise ValueError("Au moins deux stations sont nécessaires.")

    best = dict(lat=np.nan, lon=np.nan, misfit=np.inf, t0=np.nan)
    tries = 0

    while (phi_max - phi_min) > precision_deg and tries < max_iter:
        # 1. Grille de recherche
        lats = np.linspace(phi_min, phi_max, n)
        lons = np.linspace(lam_min, lam_max, n)
        dlat = (phi_max - phi_min) / n
        dlon = (lam_max - lam_min) / n
        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
        Np = grid_lat.size

        # 2. Calcul vectorisé (mais encore station par station)
        Tmod = np.full((Np, Ns), np.nan)
        for j in range(Ns):
            lat_s, lon_s = stations[j]
            # Calculs en bloc sur tous les candidats
            for p in range(Np):
                Tmod[p, j] = travel_time_seconds(grid_lat.flat[p], grid_lon.flat[p], lat_s, lon_s, depth_fn)

        # 3. Évaluation du misfit
        obs_mat = np.broadcast_to(t_obs_s, (Np, Ns))
        mask = np.isfinite(Tmod) & np.isfinite(obs_mat)
        diff = np.where(mask, obs_mat - Tmod, np.nan)

        # Estimation de t0 (origine) : moyenne robuste ou simple moyenne
        if robust:
            # Médiane robuste → moins sensible aux valeurs extrêmes
            t0_hat = np.nanmedian(diff, axis=1)
        else:
            t0_hat = np.nanmean(diff, axis=1)

        resid = obs_mat - (Tmod + t0_hat[:, None])
        resid[~mask] = np.nan

        # Option robuste : limitation d’influence des outliers
        if robust:
            mad = np.nanmedian(np.abs(resid), axis=1)
            scale = np.maximum(mad, 1.0)
            resid = np.clip(resid / scale[:, None], -3, 3) * scale[:, None]

        # Misfit RMS (physiquement interprétable en secondes)
        misfit = np.sqrt(np.nanmean(resid**2, axis=1))
        valid_counts = np.sum(mask, axis=1)
        misfit = np.where(valid_counts >= min_valid_stations, misfit, np.inf)

        # 4. Sélection du meilleur candidat
        idx_best = int(np.nanargmin(misfit))
        cand = dict(
            lat=float(grid_lat.flat[idx_best]),
            lon=float(grid_lon.flat[idx_best]),
            misfit=float(misfit[idx_best]),
            t0=float(t0_hat[idx_best]),
            valid=int(valid_counts[idx_best])
        )

        if cand["misfit"] < best["misfit"]:
            best.update(cand)

        if verbose:
            print(
                f"[Iter {tries+1}] best=({cand['lat']:.2f}, {cand['lon']:.2f})  "
                f"misfit={cand['misfit']:.2f}s  stations={cand['valid']}/{Ns}"
            )

        # 5. Raffinement autour du meilleur point
        phi_min = best["lat"] - dlat
        phi_max = best["lat"] + dlat
        lam_min = best["lon"] - dlon
        lam_max = best["lon"] + dlon
        tries += 1

    return best["lat"], best["lon"], best["t0"], best