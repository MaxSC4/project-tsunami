"""
Inversion de la source d'un tsunami en utilisant des temps d'arrivée
normalisés par rapport à une station de référence (par défaut Los Angeles).

Idée :
------
On choisit une station de référence r (index ref_index) et on travaille avec
des temps relatifs :

    Δt_i^obs = t_i^obs - t_r^obs
    ΔT_i(mod) = T_i(src) - T_r(src)

La fonction coût associe à chaque source candidate x_s = (lat, lon) le misfit :

    misfit(x_s) = RMS_i [ Δt_i^obs - ΔT_i(mod) ]

Le temps d'origine t0* n'intervient pas dans le critère – il est estimé
a posteriori à partir de la meilleure source (ajustement en temps absolu).

Interface :
-----------
    best_lat, best_lon, t0_hat, stats = absolute_inversion(...)

`stats` contient notamment :
    - 'rmse_rel' : RMSE sur les temps *relatifs* (en secondes),
    - 'rmse_abs' : RMSE sur les temps *absolus* après estimation de t0*,
    - 'residuals_abs' : résidus absolus station par station,
    - 'residuals_rel' : résidus relatifs station par station,
    - 'T_model' : temps de trajet modélisés (s),
    - 'dt_obs' / 'dt_model' : temps relatifs observés / modélisés,
    - 'ref_index' : index de la station utilisée comme référence.
"""

import numpy as np
from tsunami.speed_integrator import travel_time_seconds


def absolute_inversion(
    stations,             # (Ns, 2) : lat, lon des stations
    t_obs_s,              # (Ns,)   : temps d'arrivée observés (s)
    depth_fn,             # callable depth(lat, lon) -> profondeur (m)
    phi_min, phi_max,     # bornes latitude de recherche (°)
    lam_min, lam_max,     # bornes longitude de recherche (°)
    ref_index=0,          # index de la station de référence (Los Angeles, etc.)
    n=15,                 # résolution initiale de la grille
    precision_deg=0.1,    # critère d'arrêt spatial (°)
    max_iter=7,           # itérations max de raffinement
    min_valid_stations=3, # nb minimal de stations marines valides
    robust=True,          # clipping robuste des outliers
    verbose=True,
):
    """
    Inversion de la position de la source à partir de temps *relatifs*
    normalisés par une station de référence.

    Paramètres
    ----------
    stations : array-like (Ns, 2)
        Latitude et longitude de chaque station (en degrés).
    t_obs_s : array-like (Ns,)
        Temps d'arrivée observés en secondes (absolus quelconques).
    depth_fn : callable
        Fonction bathymétrique depth(lat, lon) -> profondeur (m, négative en mer).
    phi_min, phi_max, lam_min, lam_max : float
        Bornes de la boîte de recherche initiale (en degrés).
    ref_index : int
        Index de la station de référence (celle dont Δt_obs = 0).
        Par exemple l'index de Los Angeles dans le tableau des stations.
    n : int
        Nombre de points de grille par axe à chaque itération.
    precision_deg : float
        On arrête le raffinement quand la boîte de recherche est plus petite
        que cette valeur en latitude ET longitude.
    max_iter : int
        Nombre maximal d'itérations de raffinements successifs.
    min_valid_stations : int
        Misfit mis à +inf si moins de ce nombre de stations marines valides.
    robust : bool
        Si True, applique un clipping robuste sur les résidus (type Huber léger).
    verbose : bool
        Affiche un résumé à chaque itération.

    Retour
    ------
    best_lat, best_lon : float
        Position estimée de la source (°).
    t0_hat : float
        Estimation du temps d'origine t0* (en secondes) a posteriori.
    stats : dict
        Dictionnaire de diagnostics, contenant notamment :
            - 'rmse_rel'      : RMSE sur les temps relatifs (s),
            - 'rmse_abs'      : RMSE sur les temps absolus (s),
            - 'residuals_rel' : (Ns,) résidus relatifs,
            - 'residuals_abs' : (Ns,) résidus absolus,
            - 'T_model'       : (Ns,) temps de trajet modélisés pour la
                                meilleure source (s),
            - 'dt_obs'        : (Ns,) temps relatifs observés,
            - 'dt_model'      : (Ns,) temps relatifs modélisés,
            - 'ref_index'     : index de la station de référence,
            - 'valid'         : nombre de stations marines valides.
    """
    stations = np.asarray(stations, dtype=float)
    t_obs_s  = np.asarray(t_obs_s, dtype=float)
    Ns = stations.shape[0]

    if Ns < 2:
        raise ValueError("absolute_inversion: au moins deux stations sont nécessaires.")
    if not (0 <= ref_index < Ns):
        raise ValueError(f"absolute_inversion: ref_index={ref_index} hors limites [0, {Ns-1}].")

    # --- 0) Temps relatifs observés par rapport à la station de référence ---
    # Δt_obs[i] = t_obs[i] - t_obs[ref]
    dt_obs = t_obs_s - t_obs_s[ref_index]

    best = dict(lat=np.nan, lon=np.nan, misfit=np.inf)
    tries = 0

    # --- boucle de raffinement de la grille ---
    while (phi_max - phi_min) > precision_deg and (lam_max - lam_min) > precision_deg and tries < max_iter:
        # 1. Grille de candidats
        lats = np.linspace(phi_min, phi_max, n)
        lons = np.linspace(lam_min, lam_max, n)
        dlat = (phi_max - phi_min) / n
        dlon = (lam_max - lam_min) / n
        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
        Np = grid_lat.size

        # 2. Temps de trajet modèles Tmod[p, i]
        Tmod = np.full((Np, Ns), np.nan, float)

        for j in range(Ns):
            lat_s, lon_s = stations[j]
            for p in range(Np):
                Ti = travel_time_seconds(
                    grid_lat.flat[p],
                    grid_lon.flat[p],
                    lat_s,
                    lon_s,
                    depth_fn,
                )
                Tmod[p, j] = np.nan if not np.isfinite(Ti) else float(Ti)

        # 3. Misfit en temps relatifs par rapport à la station ref_index
        #   ΔT_model[p, i] = Tmod[p, i] - Tmod[p, ref_index]
        #   ΔT_model[p, ref_index] = 0 par construction.
        # On ne garde que les stations i pour lesquelles les deux trajets (src->i, src->ref)
        # sont marins (Tmod finite).
        model_ok = np.isfinite(Tmod)
        ref_ok   = model_ok[:, ref_index][:, None]   # (Np,1)
        valid    = model_ok & ref_ok                 # (Np, Ns)

        # Temps relatifs modélisés
        T_ref = Tmod[:, ref_index][:, None]          # (Np,1)
        dt_model = Tmod - T_ref                      # (Np, Ns)

        # Résidus relatifs
        dt_obs_mat = np.broadcast_to(dt_obs, (Np, Ns))
        resid = dt_obs_mat - dt_model
        resid[~valid] = np.nan

        # Option robuste : clipping simple des outliers station par station
        if robust:
            # MAD (median absolute deviation) par candidat
            mad = np.nanmedian(np.abs(resid), axis=1)
            scale = np.maximum(mad, 1.0)  # évite division par zéro
            # Normalisation + clipping à ±3σ, puis re-mise à l'échelle
            resid = np.clip(resid / scale[:, None], -3.0, 3.0) * scale[:, None]

        # Misfit = RMS des résidus relatifs (en secondes)
        rmse = np.sqrt(np.nanmean(resid**2, axis=1))
        valid_counts = np.sum(valid, axis=1)

        # On met rmse=inf là où il n'y a pas assez de stations marines
        rmse = np.where(valid_counts >= min_valid_stations, rmse, np.inf)

        # 4. Sélection du meilleur candidat de cette grille
        idx_best = int(np.nanargmin(rmse))
        cand = dict(
            lat=float(grid_lat.flat[idx_best]),
            lon=float(grid_lon.flat[idx_best]),
            misfit=float(rmse[idx_best]),
            valid=int(valid_counts[idx_best]),
        )

        if cand["misfit"] < best["misfit"]:
            best.update(cand)

        if verbose:
            print(
                f"[Iter {tries+1}] best=({cand['lat']:.2f}, {cand['lon']:.2f})  "
                f"RMSE_rel={cand['misfit']:.2f}s  stations={cand['valid']}/{Ns}"
            )

        # 5. Raffinement autour du meilleur point global courant
        phi_min = best["lat"] - dlat
        phi_max = best["lat"] + dlat
        lam_min = best["lon"] - dlon
        lam_max = best["lon"] + dlon
        tries += 1

    best_lat = best["lat"]
    best_lon = best["lon"]

    # ------------------------------------------------------------------
    # 6. Diagnostics finaux à partir de la meilleure source trouvée
    #    - temps de trajet T_best[i]
    #    - estimation de t0* en absolu
    #    - résidus absolus et relatifs station par station
    # ------------------------------------------------------------------
    T_best = np.full(Ns, np.nan, float)
    for i, (lat_s, lon_s) in enumerate(stations):
        Ti = travel_time_seconds(
            best_lat, best_lon,
            lat_s, lon_s,
            depth_fn,
        )
        T_best[i] = np.nan if not np.isfinite(Ti) else float(Ti)

    # Masque de stations utilisables pour l'estimation de t0*
    mask_abs = np.isfinite(T_best) & np.isfinite(t_obs_s)
    if np.sum(mask_abs) >= 2:
        offsets = t_obs_s[mask_abs] - T_best[mask_abs]
        if robust:
            t0_hat = float(np.median(offsets))
        else:
            t0_hat = float(np.mean(offsets))
    else:
        t0_hat = np.nan

    # Résidus absolus : t_obs - (t0_hat + T_best)
    residuals_abs = t_obs_s - (t0_hat + T_best)
    residuals_abs[~mask_abs] = np.nan
    rmse_abs = float(np.sqrt(np.nanmean(residuals_abs**2))) if np.any(mask_abs) else np.nan

    # Résidus relatifs finaux (à titre de diag) au point optimal
    dt_model_best = T_best - T_best[ref_index]
    residuals_rel = dt_obs - dt_model_best
    mask_rel = np.isfinite(dt_model_best) & np.isfinite(dt_obs)
    residuals_rel[~mask_rel] = np.nan
    rmse_rel_final = float(np.sqrt(np.nanmean(residuals_rel[mask_rel]**2))) if np.any(mask_rel) else np.nan

    stats = {
        "rmse_rel": rmse_rel_final,
        "rmse_abs": rmse_abs,
        "residuals_rel": residuals_rel,
        "residuals_abs": residuals_abs,
        "T_model": T_best,
        "dt_obs": dt_obs,
        "dt_model": dt_model_best,
        "ref_index": int(ref_index),
        "valid": int(np.sum(mask_abs)),
    }

    return best_lat, best_lon, t0_hat, stats
