import numpy as np
from tsunami.speed_integrator import travel_time_seconds

def triangulation_inversion(
    stations,            # (Ns,2) lat, lon en degrés
    t_obs_s,             # (Ns,) temps d'arrivée observés en secondes (absolus, p.ex. depuis la 1ère arrivée)
    depth_fn,            # depth(lat, lon) -> profondeur (m), négative en mer
    phi_min, phi_max,    # bornes latitude (°)
    lam_min, lam_max,    # bornes longitude (°)
    n=15,                # résolution de la grille par axe
    precision_deg=0.1,   # seuil d'arrêt (°)
    max_tries=7,         # nb max d’itérations de raffinement
    min_valid_stations=3,# nb minimal de stations valides pour accepter un candidat
    verbose=True,
):
    """
    Inversion (grille + raffinement) avec misfit en SECONDES PHYSIQUES.
    Pour chaque candidat, on estime t0* = mean_i( t_obs[i] - T_model[i] ) sur les stations valides,
    puis on calcule la somme des carrés des résidus absolus : sum_i( t_obs[i] - (t0* + T_model[i]) )^2.
    """
    stations = np.asarray(stations, dtype=float)
    t_obs_s  = np.asarray(t_obs_s,  dtype=float)

    tries = 0
    best_lat, best_lon, best_misfit = np.nan, np.nan, np.inf

    Ns = stations.shape[0]
    if Ns < 2:
        raise ValueError("Need at least two stations.")

    while (phi_max - phi_min) > precision_deg and tries < max_tries:
        # --- 1) Grille courante ---
        lats = np.linspace(phi_min, phi_max, n)
        lons = np.linspace(lam_min, lam_max, n)
        dphi = (phi_max - phi_min) / n
        dlam = (lam_max - lam_min) / n
        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")

        Np = grid_lat.size
        Tmod = np.full((Np, Ns), np.nan, dtype=float)

        # --- 2) Temps de trajet modèle pour (candidat p, station j) ---
        for j in range(Ns):
            lat_s, lon_s = stations[j]
            for p in range(Np):
                lat_c = float(grid_lat.flat[p])
                lon_c = float(grid_lon.flat[p])
                Tij = travel_time_seconds(lat_c, lon_c, lat_s, lon_s, depth_fn)
                Tmod[p, j] = np.nan if not np.isfinite(Tij) else Tij

        # --- 3) Misfit absolu avec estimation de t0* par candidat ---
        # Diff = t_obs - Tmod ; on met NaN là où Tmod est NaN ou t_obs non défini
        obs_mat  = np.broadcast_to(t_obs_s, (Np, Ns))
        model_ok = np.isfinite(Tmod)
        obs_ok   = np.isfinite(obs_mat)
        mask     = model_ok & obs_ok

        diff = np.full_like(Tmod, np.nan, dtype=float)
        diff[mask] = obs_mat[mask] - Tmod[mask]   # (Np, Ns)

        # t0* = nanmean(diff, axis=1)
        t0_hat = np.nanmean(diff, axis=1)         # (Np,)

        # résidus: r = t_obs - (Tmod + t0*)
        resid = obs_mat - (Tmod + t0_hat[:, None])  # (Np, Ns)
        # on ignore les colonnes invalides
        resid[~mask] = np.nan

        # misfit = somme des carrés (en secondes^2) sur stations valides
        misfit = np.nansum(resid**2, axis=1)        # (Np,)
        # rejeter les candidats avec trop peu de stations valides
        valid_counts = np.sum(mask, axis=1)
        misfit = np.where(valid_counts >= min_valid_stations, misfit, np.inf)

        # --- 4) Sélection + raffinement ---
        idx_min = int(np.argmin(misfit))
        cand_lat = float(grid_lat.flat[idx_min])
        cand_lon = float(grid_lon.flat[idx_min])
        cand_mis = float(misfit[idx_min])

        if cand_mis < best_misfit:
            best_lat, best_lon, best_misfit = cand_lat, cand_lon, cand_mis

        if verbose:
            print(f"[Iter {tries+1}] best=({cand_lat:.2f}, {cand_lon:.2f})  "
                  f"misfit={cand_mis:.3e}  valid_stations={int(valid_counts[idx_min])}/{Ns}")

        # Raffinement autour du meilleur point courant
        lam_min = cand_lon - dlam
        lam_max = cand_lon + dlam
        phi_min = cand_lat - dphi
        phi_max = cand_lat + dphi
        tries += 1

    return best_lat, best_lon, best_misfit
