import numpy as np
import matplotlib.pyplot as plt

from tsunami.io_etopo import load_etopo5, make_depth_map
from tsunami.observations import load_arrival_times
from tsunami.inverse import triangulation_inversion
from tsunami.speed_integrator import travel_time_seconds

if __name__ == "__main__":
    # === 1) Load bathymetry ===
    print("→ Loading ETOPO5 grid...")
    lats, lons, H, _ = load_etopo5("data/etopo5.grd")
    depth_fn = make_depth_map(lats, lons, H, order=1)

    # === 2) Load observation stations ===
    print("→ Loading observation stations...")
    df, t_obs_s = load_arrival_times("data/data_villes.csv")
    stations = df[["Latitude_N", "Longitude_E"]].to_numpy()

    # Normalize longitudes [0, 360)
    #stations[:, 1] = (stations[:, 1] + 360) % 360
    #print(f"   → {len(stations)} stations loaded.")

    # === 3) Define search region around Japan ===
    lat_min, lat_max = 0, 40
    lon_min, lon_max = 130, 160
    print(f"→ Grid search in {lat_min}–{lat_max}°N / {lon_min}–{lon_max}°E")

    # === 4) Compute relative arrival time deltas ===
    #delta_d = t_obs_s.copy()

    # === 5) Run inversion ===
    print("→ Running triangulation inversion...")
    best_lat, best_lon, best_misfit = triangulation_inversion(
        stations=stations,
        t_obs_s=t_obs_s,
        depth_fn=depth_fn,
        phi_min=lat_min,
        phi_max=lat_max,
        lam_min=lon_min,
        lam_max=lon_max,
        n=12,
        precision_deg=0.01,  # 0.5° precision
        max_tries=6,
        verbose=True
    )

    # === 6) Print results ===
    best_lon_w = best_lon if best_lon <= 180 else best_lon - 360
    print("\n=== Best candidate ===")
    print(f"Latitude  : {best_lat:.2f}° N")
    print(f"Longitude : {best_lon:.2f}° E  ({best_lon_w:.2f}° W)")
    print(f"Misfit (sum of squared ΔΔt): {best_misfit}")

    def eval_best_candidate(best_lat, best_lon, stations, t_obs_s, depth_fn):
        Tmod = []
        valid = []
        for (lat_s, lon_s) in stations:
            Tij = travel_time_seconds(best_lat, best_lon, lat_s, lon_s, depth_fn)
            Tmod.append(Tij)
            valid.append(np.isfinite(Tij))
        Tmod = np.array(Tmod, float)
        valid = np.array(valid, bool)

        # t0* = moyenne (t_obs - Tmod) sur stations valides
        diff = t_obs_s[valid] - Tmod[valid]
        t0_hat = np.mean(diff)

        resid = t_obs_s[valid] - (t0_hat + Tmod[valid])
        N_eff = resid.size
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae  = float(np.mean(np.abs(resid)))
        return t0_hat, resid, rmse, mae, N_eff

    t0_hat, resid, rmse, mae, N_eff = eval_best_candidate(best_lat, best_lon, stations, t_obs_s, depth_fn)
    print(f"N_eff={N_eff}, RMSE={rmse:.1f}s ({rmse/60:.2f} min), MAE={mae:.1f}s ({mae/60:.2f} min)")
    print("Residuals per station (s):", np.round(resid, 1))
    print(f"Estimated origin time offset t0* = {t0_hat:.1f} s")