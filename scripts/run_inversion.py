import numpy as np
import matplotlib.pyplot as plt

from tsunami.io_etopo import load_etopo5, make_depth_function
from tsunami.observations import load_arrival_times
from tsunami.inverse import triangulation_inversion
from tsunami.speed_integrator import travel_time_seconds

if __name__ == "__main__":
    # === 1) Load bathymetry ===
    print("→ Loading ETOPO5 grid...")
    lats, lons, H, _ = load_etopo5("data/etopo5.grd")
    depth_fn = make_depth_function(lats, lons, H, order=1)

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
    best_lat, best_lon, t0, best_misfit = triangulation_inversion(
        stations=stations,
        t_obs_s=t_obs_s,
        depth_fn=depth_fn,
        phi_min=lat_min,
        phi_max=lat_max,
        lam_min=lon_min,
        lam_max=lon_max,
        verbose=True
    )