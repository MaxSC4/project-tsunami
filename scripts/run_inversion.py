"""
Pipeline principal d'inversion.

Ce script :
    1) charge la bathymétrie ETOPO,
    2) lit les stations et leurs temps d'arrivée observés,
    3) exécute l'inversion robuste pour estimer la source (lat, lon, t0*),
    4) affiche/sauvegarde une carte avec la source, les stations et les trajets.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tsunami.io_etopo import load_etopo5, make_depth_function
from tsunami.observations import load_arrival_times
from tsunami.speed_integrator import travel_time_seconds

from tsunami.inverse import triangulation_inversion
from tsunami.absolute_inversion import absolute_inversion

from plotting.world_map import plot_world_map
from plotting.diagnostics import plot_obs_vs_model
from plotting.uncertainty import (
    estimate_spatial_uncertainty,
    compute_misfit_profiles,
    plot_misfit_profiles
)
from plotting.table_arrival_times import build_residual_table, print_latex_table


# --------------------------------------------------------------
# 1) Petites fonctions utilitaires
# --------------------------------------------------------------

def _ensure_dir(path):
    """Crée le dossier parent si besoin."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _to_station_list(df):
    """
    Convertit le DataFrame d'observations en liste de stations
    sous la forme [{"name","lat","lon"}, ...] pour l'affichage.
    S'adapte aux noms de colonnes usuels du projet.
    """
    name_col = "Ville/Port" if "Ville/Port" in df.columns else (
        "name" if "name" in df.columns else None
    )
    lat_col = "Latitude_N" if "Latitude_N" in df.columns else (
        "lat" if "lat" in df.columns else None
    )
    lon_col = "Longitude_E" if "Longitude_E" in df.columns else (
        "lon" if "lon" in df.columns else None
    )

    if not (name_col and lat_col and lon_col):
        raise ValueError("Colonnes attendues non trouvées (Ville/Port, Latitude_N, Longitude_E).")

    recs = []
    for _, row in df.iterrows():
        recs.append({
            "name": str(row[name_col]).replace("_", " "),
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
        })
    return recs


def _stations_array(df):
    """
    Retourne un tableau (Ns,2) [lat, lon] pour l'inversion.
    """
    lat_col = "Latitude_N" if "Latitude_N" in df.columns else "lat"
    lon_col = "Longitude_E" if "Longitude_E" in df.columns else "lon"
    return df[[lat_col, lon_col]].to_numpy(dtype=float)


def _auto_search_box(stations, margin_deg=10.0):
    """
    Calcule une boîte de recherche simple autour des stations,
    avec un 'margin_deg' de sécurité (degrés).
    """
    lats = stations[:, 0]
    lons = stations[:, 1]
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))

    lat_min = max(-90.0, lat_min - margin_deg)
    lat_max = min(90.0,  lat_max + margin_deg)
    lon_min = lon_min - margin_deg
    lon_max = lon_max + margin_deg
    return lat_min, lat_max, lon_min, lon_max


# --------------------------------------------------------------
# 2) Pipeline principal (avec paramètres par défaut)
# --------------------------------------------------------------

def run_pipeline(
    etopo_path="data/etopo5.grd",
    stations_csv="data/data_villes.csv",
    lon_mode="180",                 # '180' ([-180,180)) ou '360' ([0,360))
    search_box=None,                # (lat_min, lat_max, lon_min, lon_max) ou None → auto
    grid_n=15,                      # résolution initiale de la grille d'inversion
    precision_deg=0.2,              # critère d'arrêt (taille de boîte < precision_deg)
    max_iter=6,                     # itérations max de raffinement
    robust=True,                    # inversion robuste (médiane, clipping outliers)
    make_map=True,                  # tracer la carte finale
    save_map_path="outputs/wo_bowen/world_map_inversion.png",
    make_diagnostics=False,
    diag_path="outputs/wo_bowen/obs_vs_model.png",
    use_absolute=True,              # True = absolute_inversion, False = triangulation_inversion
):
    """
    Exécute le pipeline d'inversion et (optionnellement) produit une carte.
    Retourne un dict de résultats (lat, lon, t0, misfit, etc.)
    """
    # --- 1) Bathymétrie ---
    print("→ Loading ETOPO grid...")
    lats, lons, H, _ = load_etopo5(etopo_path)
    depth_fn = make_depth_function(lats, lons, H, order=1)

    # --- 2) Observations ---
    print("→ Loading stations & observed arrival times...")
    df, t_obs_s = load_arrival_times(stations_csv)
    stations = _stations_array(df)
    station_list = _to_station_list(df)
    Ns = len(stations)
    print(f"   → {Ns} stations loaded.")

    # --- 3) Boîte de recherche ---
    if search_box is None:
        lat_min, lat_max, lon_min, lon_max = _auto_search_box(stations, margin_deg=10.0)
        print(f"→ Auto search box: {lat_min:.1f}–{lat_max:.1f}°N / {lon_min:.1f}–{lon_max:.1f}°E")
    else:
        lat_min, lat_max, lon_min, lon_max = map(float, search_box)
        print(f"→ User search box: {lat_min:.1f}–{lat_max:.1f}°N / {lon_min:.1f}–{lon_max:.1f}°E")

    # --- 3bis) Trouver l’index de la station de référence (Los Angeles) ---
    name_col = "Ville/Port" if "Ville/Port" in df.columns else "name"
    names_raw = df[name_col].astype(str)

    ref_index = 0
    try:
        ref_index = names_raw.tolist().index("Los_Angeles")
    except ValueError:
        try:
            ref_index = names_raw.str.replace("_", " ").tolist().index("Los Angeles")
        except ValueError:
            print("[run_pipeline] Warning: 'Los_Angeles' non trouvée, station[0] utilisée comme référence.")
            ref_index = 0

    # --- 4) Inversion ---
    print("→ Running inversion...")

    if use_absolute:
        # nouvelle inversion “centrée Los Angeles”
        best_lat, best_lon, t0_hat, stats = absolute_inversion(
            stations=stations,
            t_obs_s=t_obs_s,
            depth_fn=depth_fn,
            phi_min=lat_min, phi_max=lat_max,
            lam_min=lon_min, lam_max=lon_max,
            ref_index=ref_index,
            n=grid_n,
            precision_deg=precision_deg,
            max_iter=max_iter,
            min_valid_stations=10,
            robust=robust,
            verbose=True,
        )
    else:
        # ancienne inversion triangulation (pour comparaison)
        best_lat, best_lon, t0_hat, stats = triangulation_inversion(
            stations=stations,
            t_obs_s=t_obs_s,
            depth_fn=depth_fn,
            phi_min=lat_min, phi_max=lat_max,
            lam_min=lon_min, lam_max=lon_max,
            n=grid_n,
            precision_deg=precision_deg,
            max_iter=max_iter,
            min_valid_stations=10,
            robust=robust,
            verbose=True,
        )

    print("\n=== Inversion result ===")
    print(f"Source latitude   : {best_lat:.3f}°")
    print(f"Source longitude  : {best_lon:.3f}°")

    # Conversion du temps POSIX (secondes) en date UTC
    try:
        t0_datetime = datetime.datetime.fromtimestamp(t0_hat, tz=datetime.timezone.utc)
        t0_str = t0_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"Estimated t0*     : {t0_hat:.1f} s (POSIX time)")
        print(f"                   → {t0_str}")
    except (OSError, OverflowError, ValueError):
        print(f"Estimated t0*     : {t0_hat:.1f} s (not a valid POSIX timestamp)")

    # Diagnostic misfit
    if "rmse_rel" in stats:
        print(f"RMS misfit (relative) : {stats['rmse_rel']:.2f} s")
    if "rmse_abs" in stats:
        print(f"RMS misfit (absolute) : {stats['rmse_abs']:.2f} s")
    if "valid" in stats:
        print(f"Valid stations        : {stats['valid']} / {Ns}")

    # --- 4bis) Estimation de l'incertitude spatiale ---
    print("→ Estimating spatial uncertainty around source...")
    unc = estimate_spatial_uncertainty(
        best_lat, best_lon,
        stations=stations,
        t_obs_s=t_obs_s,
        depth_fn=depth_fn,
        span_lat_deg=4.0,
        span_lon_deg=6.0,
        step_deg=0.25,
        rel_increase=0.10,
        n_samples=2000,
        h_min=0.0,
        robust=True,
    )

    dlat_deg = unc["dlat_deg"]
    dlon_deg = unc["dlon_deg"]
    radius_km = unc["radius_km"]

    print(f"Lat uncertainty   : Δφ ≈ {dlat_deg:.2f}°")
    print(f"Lon uncertainty   : Δλ ≈ {dlon_deg:.2f}°")
    print(f"Spatial radius    : ≈ {radius_km:.0f} km (effective)")
    print(f"Local RMSE min ≈ {unc['rmse_min']:.1f} s")

    print("→ Computing 1D misfit profiles (lat/lon)...")
    profiles = compute_misfit_profiles(
        best_lat, best_lon,
        stations=stations,
        t_obs_s=t_obs_s,
        depth_fn=depth_fn,
        span_lat_deg=4.0,
        span_lon_deg=6.0,
        step_deg=0.25,
        rel_increase=0.10,
        n_samples=2000,
        h_min=0.0,
        robust=True,
    )

    plot_misfit_profiles(
        best_lat, best_lon,
        profiles,
        title="1D RMS misfit profiles around estimated source",
        savepath="outputs/wo_bowen/misfit_profiles.png",
        show=True,
    )

    # --- 5) Carte ---
    if make_map:
        print("→ Rendering world map...")
        _ensure_dir(save_map_path)

        # Calcul des temps de trajet modélisés depuis la source optimale
        T_model = []
        for lat_s, lon_s in stations:
            T_model.append(
                travel_time_seconds(
                    best_lat, best_lon,
                    lat_s, lon_s,
                    depth_fn,
                    n_samples=2000,
                    h_min=0.0,
                )
            )
        T_model = np.array(T_model, float)

        source = {
            "lat": best_lat,
            "lon": best_lon,
            "label": "Estimated source",
            "radius_km": radius_km if np.isfinite(radius_km) else 20.0,
        }

        plot_world_map(
            etopo_path=etopo_path,
            stations=station_list,   # liste de dicts "name/lat/lon"
            source=source,
            lon_mode=lon_mode,
            title="ETOPO — Stations & great-circle paths",
            savepath=save_map_path,
            show=True,
            station_travel_times_s=T_model
        )

    # --- 6) Figure diag t_obs vs T_mod (facultative) ---
    if make_diagnostics:
        station_names = list(names_raw.str.replace("_", " "))
        print("→ Making diagnostics figure (observed vs modelled)...")
        _ = plot_obs_vs_model(
            t_obs_s=t_obs_s,
            stations=stations,
            src_lat=best_lat, src_lon=best_lon,
            depth_fn=depth_fn,
            station_names=station_names,
            robust=True,
            n_samples=2000, h_min=0.0,
            show_free_fit=True,
            figpath=diag_path,
            title="Observed vs modelled arrival times",
            show=True,
        )

    # --- 7) Tableau LaTeX des temps obs/mod + résidus ---
    df["t_obs_s"] = t_obs_s
    table = build_residual_table(df, T_model, output_csv="outputs/wo_bowen/residuals_table.csv")
    print_latex_table(table)

    # Résultats pour réutilisation
    return {
        "lat": best_lat,
        "lon": best_lon,
        "t0": t0_hat,
        "misfit": stats.get("rmse_abs", np.nan),
        "stats": stats,
    }


# --------------------------------------------------------------
# 3) Exécutable direct
# --------------------------------------------------------------

if __name__ == "__main__":
    results = run_pipeline(
        etopo_path="data/etopo5.grd",
        stations_csv="data/data_villes.csv",
        lon_mode="360",
        search_box=(-60.0, 60.0, 100.0, 290.0),
        grid_n=15,
        precision_deg=0.2,
        max_iter=6,
        robust=True,
        make_map=True,
        save_map_path="outputs/new_plot/world_map_inversion.png",
        make_diagnostics=True,
        diag_path="outputs/new_plot/obs_vs_model.png",
        use_absolute=True,
    )

    print("\nDone.")
