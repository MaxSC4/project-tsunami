import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter

from tsunami.io_etopo import load_etopo5
from tsunami.geo import great_circle_points

# ---------------------- utils longitudes/latitudes ----------------------

def _lon_to_180(lon):
    """scalaire -> [-180,180)"""
    out = ((float(lon) + 180.0) % 360.0) - 180.0
    return -180.0 if out == 180.0 else out

def _lon_to_360(lon):
    """scalaire -> [0,360)"""
    out = float(lon) % 360.0
    return out + 360.0 if out < 0 else out

def _arr_lon_to_180(arr):
    arr = np.asarray(arr, dtype=float)
    out = ((arr + 180.0) % 360.0) - 180.0
    out[out == 180.0] = -180.0
    return out

def _arr_lon_to_360(arr):
    arr = np.asarray(arr, dtype=float) % 360.0
    out = np.where(arr < 0, arr + 360.0, arr)
    return out

def _flip_lat_for_imshow(lat_vals, lat_min, lat_max):
    """Si la matrice n'est pas retournée mais on utilise origin='lower', il faut inverser Y pour les overlays."""
    return lat_max + lat_min - lat_vals

# ---------------------- stations ----------------------

def _read_stations_csv(csv_path):
    """Lit un fichier de stations (colonnes: Ville/Port, Latitude_N, Longitude_E, ...)."""
    import pandas as pd
    df = pd.read_csv(csv_path, sep=r"\s+", engine="python")
    colmap = {"Ville/Port": "name", "Latitude_N": "lat", "Longitude_E": "lon"}
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    required = {"name", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in stations CSV: {missing}")
    df["name"] = df["name"].astype(str).str.replace("_", " ")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df[["name", "lat", "lon"]].to_dict("records")

# ---------------------- tracés : path & source ----------------------

def _split_on_boundary(x, boundary):
    """
    Découpe les segments lorsqu'on traverse un bord:
      - si boundary == 180 → teste |diff| > 180 (cas [-180,180))
      - si boundary == 0   → teste |diff| > 180 (cas [0,360))
    """
    d = np.diff(x)
    # traversée "longue" -> on split
    jumps = np.abs(d) > 180.0
    idx = np.where(jumps)[0] + 1
    parts = np.split(np.arange(len(x)), idx)
    return parts

def _plot_gc_path(ax, lat1, lon1, lat2, lon2, npts,
                  lat_min, lat_max, lon_mode="180",
                  flip_lat=False, **style):
    """
    Trace un grand cercle en gérant:
      - le repère lon_mode: '180' ([-180,180)) ou '360' ([0,360))
      - la découpe au bord (±180 ou 0/360)
      - le flip Y pour coller à imshow(origin='lower') si nécessaire
    """
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=npts)
    lats = pts[:, 0]
    lons = pts[:, 1]

    if lon_mode == "180":
        lons = _arr_lon_to_180(lons)
        boundary = 180.0
    else:
        lons = _arr_lon_to_360(lons)
        boundary = 0.0  # conceptuel; on découpe quand saut > 180

    if flip_lat:
        lats = _flip_lat_for_imshow(lats, lat_min, lat_max)

    for seg in _split_on_boundary(lons, boundary):
        ax.plot(lons[seg], lats[seg], **style)

def _plot_source(ax, lat, lon, lat_min, lat_max,
                 lon_mode="180", label=None, radius_km=None, flip_lat=False, style=None, zorder=8):
    """Marque la source estimée (+ cercle d’incertitude optionnel)."""
    if lat is None or lon is None:
        return
    st = {"marker": "*", "s": 170, "c": "gold", "edgecolors": "k", "linewidths": 0.9}
    if style:
        st.update(style)

    x = _lon_to_180(lon) if lon_mode == "180" else _lon_to_360(lon)
    y = float(lat)
    if flip_lat:
        y = _flip_lat_for_imshow(y, lat_min, lat_max)

    ax.scatter([x], [y], zorder=zorder, **st)
    if label:
        ax.text(x + 1.0, y + 0.8, label, fontsize=9, weight="bold", color="k", zorder=zorder+1)

    if radius_km and radius_km > 0:
        Rdeg = radius_km / 111.0
        theta = np.linspace(0, 2*np.pi, 361)
        lat_c = float(lat) + Rdeg * np.sin(theta)
        coslat = max(np.cos(np.deg2rad(float(lat))), 1e-6)
        lon_c = float(lon) + (Rdeg / coslat) * np.cos(theta)
        if lon_mode == "180":
            lon_c = _arr_lon_to_180(lon_c)
        else:
            lon_c = _arr_lon_to_360(lon_c)
        if flip_lat:
            lat_c = _flip_lat_for_imshow(lat_c, lat_min, lat_max)
        ax.plot(lon_c, lat_c, color=st.get("c", "gold"), alpha=0.7, lw=1.2, zorder=zorder-1)

# ---------------------- carte principale ----------------------

def plot_world_etopo_with_stations(etopo_path="data/etopo5.grd",
                                   stations_csv="data/data_villes.csv",
                                   figsize=(12, 6),
                                   savepath="outputs/world_map.png",
                                   title="Global bathymetry (oceans) with stations",
                                   paths=None,
                                   path_style=None,
                                   source=None,
                                   lon_mode="180"):   # '180' → [-180,180), '360' → [0,360) (Pacifique centré)
    """
    - lon_mode='180' : carte centrée sur Greenwich (par défaut)
    - lon_mode='360' : carte centrée Pacifique (0..360), évite les effets de dateline
    - paths: [{"lat1","lon1","lat2","lon2","npts","style":{...}}, ...]
    - source: {"lat":..., "lon":..., "label":..., "radius_km":..., "style":{...}}
    """
    # 1) Charger la grille
    lats, lons, H, _ = load_etopo5(etopo_path)

    # 2) Construire l'axe longitude et ré-ordonner H selon lon_mode
    if lon_mode == "180":
        # on travaille en [-180,180)
        lons_wrapped = _arr_lon_to_180(lons)
        order = np.argsort(lons_wrapped)
        xlon = lons_wrapped[order]
        Hx = H[:, order]
        x_min, x_max = -180.0, 180.0
    else:
        # on travaille en [0,360)
        lons_wrapped = _arr_lon_to_360(lons)
        order = np.argsort(lons_wrapped)
        xlon = lons_wrapped[order]
        Hx = H[:, order]
        x_min, x_max = 0.0, 360.0

    # 3) Orientation: H n'est pas retournée; avec imshow(origin='lower'), on doit flipper les overlays
    H_plot = Hx
    lat_min, lat_max = lats[0], lats[-1]
    flip_lat = False  # IMPORTANT: pour stations/paths/source

    # 4) Masques terre/océan
    land_mask  = (H_plot >= 0.0)
    ocean_mask = ~land_mask
    land_img = np.where(land_mask, 1.0, np.nan)
    ocean = np.where(ocean_mask, H_plot, np.nan)

    # 5) Bornes océan
    if np.all(np.isnan(ocean)):
        raise ValueError("No ocean pixels detected in H_plot; check ETOPO data.")
    vmin = np.nanpercentile(ocean, 5)
    vmax = 0.0

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # 6) Terre en gris
    land_cmap = ListedColormap(["0.8"])
    ax.imshow(
        land_img,
        extent=[x_min, x_max, lat_min, lat_max],
        origin="lower",
        interpolation="nearest",
        cmap=land_cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
        zorder=1,
    )

    # 7) Océan
    im = ax.imshow(
        ocean,
        extent=[x_min, x_max, lat_min, lat_max],
        origin="lower",
        interpolation="nearest",
        cmap="Blues_r",
        vmin=vmin, vmax=vmax,
        aspect="auto",
        zorder=2,
    )

    # 8) Axes, titre
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    if lon_mode == "180":
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_xlim(-180, 180)
    else:
        ax.set_xticks(np.arange(0, 361, 60))
        ax.set_xlim(0, 360)
    ax.set_yticks(np.arange(-80, 81, 20))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_title(title)

    # 9) Colorbar
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03,
                      format=FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
    cb.set_label("Ocean depth (m)")

    # 10) Stations
    try:
        stations = _read_stations_csv(stations_csv)
        if stations:
            xs = np.array([(_lon_to_180(s["lon"]) if lon_mode=="180" else _lon_to_360(s["lon"]))
                           for s in stations], float)
            ys = np.array([s["lat"] for s in stations], float)
            if flip_lat:
                ys = _flip_lat_for_imshow(ys, lat_min, lat_max)
            ax.scatter(xs, ys, s=36, c="k", edgecolors="white", linewidths=0.8, zorder=5, label="Stations")
            for s, x, y in zip(stations, xs, ys):
                label = s["name"] if s["name"] else ""
                if label:
                    ax.text(x + 1.2, y + 0.8, label, fontsize=8, color="k", zorder=6)
            ax.legend(loc="lower left", frameon=True)
    except FileNotFoundError:
        pass

    # 11) Paths (grands cercles)
    if paths:
        default_style = {"color": "red", "lw": 2, "alpha": 0.9}
        if path_style:
            default_style |= path_style
        for p in paths:
            s = default_style | p.get("style", {})
            npts = p.get("npts", 1001)
            _plot_gc_path(
                ax,
                p["lat1"], p["lon1"], p["lat2"], p["lon2"],
                npts=npts,
                lat_min=lat_min, lat_max=lat_max,
                lon_mode=lon_mode,
                flip_lat=flip_lat,
                **s
            )

    # 12) Source estimée (optionnelle)
    if source:
        _plot_source(
            ax,
            lat=source.get("lat"),
            lon=source.get("lon"),
            label=source.get("label", "Estimated source"),
            radius_km=source.get("radius_km"),
            style=source.get("style"),
            lat_min=lat_min, lat_max=lat_max,
            lon_mode=lon_mode,
            flip_lat=flip_lat,
            zorder=8
        )

    # 13) Sauvegarde + affichage
    import os
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig, ax

"""
plot_world_etopo_with_stations(
    etopo_path="data/etopo5.grd",
    stations_csv="data/data_villes.csv",
    savepath="outputs/world_map.png",
    title="ETOPO5 — Stations & Path",
    lon_mode="180",           # [-180, 180)
    paths=[{"lat1":41.7878,"lon1":140.7090,"lat2":34.05,"lon2":-118.25,
            "npts":1001,"style":{"color":"crimson","lw":2.5}}],
    source={"lat":24.89,"lon":151.40,"label":"Estimated source","radius_km":200}
)
"""

plot_world_etopo_with_stations(
    etopo_path="data/etopo5.grd",
    stations_csv="data/data_villes.csv",
    savepath="outputs/world_map_pacific.png",
    title="ETOPO5 — Pacific-centered",
    lon_mode="360",           # [0, 360)
    #paths=[{"lat1":41.7878,"lon1":140.7090,"lat2":34.05,"lon2":-118.25,
    #        "npts":1201,"style":{"color":"crimson","lw":2.5}}],
    source={"lat":41.03,"lon":162.87,"label":"Estimated source","radius_km":200}
)
