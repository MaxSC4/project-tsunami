import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter

from tsunami.io_etopo import load_etopo5

def _wrap_to_180(lons):
    """Convertit une grille de longitudes [0, 360) en [-180, 180) et renvoie l'ordre de réindexation."""
    lons_wrapped = ((lons + 180.0) % 360.0) - 180.0
    order = np.argsort(lons_wrapped)  # longitudes croissantes
    return lons_wrapped[order], order

def _roll_grid_for_wrapped_lons(H, order):
    """Réordonne les colonnes de H selon l'ordre de longitudes (après wrap)."""
    return H[:, order]

def _read_stations_csv(csv_path):
    """Lit un fichier de stations au format:
       Ville/Port  Latitude_N  Longitude_E  date_arrivee  heure_arrivee
       (séparé par espaces). Renvoie une liste de dicts {name, lat, lon}."""
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


def plot_world_etopo_with_stations(etopo_path="data/etopo5.grd",
                                   stations_csv="data/data_villes.csv",
                                   figsize=(12, 6),
                                   savepath="outputs/world_map.png",
                                   title="Global bathymetry (oceans) with stations"):
    """
    Carte ETOPO5 recadrée en [-180,180), continents grisés, océan coloré.
    - Colorbar en profondeur positive (m).
    """
    # 1) Charger la grille
    lats, lons, H, _ = load_etopo5(etopo_path)  # H: [nlat, nlon], lats/lons croissants (selon ton loader)

    # 2) Wrap longitudes en [-180,180) et réordonner H
    lons_wrapped, order = _wrap_to_180(lons)
    H_wrapped = _roll_grid_for_wrapped_lons(H, order)

    # 3) Orientation (si la grille source est N->S, on garde le flip ici)
    H_plot = np.flipud(H_wrapped)   # garde la carte orientée N en haut
    lat_min, lat_max = lats[0], lats[-1]
    lon_min, lon_max = lons_wrapped[0], lons_wrapped[-1]

    # 4) Masques terre/océan sur la matrice affichée
    land_mask  = (H_plot >= 0.0)
    ocean_mask = ~land_mask

    # Couche TERRE: gris clair (NaN ailleurs)
    land_img = np.where(land_mask, 1.0, np.nan)

    # Couche OCÉAN: on ne garde que H<0 (négatif), NaN ailleurs
    ocean = np.where(ocean_mask, H_plot, np.nan)

    # 5) Échelle de l'océan: bornes robustes (en négatif), colorbar en positif
    #   vmin <= 0 <= vmax ; ici on borne entre un percentile océan et 0
    if np.all(np.isnan(ocean)):
        raise ValueError("No ocean pixels detected in H_plot; check ETOPO data.")
    vmin = np.nanpercentile(ocean, 5)   # ~ -6000 m typiquement
    vmax = 0.0

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # 6) Afficher TERRE (gris)
    land_cmap = ListedColormap(["0.8"])  # gris clair
    ax.imshow(
        land_img,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        interpolation="nearest",
        cmap=land_cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
        zorder=1,
    )

    # 7) Afficher OCÉAN (palette bleue inversée)
    im = ax.imshow(
        ocean,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        interpolation="nearest",
        cmap="Blues_r",
        vmin=vmin, vmax=vmax,
        aspect="auto",
        zorder=2,
    )

    # 8) Axes, titre, grille
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-80,  81, 20))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_title(title)

    # 9) Colorbar (profondeur positive)
    def depth_fmt(x, pos):
        return f"{int(abs(x))}"  # affiche |x| (m)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03, format=FuncFormatter(depth_fmt))
    cb.set_label("Ocean depth (m)")

    # 10) Stations
    try:
        stations = _read_stations_csv(stations_csv)
        if stations:
            wrap_lon = lambda x: ((x + 180.0) % 360.0) - 180.0
            xs = np.array([wrap_lon(s["lon"]) for s in stations], float)
            ys = np.array([s["lat"] for s in stations], float)
            ax.scatter(xs, ys, s=36, c="k", edgecolors="white", linewidths=0.8, zorder=5, label="Stations")
            for s, x, y in zip(stations, xs, ys):
                label = s["name"] if s["name"] else ""
                if label:
                    ax.text(x + 1.2, y + 0.8, label, fontsize=8, color="k", zorder=6)
            ax.legend(loc="lower left", frameon=True)
    except FileNotFoundError:
        pass

    # 11) Sauvegarde + affichage
    import os
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

    return fig, ax


if __name__ == "__main__":
    plot_world_etopo_with_stations(
        etopo_path="data/etopo5.grd",
        stations_csv="data/data_villes.csv",
        figsize=(12, 6),
        savepath="outputs/world_map.png",
        title="Project Tsunami — ETOPO5 & Stations"
    )
