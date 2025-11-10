"""
Carte globale simple pour visualiser :
    - la bathymétrie (ETOPO) avec une palette claire,
    - une source de tsunami (marqueur et cercle d'incertitude optionnel),
    - les trajets (grands cercles) de la source vers les stations,
    - les stations avec leurs noms et une légende.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter

from tsunami.io_etopo import load_etopo5
from tsunami.geo import great_circle_points


# ---------------------------------------------------------------------
# 1) Petites fonctions utilitaires
# ---------------------------------------------------------------------

def _wrap180(lon):
    """Met une longitude dans l'intervalle [-180, 180)."""
    out = (np.asarray(lon, float) + 180.0) % 360.0 - 180.0
    if np.isscalar(lon):
        return float(-180.0 if out == 180.0 else out)
    out[out == 180.0] = -180.0
    return out

def _wrap360(lon):
    """Met une longitude dans l'intervalle [0, 360)."""
    out = np.asarray(lon, float) % 360.0
    if np.isscalar(lon):
        return float(out + 360.0 if out < 0 else out)
    out = np.where(out < 0, out + 360.0, out)
    return out

def _split_on_dateline(lons):
    """
    Découpe une polyligne quand le saut de longitude dépasse 180°.
    Renvoie une liste d'indices de segments (pour tracer sans traverser la carte).
    """
    lons = np.asarray(lons, float)
    if lons.size == 0:
        return []
    jumps = np.abs(np.diff(lons)) > 180.0
    idx = np.where(jumps)[0] + 1
    parts = np.split(np.arange(lons.size), idx)
    return parts

def _read_stations_csv(csv_path):
    """
    Lit un CSV de stations minimal (colonnes au moins: name, lat, lon).
    Si ton fichier a d'autres noms de colonnes, adapte le mapping ci-dessous.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, sep=r"\s+|,|;", engine="python")
    # Mapping souple pour s'adapter au fichier fourni dans data_villes.csv
    mapping = {
        "Ville/Port": "name",
        "Latitude_N": "lat",
        "Longitude_E": "lon",
        "name": "name",
        "lat": "lat",
        "lon": "lon",
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    for col in ("name", "lat", "lon"):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV: {col}")
    df["name"] = df["name"].astype(str).str.replace("_", " ")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    # Sortie: liste de dicts {"name":..., "lat":..., "lon":...}
    return df[["name", "lat", "lon"]].to_dict("records")


# ---------------------------------------------------------------------
# 2) Tracés : grands cercles, source, stations
# ---------------------------------------------------------------------

def _plot_great_circle(ax, lat1, lon1, lat2, lon2, npts=1001, lon_mode="180", line_kwargs=None):
    """
    Trace le grand cercle entre (lat1, lon1) et (lat2, lon2).
    Gère le wrap longitudes et coupe aux bords pour un rendu propre.
    """
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=npts)
    lats = pts[:, 0]
    lons = pts[:, 1]
    lons = _wrap180(lons) if lon_mode == "180" else _wrap360(lons)

    parts = _split_on_dateline(lons)
    style = {"lw": 2.0, "color": "crimson", "alpha": 0.9}
    if line_kwargs:
        style.update(line_kwargs)
    for seg in parts:
        ax.plot(lons[seg], lats[seg], **style)

def _plot_source(ax, lat, lon, lon_mode="180", label="Source", radius_km=None, style=None, zorder=8):
    """Affiche la source (étoile dorée) et un cercle d’incertitude optionnel."""
    if lat is None or lon is None:
        return
    x = _wrap180(lon) if lon_mode == "180" else _wrap360(lon)
    y = float(lat)

    st = {"marker": "*", "s": 180, "c": "gold", "edgecolors": "k", "linewidths": 0.9}
    if style:
        st.update(style)

    ax.scatter([x], [y], zorder=zorder, **st)
    if label:
        ax.text(x + 1.0, y + 0.8, str(label), fontsize=9, weight="bold", color="k", zorder=zorder+1)

    if radius_km and radius_km > 0:
        # Cercle approximatif (en degrés) autour de la source
        Rdeg = radius_km / 111.0  # ~conversion km→deg
        theta = np.linspace(0, 2*np.pi, 361)
        lat_c = float(lat) + Rdeg * np.sin(theta)
        coslat = max(np.cos(np.deg2rad(float(lat))), 1e-6)
        lon_c = float(lon) + (Rdeg / coslat) * np.cos(theta)
        lon_c = _wrap180(lon_c) if lon_mode == "180" else _wrap360(lon_c)
        ax.plot(lon_c, lat_c, color=st.get("c", "gold"), alpha=0.7, lw=1.2, zorder=zorder-1)

def _plot_stations(ax, stations, lon_mode="180", with_labels=True):
    """Place les stations, et affiche leurs noms si demandé."""
    xs = []
    ys = []
    names = []
    for s in stations:
        names.append(s.get("name", ""))
        ys.append(float(s["lat"]))
        xs.append(_wrap180(s["lon"]) if lon_mode == "180" else _wrap360(s["lon"]))
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ax.scatter(xs, ys, s=36, c="k", edgecolors="white", linewidths=0.8, zorder=5, label="Stations")
    if with_labels:
        for name, x, y in zip(names, xs, ys):
            if name:
                ax.text(x + 1.2, y + 0.8, name, fontsize=8, color="k", zorder=6)


# ---------------------------------------------------------------------
# 3) Carte principale (fonction unique, simple)
# ---------------------------------------------------------------------

def plot_world_map(
    etopo_path="data/etopo5.grd",
    stations=None,                # liste [{"name","lat","lon"}, ...] OU chemin CSV (str) OU None
    source=None,                  # dict ex: {"lat":..., "lon":..., "label":..., "radius_km":200}
    lon_mode="180",               # "180" → [-180,180) ; "360" → [0,360)
    figsize=(12, 6),
    title="Global bathymetry with stations",
    ocean_cmap="Blues_r",         # palette océan (foncé = profond)
    land_color="0.85",            # gris clair pour les terres
    show_colorbar=True,
    path_style=None,              # style par défaut des traits de grands cercles
    savepath=None,                # si non None → sauvegarde
    show=True                     # si True → plt.show()
):
    """
    Trace une carte globale simple et renvoie (fig, ax).
    - Si `stations` est une chaîne, on suppose un CSV à lire.
    - Si `stations` est une liste de dicts, on l’utilise directement.
    - Si `source` est fourni, on trace la source + ses trajets vers chaque station.
    """
    # 1) Charger la grille ETOPO (altitudes : négatives en mer, positives à terre)
    lats, lons, H, _ = load_etopo5(etopo_path)

    # 2) Réordonner en longitude selon le mode choisi
    if lon_mode == "180":
        lons_wrapped = _wrap180(lons)
        order = np.argsort(lons_wrapped)
        xlon = lons_wrapped[order]
        Hx = H[:, order]
        x_min, x_max = -180.0, 180.0
    else:
        lons_wrapped = _wrap360(lons)
        order = np.argsort(lons_wrapped)
        xlon = lons_wrapped[order]
        Hx = H[:, order]
        x_min, x_max = 0.0, 360.0

    # 3) Masques terre/océan et bornes de couleurs
    land_mask = (Hx >= 0.0)
    ocean_vals = np.where(~land_mask, Hx, np.nan)  # altitudes négatives
    if np.all(np.isnan(ocean_vals)):
        raise ValueError("Aucun pixel océan détecté ; vérifie la grille ETOPO.")
    vmin = np.nanpercentile(ocean_vals, 5)  # profondeur “courante”
    vmax = 0.0

    # 4) Figure
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Terre en à-plat gris
    land_img = np.where(land_mask, 1.0, np.nan)
    ax.imshow(
        land_img,
        extent=[x_min, x_max, lats[0], lats[-1]],
        origin="lower",
        interpolation="nearest",
        cmap=ListedColormap([land_color]),
        norm=Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
        zorder=1,
    )

    # Océan (plus “bleu” quand c’est profond)
    im = ax.imshow(
        ocean_vals,
        extent=[x_min, x_max, lats[0], lats[-1]],
        origin="lower",
        interpolation="nearest",
        cmap=ocean_cmap,
        vmin=vmin, vmax=vmax,
        aspect="auto",
        zorder=2,
    )

    # Axes & grille
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

    # Colorbar (graduée en mètres de profondeur)
    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03, format=FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
        cb.set_label("Ocean depth (m)")

    # 5) Stations (entrée flexible)
    station_list = []
    if isinstance(stations, str) and stations:
        try:
            station_list = _read_stations_csv(stations)
        except FileNotFoundError:
            station_list = []
    elif isinstance(stations, (list, tuple)):
        station_list = list(stations)

    if station_list:
        _plot_stations(ax, station_list, lon_mode=lon_mode, with_labels=True)

    # 6) Source + trajets vers les stations
    if source and ("lat" in source) and ("lon" in source):
        _plot_source(
            ax,
            lat=source["lat"],
            lon=source["lon"],
            lon_mode=lon_mode,
            label=source.get("label", "Estimated source"),
            radius_km=source.get("radius_km"),
            style=source.get("style"),
            zorder=8
        )
        if station_list:
            default_style = {"color": "crimson", "lw": 2.0, "alpha": 0.9}
            if path_style:
                default_style.update(path_style)
            for st in station_list:
                _plot_great_circle(
                    ax,
                    source["lat"], source["lon"],
                    st["lat"], st["lon"],
                    npts=1001,
                    lon_mode=lon_mode,
                    line_kwargs=default_style
                )

        if station_list:
            ax.legend(loc="lower left", frameon=True)

    # 7) Sauvegarde / affichage
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------
# 4) Exemple d’utilisation (exécutable directement)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Carte centrée “Greenwich” avec source + stations + trajets
    plot_world_map(
        etopo_path="data/etopo5.grd",
        stations="data/data_villes.csv",
        source={"lat": 24.9, "lon": 151.4, "label": "Estimated source", "radius_km": 200},
        lon_mode="180",
        title="ETOPO — Stations & great-circle paths",
        savepath="outputs/world_map.png",
        show=True
    )

    # Variante centrée Pacifique (0..360°)
    plot_world_map(
        etopo_path="data/etopo5.grd",
        stations="data/data_villes.csv",
        source={"lat": 41.03, "lon": 162.87, "label": "Source", "radius_km": 200},
        lon_mode="360",
        title="ETOPO — Pacific-centered",
        savepath="outputs/world_map_pacific.png",
        show=True
    )
