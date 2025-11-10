"""
Lecture et utilisation d'une grille ETOPO (ASCII).

Ce module permet :
    1. de lire un fichier de relief global (ETOPO5),
    2. de construire une fonction "depth(lat, lon)" qui renvoie la profondeur d’eau (m).

Usage :
    lats, lons, H, nodata = load_etopo5("data/etopo5.grd")
    depth = make_depth_function(lats, lons, H)
    print(depth([20, 25], [140, 142]))  # profondeurs en mètres

Convention :
    - H contient l'altitude en mètres (négatif = océan, positif = terre)
    - depth() renvoie la profondeur POSITIVE en mer, 0 à terre, NaN si donnée manquante
"""

import numpy as np
from scipy.ndimage import map_coordinates

# --------------------------------------------------------------
# 1. Lecture du fichier ETOPO
# --------------------------------------------------------------
def load_etopo5(path):
    """
    Lit une grille ETOPO au format ASCII et renvoie :
        lats  : latitudes (du sud vers le nord)
        lons  : longitudes (en degrés)
        H     : matrice 2D (altitudes en m)
        nodata: valeur utilisée pour signaler les données manquantes
    """
    meta = {}
    with open(path, "r") as f:
        # Les 6 premières lignes contiennent les métadonnées
        for _ in range(6):
            k, v = f.readline().strip().split()
            meta[k.lower()] = float(v)

        # Le reste du fichier contient les valeurs
        data = np.loadtxt(f)

    # Récupération des métadonnées utiles
    ncols = int(meta["ncols"])
    nrows = int(meta["nrows"])
    cell = meta["cellsize"]
    nodata = meta.get("nodata_value", -99999.0)
    lon0 = meta.get("xllcenter", 0.0)
    lat0 = meta.get("yllcenter", -90.0)

    # Coordonnées des centres de cellules
    lons = lon0 + np.arange(ncols) * cell
    lats = lat0 + np.arange(nrows) * cell

    # On remet la matrice dans la bonne orientation : sud → nord
    H = data.reshape(nrows, ncols)
    H = np.flipud(H)

    return lats, lons, H, nodata


# --------------------------------------------------------------
# 2. Création d'une fonction depth(lat, lon)
# --------------------------------------------------------------
def make_depth_function(lats, lons, H, order=1, nodata=None):
    """
    Crée une fonction depth(lat, lon) qui renvoie la profondeur (m).

    Paramètres :
        lats, lons : vecteurs de latitudes et longitudes de la grille
        H          : matrice d’altitudes (négatif = océan)
        order      : 0 = nearest, 1 = bilinéaire, 3 = bicubique
        nodata     : valeur spéciale pour les données manquantes (optionnel)

    Retourne :
        depth(lat, lon) -> profondeur positive (m) ou 0 sur la terre
    """
    nrows, ncols = H.shape
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    lat0 = lats[0]
    lon0 = lons[0]
    lon_span = dlon * ncols

    def _wrap_lon(lon):
        """Assure le rebouclage en longitude (0–360° ou -180–180°)."""
        return lon0 + ((lon - lon0) % lon_span)

    def depth(lat_in, lon_in):
        """Renvoie la profondeur d’eau (mètres)."""
        lat_in = np.asarray(lat_in, dtype=float)
        lon_in = np.asarray(lon_in, dtype=float)
        lon_w = _wrap_lon(lon_in)

        # Conversion latitude/longitude vers indices de grille
        i = (lat_in - lat0) / dlat
        j = (lon_w - lon0) / dlon
        coords = np.vstack([i, j])

        # Interpolation
        vals = map_coordinates(H, coords, order=order, mode="nearest")

        # Nettoyage : valeurs nodata et altitudes positives → 0 ou NaN
        vals = np.where(np.isfinite(vals), vals, np.nan)
        if nodata is not None:
            vals[vals == nodata] = np.nan

        # Altitudes négatives → profondeur positive
        depth_m = np.where(vals < 0, -vals, 0.0)
        return depth_m.astype(float)

    return depth

# --------------------------------------------------------------
# 3. Test
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Test : lecture ETOPO et calcul d'une profondeur")
    lats, lons, H, nodata = load_etopo5("data/etopo5.grd")
    depth = make_depth_function(lats, lons, H, order=1)
    val = depth([80], [76])[0]
    print(f"Profondeur au point (80N, 76E) : {val:.1f} m")