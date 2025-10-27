import numpy as np
from scipy.ndimage import map_coordinates

def load_etopo5(path):
    """Lit une grille (ETOP05) et renvoie (lats, lons, H, nodata)."""

    # On lit les 6 premières lignes pour récupérer les mots-clés (ncols, nrows, cellsize, etc.)
    # On les stocke dans un dictionnaire
    # Puis on lit le reste du fichier
    meta = {}
    with open(path, "r") as f:
        for _ in range(6):
            k, v = f.readline().strip().split()
            meta[k.lower()] = float(v)
        data = np.loadtxt(f)

    # ncols, nrows : dimensions de la grille
    # cell : taille d’une cellule (en degrés)
    # nodata : valeur spéciale utilisée pour signaler une cellule “sans donnée”
    ncols = int(meta["ncols"])
    nrows = int(meta["nrows"])
    cell = meta["cellsize"]
    nodata = meta.get("nodata_value", -99999.0)

    # centres de cellules (xllcenter/yllcenter)
    # lon0 : longitude du premier centre de cellule
    # lat0 : latitude du premier centre de cellule
    # lons et lats : vecteurs des longitudes et latitudes de chaque colonne/ligne
    # ETOPO5 est ordonné “ligne par ligne” du sud vers le nord, donc lats[0] = -90°, lats[-1] = +90°
    lon0 = meta.get("xllcenter", 0.0)
    lat0 = meta.get("yllcenter", -90.0 + cell/2)
    lons = lon0 + np.arange(ncols) * cell
    lats = lat0 + np.arange(nrows) * cell

    # reforme la matrice 2D H avec les dimensions correctes (nrows × ncols)
    # chaque case = altitude (négatif = océan, positive = terre)
    H = data.reshape(nrows, ncols)
    return lats, lons, H, nodata


def make_depth_map(lats, lons, H, order=1):
    nrows, ncols = H.shape
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    lat0 = float(lats[0])
    lon0 = float(lons[0])
    lon_span = dlon * ncols

    def _lon_wrap_arr(lon_arr):
        return lon0 + ((lon_arr - lon0) % lon_span)

    def depth(lats_in, lons_in):
        lats_in = np.asarray(lats_in, dtype=float)
        lons_in = np.asarray(lons_in, dtype=float)

        lw = _lon_wrap_arr(lons_in)
        i = (lats_in - lat0) / dlat
        j = (lw - lon0) / dlon

        coords = np.vstack([i, j])

        vals = map_coordinates(H, coords, order=order, mode="nearest")

        return vals.astype(float)

    return depth


# TEST
"""
lats, lons, H, _ = load_etopo5("data/etopo5.grd")
depth = make_depth_map(lats, lons, H)

lat0, lon0 = 10.0, 142.5
val = depth(np.array([lat0]), np.array([lon0]))[0]
print("Point unique:", val)
"""