import numpy as np

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

"""
ON REGARDE PAS CA POUR L'INSTANT
def depth(lat, lon, lats, lons, H, nodata=-99999.0):
    # Interpolation pour obtenir la profondeur à une position précise (lat, lon) entre les points de la grille. Renvoie np.nan si on touche du NODATA.
    # ramène la longitude dans l’intervalle [0, 360) (ex: -150° = 210°)
    lon = (lon + 360.0) % 360.0

    # indices bas-gauche
    i = np.searchsorted(lats, lat) - 1
    j = np.searchsorted(lons, lon) - 1
    i = np.clip(i, 0, len(lats)-2)
    j = np.clip(j, 0, len(lons)-2)

    y0, y1 = lats[i], lats[i+1]
    x0, x1 = lons[j], lons[j+1]
    dy = (lat - y0) / (y1 - y0 + 1e-12)
    dx = (lon - x0) / (x1 - x0 + 1e-12)

    patch = H[i:i+2, j:j+2]
    if np.any(patch == nodata):
        return np.nan

    return ((1-dx)*(1-dy)*patch[0,0] + dx*(1-dy)*patch[0,1] + (1-dx)*dy*patch[1,0] + dx*dy*patch[1,1])
"""


