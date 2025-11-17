"""
Ce module calcule le temps de propagation d’un tsunami entre deux points
géographiques en intégrant 1/v(h) le long du grand cercle qui les relie,
où v(h) = sqrt(g * h) et h est la profondeur locale (en mètres).
"""

import numpy as np

from tsunami.geo import great_circle_points, arc_length_m
from tsunami.speed_model import speed

# ---------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------
def _safe_depth(depth_fn, lat, lon):
    """
    Applique depth_fn sur des tableaux lat/lon en essayant d'abord
    un appel vectorisé. Si la fonction ne supporte pas les tableaux,
    on retombe sur une version np.vectorize plus lente mais sûre.
    """
    try:
        depths = np.asarray(depth_fn(lat, lon), dtype=float)
        if depths.shape != lat.shape:
            raise ValueError
        return depths
    except Exception:
        vf = np.vectorize(lambda la, lo: depth_fn(float(la), float(lo)), otypes=[float])
        return vf(lat, lon)


def _longest_true_segment(mask):
    """
    Trouve la plus longue suite continue de valeurs True dans un tableau
    1D booléen.

    Renvoie (début, fin) inclusifs. Si aucun True → (-1, -1).
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return -1, -1

    gaps = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[gaps + 1]]
    stops  = np.r_[idx[gaps], idx[-1]]
    lengths = stops - starts + 1
    k = np.argmax(lengths)
    return int(starts[k]), int(stops[k])


# ---------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------

def travel_time_seconds(lat1, lon1, lat2, lon2, depth_fn, n_samples=2000, h_min=0.0):
    """
    Calcule le temps de propagation d’un tsunami (en secondes)
    entre deux points (lat, lon) donnés.

    Paramètres
    ----------
    lat1, lon1 : float
        Point de départ (degrés).
    lat2, lon2 : float
        Point d’arrivée (degrés).
    depth_fn : callable
        Fonction profondeur(lat, lon) -> profondeur d'eau (m, POSITIVE en mer).
    n_samples : int, optionnel
        Nombre de points d’échantillonnage le long du grand cercle
        (plus grand = plus précis, mais plus lent).
    h_min : float, optionnel
        Profondeur minimale (m) utilisée pour éviter v = 0
        dans les zones très peu profondes. h effective = max(h, h_min).

    Retour
    ------
    float
        Temps de trajet en secondes.

        - 0.0 si les deux points sont identiques
        - +inf si le trajet ne possède aucun segment océanique valable (tout sur la terre ou très fragmenté).
    """
    # Cas trivial : même point
    dist = arc_length_m(lat1, lon1, lat2, lon2)
    if dist == 0.0:
        return 0.0

    # 1) Points le long du grand cercle
    n_samples = max(3, int(n_samples))
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=n_samples)
    lats = pts[:, 0]
    lons = pts[:, 1]

    # 2) Profondeurs correspondantes
    depths = _safe_depth(depth_fn, lats, lons)

    # 3) Masque océan (profondeur d'eau > 0)
    ocean = np.isfinite(depths) & (depths > 0.0)

    # 4) On garde la plus longue portion océanique continue
    i0, i1 = _longest_true_segment(ocean)
    if i0 < 0 or i1 - i0 + 1 < 3:
        # pas de segment marin exploitable
        return float("inf")

    # 5) Profondeur efficace
    h = np.maximum(depths[i0:i1 + 1], float(h_min))

    # 6) Vitesse locale
    v = speed(h)
    if not np.all(np.isfinite(v)):
        return float("inf")

    # 7) Intégration numérique du temps total
    ds = dist / (len(lats) - 1)
    inv_v = 1.0 / v
    T = np.sum(inv_v[:-1]) * ds
    return float(T)