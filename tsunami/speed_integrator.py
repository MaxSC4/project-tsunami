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
    try:
        # Essaye d'appeler la fonction directement
        depths = np.asarray(depth_fn(lat, lon), dtype=float)
        if depths.shape != lat.shape:
            raise ValueError
        return depths
    except Exception:
        # Si erreur : on applique un mode lent mais sûr
        vf = np.vectorize(lambda la, lo: depth_fn(float(la), float(lo)), otypes=[float])
        return vf(lat, lon)

def _longest_true_segment(mask):
    """
    Trouve la plus longue suite continue de valeurs True dans un tableau.
    Renvoie (début, fin) inclusifs.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return -1, -1

    gaps = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[gaps + 1]]
    stops = np.r_[idx[gaps], idx[-1]]
    lengths = stops - starts + 1
    k = np.argmax(lengths)
    return int(starts[k]), int(stops[k])


# --------------------------------------------------------------
# Fonction principale
# --------------------------------------------------------------

def travel_time_seconds(lat1, lon1, lat2, lon2,
                        depth_fn, n_samples=2000,
                        h_min=50.0, shore_trim=10):
    """
    Calcule le temps de propagation d’un tsunami (en secondes) entre deux points (lat, lon) donnés.

    Paramètres :
        lat1, lon1 : point de départ (degrés)
        lat2, lon2 : point d’arrivée (degrés)
        depth_fn   : fonction profondeur(lat, lon) -> profondeur (m, positive en mer)
        n_samples  : nombre de points pour le calcul (plus = plus précis)
        h_min      : profondeur minimale (m) pour éviter v=0
        shore_trim : nombre de points à retirer à chaque extrémité (effet côte)

    Retour :
        Temps de trajet en secondes (float)
        +inf si le trajet passe entièrement sur la terre
        0.0 si les deux points sont identiques
    """
    # Cas trivial
    dist = arc_length_m(lat1, lon1, lat2, lon2)
    if dist == 0.0:
        return 0.0

    # 1. Points le long du grand cercle
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=max(3, int(n_samples)))
    lats = pts[:, 0]
    lons = pts[:, 1]

    # 2. Profondeurs à ces points
    depths = _safe_depth(depth_fn, lats, lons)

    # 3. Masque des zones océaniques (profondeur > 0)
    ocean = np.isfinite(depths) & (depths > 0)

    # 4. On garde la plus longue portion océanique continue
    i0, i1 = _longest_true_segment(ocean)
    if i0 < 0 or i1 - i0 + 1 < 3:
        return float('inf')

    # 5. On retire les bords (effet des côtes)
    i0 += max(0, int(shore_trim))
    i1 -= max(0, int(shore_trim))
    if i1 - i0 + 1 < 3:
        return float('inf')

    # 6. Profondeur efficace (évite h=0)
    h = np.maximum(depths[i0:i1 + 1], float(h_min))

    # 7. Vitesse locale (m/s)
    v = speed(h)
    if not np.all(np.isfinite(v)):
        return float('inf')

    # 8. Intégration du temps total
    ds = dist / (len(lats) - 1)   # distance entre deux points
    inv_v = 1.0 / v
    T = np.sum(inv_v[:-1]) * ds   # somme des petites durées
    return float(T)