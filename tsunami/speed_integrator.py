import numpy as np
from geo import great_circle_points, arc_length_m
from speed_model import speed

def travel_time_seconds(lat1, lon1, lat2, lon2, depth, n_samples=3500):
    # distance totale en m
    dist = arc_length_m(lat1, lon1, lat2, lon2)
    if dist == 0: # si la distance est nulle, le temps aussi
        return 0.0

    # génère npts points régulièrement espacés le long du grand cercle
    # les points représentent le chemin parcouru par l’onde entre les deux lieux.
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts = n_samples)
    lats, lons = pts[:, 0], pts[:, 1]

    # regarde la grille de bathy (ETOPO5) via la fonction depth(lat, lon)
    # pour obtenir la profondeur de l’océan en mètres le long du trajet.
    depths = np.empty_like(lats)
    for i in range(len(lats)):
        depths[i] = depth(lats[i], lons[i])

    # Si un seul point tombe sur la terre (profondeur <= 0 ou NaN),
    # alors le trajet n’est pas valide
    mask = np.isfinite(depths) & (depths > 0)
    if not np.all(mask):
        return np.inf # temps infini = temps impossible (on peut changer ça, on en avait parlé)

    v = speed(depths) # m/s
    inv_v = 1.0 / v # inverse de la vitesse (s/m)
    ds = dist / (n_samples - 1) # longueur d’un segment (m)
    T = float(np.sum(inv_v[:-1]) * ds) # intégration

    return T



