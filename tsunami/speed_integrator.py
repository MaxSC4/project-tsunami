# tsunami/speed_integrator.py
import numpy as np
from tsunami.geo import great_circle_points, arc_length_m
from tsunami.speed_model import speed

def travel_time_seconds(lat1, lon1, lat2, lon2, depth_fn,
                        n_samples=2000, h_min=50.0, shore_trim=10):
    """
    Intègre 1/v le long du grand cercle, avec tolérance aux côtes.
    h_min : profondeur minimale (m) pour éviter vitesses infinies.
    shore_trim : nb de points coupés à chaque extrémité du trajet.
    """
    dist = arc_length_m(lat1, lon1, lat2, lon2)
    if dist == 0:
        return 0.0

    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=n_samples)
    lats, lons = pts[:, 0], pts[:, 1]
    depths = depth_fn(lats, lons)

    # Coupe les extrémités (station côtière)
    if 2 * shore_trim < len(depths):
        depths = depths[shore_trim:-shore_trim]
        dist *= (len(depths) - 1) / (n_samples - 1)

    # Garde uniquement le segment marin le plus long
    ocean = (np.isfinite(depths) & (depths < 0))
    if not np.any(ocean):
        return np.nan

    idx = np.where(ocean)[0]
    gaps = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, gaps + 1]
    stops  = np.r_[gaps, len(idx) - 1]
    blocks = [(idx[s], idx[e]) for s, e in zip(starts, stops)]
    i0, i1 = max(blocks, key=lambda ab: ab[1] - ab[0])
    if i1 - i0 + 1 < 50:
        return np.nan

    dseg = np.abs(depths[i0:i1+1])
    h_eff = np.maximum(dseg, h_min)
    v = speed(h_eff)
    inv_v = 1.0 / v
    ds = dist / (len(h_eff) - 1)
    return np.sum(inv_v[:-1]) * ds
