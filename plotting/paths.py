import numpy as np
import matplotlib.pyplot as plt
from tsunami.geo import great_circle_points

def _wrap180(lon):
    out = (lon + 180.0) % 360.0 - 180.0
    # 180 → -180 pour rester en [-180,180)
    if np.isscalar(out):
        return -180.0 if out == 180.0 else out
    out[out == 180.0] = -180.0
    return out

def _split_on_dateline(lons, lats):
    """
    Coupe la polyline quand le saut de longitude dépasse 180°, pour éviter
    une ligne qui traverse toute la carte.
    Retourne une liste de segments (lons_i, lats_i).
    """
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    segs = []
    start = 0
    for k in range(1, len(lons)):
        if abs(lons[k] - lons[k-1]) > 180.0:
            segs.append((lons[start:k], lats[start:k]))
            start = k
    segs.append((lons[start:], lats[start:]))
    return segs

def plot_great_circle(ax, lat1, lon1, lat2, lon2, npts=1001, **line_kwargs):
    """
    Trace le grand cercle entre (lat1,lon1) et (lat2,lon2) sur l'axe `ax`.
    Gère le wrap longitudes et la coupure ±180°.
    """
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=npts)
    lats = pts[:, 0]
    lons = _wrap180(pts[:, 1])

    for (lx, ly) in _split_on_dateline(lons, lats):
        ax.plot(lx, ly, **({"lw": 2, "color": "k"} | line_kwargs))
