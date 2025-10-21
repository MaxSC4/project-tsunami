import numpy as np
import matplotlib.pyplot as plt

from tsunami.geo import great_circle_points, arc_length_m
from tsunami.io_etopo import load_etopo5, make_depth_map

def topo_profile_along_path(lat1, lon1, lat2, lon2, depth, n_samples=1001):
    """
    Retourne le profil bathymétrique le long du grand cercle.
    - depth(lat, lon) doit renvoyer la profondeur (m), >0 en mer (NaN/<=0 sur terre).
    - n_samples: nombre de points le long du chemin.

    Retour:
        s_km      : distances cumulées (km) de 0 à distance_totale
        depths_m  : profondeurs (m) à chaque point (mêmes longueurs que s_km)
    """
    # distance totale (m) + points
    dist = arc_length_m(lat1, lon1, lat2, lon2)
    if dist == 0:
        return np.array([0.0]), np.array([np.nan])

    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=n_samples)
    lats, lons = pts[:, 0], pts[:, 1]

    # profondeur le long du chemin (boucle simple, lisible)
    depths = np.empty_like(lats)
    for i in range(len(lats)):
        depths[i] = depth(lats[i], lons[i])

    # distances cumulées: pas constant car points uniformes en angle → ds = dist/(n_samples-1)
    ds = dist / (n_samples - 1)
    s_m = np.arange(n_samples, dtype=float) * ds
    s_km = s_m / 1000.0

    return s_km, depths

def plot_topo_profile(s_km, depths_m, title="Profil bathymétrique le long du trajet"):
    """
    Trace la profondeur (positive en m) vs distance (km).
    Si tu veux "plonger" vers le bas, trace -depths_m.
    """
    plt.figure()
    plt.plot(s_km, -depths_m)   # signe - pour afficher la mer vers le bas
    plt.xlabel("Distance le long du trajet (km)")
    plt.ylabel("Profondeur (m, vers le bas)")
    plt.title(title)
    plt.grid(True)
    plt.show()

lats, lons, H, nodata = load_etopo5("data/etopo5.grd")
depth = make_depth_map(lats, lons, H)

lat1, lon1 = 0.0, 150.0
lat2, lon2 = 35.0, 140.0

s_km, depths_m = topo_profile_along_path(lat1, lon1, lat2, lon2, depth, n_samples=1201)

print(f"Distance totale: {s_km[-1]:.1f} km")
print(f"Profondeur min/max (m): {np.nanmin(depths_m):.1f} / {np.nanmax(depths_m):.1f}")

plot_topo_profile(s_km, depths_m, title="Pacifique → Japon (profil bathy)")


