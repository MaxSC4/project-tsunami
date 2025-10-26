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
    dist = arc_length_m(lat1, lon1, lat2, lon2)
    if dist == 0:
        return np.array([0.0]), np.array([np.nan])

    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=n_samples)
    lats, lons = pts[:, 0], pts[:, 1]

    depths = np.empty_like(lats)
    for i in range(len(lats)):
        depths[i] = depth(lats[i], lons[i])

    ds = dist / (n_samples - 1)
    s_km = np.arange(n_samples, dtype=float) * ds / 1000.0
    return s_km, depths


def plot_topo_profiles(s_km, depths_interp, depths_raw, title="Profil bathymétrique le long du trajet"):
    plt.figure(figsize=(10, 5))
    plt.plot(s_km, -depths_interp, label="Interpolé (linéaire)", linewidth=2)
    plt.step(s_km, -depths_raw, where='mid', label="Non interpolé (nearest)", alpha=0.7)
    plt.xlabel("Distance le long du trajet (km)")
    plt.ylabel("Profondeur (m, vers le bas)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Exemple d’utilisation ===
lats, lons, H, nodata = load_etopo5("data/etopo5.grd")

# depth interpolé (bilinéaire)
depth_interp = make_depth_map(lats, lons, H, order=1)

# depth brut (nearest neighbor)
depth_raw = make_depth_map(lats, lons, H, order=0)

lat1, lon1 = 0.0, 150.0
lat2, lon2 = 35.0, 140.0

# calcul des deux profils
s_km, depths_interp = topo_profile_along_path(lat1, lon1, lat2, lon2, depth_interp, n_samples=1001)
s_km, depths_raw = topo_profile_along_path(lat1, lon1, lat2, lon2, depth_raw, n_samples=1001)

print(f"Distance totale: {s_km[-1]:.1f} km")
print(f"Profondeur min/max (interp): {np.nanmin(depths_interp):.1f} / {np.nanmax(depths_interp):.1f}")
print(f"Profondeur min/max (raw): {np.nanmin(depths_raw):.1f} / {np.nanmax(depths_raw):.1f}")

plot_topo_profiles(s_km, depths_interp, depths_raw, title="Pacifique → Japon : profil bathy comparé")
