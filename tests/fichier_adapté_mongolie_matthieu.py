# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:05:38 2025

@author: matth
"""

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


g = 9.81
R_EARTH = 6371000.0 # en mètres

#%%


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

"""a comprendre"""
def make_depth_map(lats, lons, H):
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

        vals = map_coordinates(H, coords, order=1, mode="nearest")

        return vals.astype(float)

    return depth



def deg2rad(x): return np.deg2rad(x)
def rad2deg(x): return np.rad2deg(x)

def great_circle_points(lat1, lon1, lat2, lon2, npts=400):
    """npts points (lat, lon) uniformes en angle entre A et B (en degrés)."""
    phi1, lambda1, phi2, lambda2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))

    # vecteurs 3D
    def sph2cart(phi, lambda_):
        return np.array([np.cos(phi)*np.cos(lambda_), np.cos(phi)*np.sin(lambda_), np.sin(phi)])

    A, B = sph2cart(phi1, lambda1), sph2cart(phi2, lambda2)

    # angle
    w = np.arccos(np.dot(A, B))
    
    if w == 0:
        return np.array([[lat1, lon1]])

    ts = np.linspace(0, 1, npts)
    pts = []
    for t in ts:
        num = np.sin((1-t)*w)*A + np.sin(t*w)*B
        p = num / np.sin(w)
        
        lat_phi = np.arcsin(p[2])                       ######"remise en spherique"
        lon_lambda = np.arctan2(p[1], p[0])
        pts.append([np.rad2deg(lat_phi), np.rad2deg(lon_lambda)])
    return np.array(pts)

def arc_length_m(lat1, lon1, lat2, lon2):
    phi1, lambda1, phi2, lambda2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    w = np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1))
    return R_EARTH * w


def speed(h):
    
    return np.sqrt(np.abs(-g * h))

def travel_time_seconds(lat1, lon1, lat2, lon2, depth, n_samples=1000):
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
    mask = np.isfinite(depths) & (depths < 0)
    if not np.all(mask):
        return np.inf # temps infini = temps impossible (on peut changer ça, on en avait parlé)

    v = speed(depths) # m/s
    inv_v = 1.0 / v # inverse de la vitesse (s/m)
    ds = dist / (n_samples - 1) # longueur d’un segment (m)
    T = float(np.sum(inv_v[:-1]) * ds) # intégration

    return T






#%%
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









#%%
lats, lons, H, nodata = load_etopo5("../data/etopo5.grd")
depth = make_depth_map(lats, lons, H)

lat1, lon1 = 0.0, 150.0
lat2, lon2 = 35.0, 140.0

s_km, depths_m = topo_profile_along_path(lat1, lon1, lat2, lon2, depth, n_samples=1201)

print(f"Distance totale: {s_km[-1]:.1f} km")
print(f"Profondeur min/max (m): {np.nanmin(depths_m):.1f} / {np.nanmax(depths_m):.1f}")

plot_topo_profile(s_km, depths_m, title="Pacifique → Japon (profil bathy)")


#%%
def zone_depart(lat_min, lat_max, lon_min, lon_max, step=2.0):
    coords=[]

    for lat in np.arange(lat_min, lat_max + step, step):
        for lon in np.arange(lon_min, lon_max + step, step):
            coords.append([lat, lon])
    coords = np.array(coords)       
    return coords
    
    """
   Génère une liste de coordonnées (lat, lon)
   dans une zone rectangulaire définie par ses bornes et un pas donné.

   Paramètres :
       lat_min, lat_max : bornes de latitude (°)
       lon_min, lon_max : bornes de longitude (°)
       step : pas entre deux points (°)

   Retour :
       coords : liste de tuples (lat, lon)
   """
   


#%%
#%% Calcul des temps de parcours depuis une zone de départ

#  Définition du point d'arrivée (ex : Japon)
lat_arrivee, lon_arrivee = 40.0, 140.0

# Définition de la zone de départ 
zone = zone_depart(lat_min=-50, lat_max=50, lon_min=120, lon_max=260, step=10.0)

# Liste pour stocker les résultats
resultats = []

#Boucle sur chaque point de la zone
for lat_dep, lon_dep in zone:
    T = travel_time_seconds(lat_dep, lon_dep, lat_arrivee, lon_arrivee, depth)

    # On convertit le temps en heures pour affichage
    T_h = T / 3600.0 if np.isfinite(T) else np.nan
    if not np.isnan(T_h):
        resultats.append([lat_dep, lon_dep, T_h])

#onversion en tableau numpy pour traitement ultérieur
resultats = np.array(resultats)


#%%#%%####
def fake_depth(lat, lon):
    return 4000.0

lat1, lon1 = 0.0, 150.0
lat2, lon2 = 35.0, 140.0

d = arc_length_m(lat1, lon1, lat2, lon2)
v = np.sqrt(9.81 * 4000)
T_theo = d / v

T_num = travel_time_seconds(lat1, lon1, lat2, lon2, fake_depth)

print(f"Distance : {d/1000:.1f} km")
print(f"Vitesse  : {v:.1f} m/s")
print(f"Théorique: {T_theo/3600:.2f} h")
print(f"Numérique: {T_num/3600:.2f} h")
print(f"Écart : {100*(T_num - T_theo)/T_theo:.2f} %")