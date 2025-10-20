import numpy as np
from speed_model import speed

R_EARTH = 6371000.0 # en mètres

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
        lat_phi = np.arcsin(p[2])
        lon_lambda = np.arctan2(p[1], p[0])
        pts.append([np.rad2deg(lat_phi), np.rad2deg(lon_lambda)])
    return pts


def arc_length_m(lat1, lon1, lat2, lon2):
    phi1, lambda1, phi2, lambda2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    w = np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1))
    return R_EARTH * w


"""on integre T= int(ds/vteta)
ici ds= R dteta
T= R* int( dteta/v(teta))
T=R*somme (delta teta/v(teta))
v teta n'a pas de sens, 
on veut v(lat,lon) v on associe v aux valeurs de point 
v( point 
  )

J'associe a chaque point du tableau de trajectoire la valeur v
"""


def TEMPS(H):
    pts = great_circle_points(32, 50, 23, 20, npts=400)
    deltateta=pts["angle"]/(pts["npts"]-1)
    
    V=speed(H)
    #lats, lons = pts[:, 0], pts[:, 1]
    
    T=R_EARTH*deltateta*np.sum(1/(V[lats,lons]))
    return T
    
    
    
    
#s=arc_length_m(lat1, lon1, lat2, lon2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    