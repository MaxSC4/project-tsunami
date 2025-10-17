import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''ncols 4320
nrows 2160
xllcenter 0.000000
yllcenter -89.916667
cellsize 0.0833333
nodata_value -99999'''



g=9.81

cellsize=0.0833333333
ncols=4320
nrows=2160





xllcenter=0.000000

yllcenter=-89.916667

xmin =xllcenter -cellsize / 2
xmax = xllcenter + ncols * cellsize - cellsize / 2
ymin = yllcenter - cellsize / 2
ymax = yllcenter + nrows * cellsize - cellsize / 2

extendex = [xmin, xmax, ymin, ymax]



A=pd.read_csv('etopo5.grd',delimiter=' ',skiprows=6)  #lecture du fichier de base

positionville=pd.read_csv('data_villes.csv',delimiter=' ')
positionville["Longitude_E"] = positionville["Longitude_E"] % 360


#faire une boucle pour extraire les positions des villes



masque=A<=0
Bathymetrie=A[masque]



Va=np.sqrt(-Bathymetrie*g)

plt.figure()

plt.imshow(Va,extent=extendex)
plt.plot(positionville["Longitude_E"], positionville["Latitude_N"], 'ro', markersize=8)
for i, nom in enumerate(positionville["Ville/Port"]):
    plt.text(positionville["Longitude_E"][i] + 1.5,
             positionville["Latitude_N"][i] + 1.5,
             nom, color='white', fontsize=8)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Carte avec les positions des villes")



#%%
#CONVERSION DE DEGRe au RADIAN 
def conversion_deg2rad(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    return lat1_rad, lon1_rad, lat2_rad, lon2_rad
#DONNE LA DISTANCE ENTRE DEUX POINTS DE LATLON EN RADIAN

def Dist (lat1_rad, lon1_rad, lat2_rad, lon2_rad):
      
    R=6370000
    d_lon=lon2_rad-lon1_rad
    D=R*np.arccos(np.cos(lat1_rad)*np.cos(lat2_rad)*np.cos(d_lon)+np.sin(lat1_rad)*np.sin(lat2_rad))
    return D
    


#CONVERSION LATLON A XYZ
def sph2cart(lat_deg, lon_deg, R=1.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])


#CONVERSION XYZ a latlon
def cart2sph(x, y, z):
    lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon




"""Pour avoir l'equation du grand cercle, je vais resoudre l'equation du plan AB et de la terre
Le plan est defini par Avect et Bvect
sphere de centre O
x^^2+y**2+z**2=R**2



"""
#%% 
"""
A et B les coordonnées xyz en latlon

A = np.array([1, 2, 3])
B = np.array([2, -1, 1])
"""
n = np.cross(A, B)
print("Vecteur normal :", n)
print(f"Équation du plan : {n[0]:.2f}·x + {n[1]:.2f}·y + {n[2]:.2f}·z = 0")


r = 5   # rayon de la sphère
print(f"Équation de la sphère : x² + y² + z² = {r**2}")


u_hat = A / np.linalg.norm(A)
v_perp = B - np.dot(B, u_hat) * u_hat
v_hat = v_perp / np.linalg.norm(v_perp)



# --- Paramétrisation du cercle ---
t = np.linspace(0, 2*np.pi, 400)
circle = np.array([r * (np.cos(tt)*u_hat + np.sin(tt)*v_hat) for tt in t])






