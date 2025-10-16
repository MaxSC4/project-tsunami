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




#estimation de l'extent la liste [x min, x max, ymin, y max]
#blabla


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

#%% Distance entre deux pointd 1 et 2 en lat/lon
def Dist (lat1,lon1, lat2, lon2):
    lat1=np.radians(lat1)
    lon1=np.radians(lon1)
    lat2=np.radians(lat2)
    lon2=np.radians(lon2)
    
    
    
    R=6370000
    d_lon=lon2-lon1
    D=R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(d_lon)+np.sin(lat1)*np.sin(lat2))
    return D
    
Dist(48,2,40,76)

#%%























