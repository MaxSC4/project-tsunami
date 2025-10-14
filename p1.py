# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
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


xllcenter=0.000000

yllcenter=-89.916667

xmin =xllcenter -cellsize / 2
xmax = xllcenter + ncols * cellsize - cellsize / 2
ymin = yllcenter - cellsize / 2
ymax = yllcenter + nrows * cellsize - cellsize / 2

extendex = [xmin, xmax, ymin, ymax]



A=pd.read_csv('etopo5.grd',delimiter=' ',skiprows=6)


masque=A<=0



Bathymetrie=A[masque]



Va=np.sqrt(-Bathymetrie*g)

plt.plot
plt.figure()
plt.imshow(Va,'turbo',extent=extendex)




