import numpy as np

def travel_time(lat1, lon1, lat2, lon2):
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    R = 6371 # Rayon de la Terre (en m)

    a = np.sin(lon1) * np.sin(lon2) + np.cos(lon1) * np.cos(lon2) * np.cos(lat2 - lat1)
    b = R * np.arccos(a)

    return b

print(travel_time(40, 76, 48, 0))