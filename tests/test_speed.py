from tsunami.geo import arc_length_m
from tsunami.speed_integrator import travel_time_seconds
import numpy as np

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
