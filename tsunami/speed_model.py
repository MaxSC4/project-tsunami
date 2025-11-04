import numpy as np
g = 9.81

"""
def speed(h):
    return np.sqrt(np.abs(-g * h))
"""

def speed(h):
    """
    Renvoie la vitesse de phase du tsunami (m/s) pour une profondeur h (m).
    h doit être POSITIVE (profondeur d'eau, pas altitude).
    On impose une vitesse minimale pour éviter les divisions par zéro.
    """
    h = np.asarray(h, dtype=float)
    # on met un plancher de 1 m pour éviter v = 0
    h_safe = np.maximum(h, 1.0)
    v = np.sqrt(g * h_safe)
    v[h <= 0] = np.nan
    return v