import numpy as np
g = 9.81

def speed(h):
    
    return np.sqrt(np.abs(-g * h))
'''ajouter nan pour h positif'''
