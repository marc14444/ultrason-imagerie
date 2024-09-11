import numpy as np
from scipy.interpolate import griddata

def interpolate_data(points, values, grid_x, grid_y):
    """
    Interpoler les données sur une grille.
    
    :param points: Points de données (x, y)
    :param values: Valeurs associées aux points
    :param grid_x: Grille x
    :param grid_y: Grille y
    :return: Valeurs interpolées sur la grille
    """
    return griddata(points, values, (grid_x, grid_y), method='cubic')





#from scipy.interpolate import griddata

#def interpolate_data(points, distances, grid_x, grid_y, method='cubic'):
    """
    Interpoler les distances mesurées sur une grille 2D.
    
    :param points: Coordonnées des points mesurés (x, y)
    :param distances: Distances mesurées
    :param grid_x: Grille 2D en x
    :param grid_y: Grille 2D en y
    :param method: Méthode d'interpolation (cubic, linear, etc.)
    :return: Grille interpolée (valeurs z)
    """
    #return griddata(points, distances, (grid_x, grid_y), method=method)

