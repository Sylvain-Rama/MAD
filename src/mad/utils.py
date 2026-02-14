import math


def shortest_angle(a, b):
    """Renvoie la différence angulaire [-pi, pi]"""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return d
