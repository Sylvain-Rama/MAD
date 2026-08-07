import numpy as np
from mad.objs import MovableObj, Planet

def random_point_at_surface(planet:Planet, altitude: float = 10, name: str = "SurfaceObj", dims: int = 2) -> MovableObj:
    # Create a random object at the 2D or 3D surface (+ altitude) of the planet.
    # the dims parameter allows to create a 2D or 3D point, with the other dimensions set to 0.

    if not 0 < dims < 4:
        raise ValueError("Dimensions for the point definition must be between 1 and 3")
    v = np.random.normal(size=dims)
    v /= np.linalg.norm(v)

    return MovableObj(position=(planet.radius + altitude) * v + planet.position[:dims], name=name)

def point_at_distance(
    planet: Planet, obj: MovableObj, distance_km: float, altitude: float = 10, name="RangedObj", dims: int = 2
) -> MovableObj:
    # Create a new random object at set distance from another point on the planet.
    # 2D or 3D mode.

    u = obj.normalize[:dims]
    sigma = (distance_km * 1000) / planet.radius

    # random orthogonal direction
    v = np.random.normal(size=dims)
    v -= np.dot(v, u) * u
    v /= np.linalg.norm(v)

    point = np.cos(sigma) * u + np.sin(sigma) * v

    return MovableObj(position=(planet.radius + altitude) * point + planet.position[:dims], name=name)
