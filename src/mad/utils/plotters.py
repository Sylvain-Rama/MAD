import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches
import numpy as np
from numpy.typing import NDArray
from mad.objs.base import MovableObj, BallisticObj
from mad.objs.planets import Planet


def plot_2D_planet_with_points(
    self, planet: Planet, points: list[MovableObj] | None = None, ax=None, display="planet"
) -> matplotlib.figure.Figure | None:
    # 2D plot of the planet. If using point in 2D, they will appear at the circumference.
    plot_fig = False
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))
        plot_fig = True

    if display == "arc" and points is not None and len(points) == 2:

        theta1 = np.degrees(
            np.arctan2(points[0].position[1] - planet.position[1], points[0].position[0] - planet.position[0])
        )
        theta2 = np.degrees(
            np.arctan2(points[1].position[1] - planet.position[1], points[1].position[0] - planet.position[0])
        )
        planet_body = matplotlib.patches.Arc(
            (float(planet.position[0]), float(planet.position[1])),
            2 * planet.radius,
            2 * planet.radius,
            angle=0,
            theta1=min(theta1, theta2),
            theta2=max(theta1, theta2),
            ec="black",
            label=planet.name,
            ls="--",
        )
        ax.add_patch(planet_body)
    elif display == "planet":
        planet_body = matplotlib.patches.Circle(
            (float(planet.position[0]), float(planet.position[1])),
            radius=planet.radius,
            ec="black",
            fill=False,
            label=planet.name,
            ls="--",
        )
        ax.add_patch(planet_body)

    if points is not None:
        for point in points:
            ax.scatter(x=point.position[0], y=point.position[1], s=50, label=point.name)

    ax.set_aspect("equal")
    ax.legend()
    ax.grid()

    return fig if plot_fig else None  # type: ignore
