"""Helper plotters for MAD simulation."""

import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches
import matplotlib.axes as mplaxes
import numpy as np
import pandas as pd
from mad.objs import MovableObj, Planet


def plot_2D_planet_with_points(
    planet: Planet, points: list[MovableObj] | None = None, ax: mplaxes.Axes | None = None, display="planet"
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


def plot_ballistic_range_table(df: pd.DataFrame, selected_altitudes_km: list[float], table_name: str):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), sharex=True, sharey=True)
    ax = ax.ravel()

    # Compute global vmin/vmax across all selected altitudes for a shared colorscale
    dfs = [
        df[df["altitude_km"] == alt].pivot(index="velocity_m_s", columns="gamma_deg", values="range_km")
        for alt in selected_altitudes_km
    ]
    vmin = min(d.values.min() for d in dfs if not d.empty)
    vmax = max(d.values.max() for d in dfs if not d.empty)

    imgs = []
    for i, (selected_altitude, df2) in enumerate(zip(selected_altitudes_km, dfs)):
        if df2.empty:
            continue
        img = ax[i].imshow(
            df2.values,
            extent=(df2.columns.min(), df2.columns.max(), df2.index.min(), df2.index.max()),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="inferno",
        )
        imgs.append(img)
        ax[i].set_xlabel("gamma (deg)")
        ax[i].set_ylabel("velocity (m/s)")
        ax[i].set_title(f"{selected_altitude} km Altitude")

    fig.suptitle(f"Ballistic Range Table for {table_name}")
    fig.tight_layout(pad=1.2)
    _ = fig.colorbar(imgs[0], ax=ax, fraction=0.2, pad=0.04, label="Range (km)")


def plot_ballistic_range_table_gamma(
    df: pd.DataFrame, selected_gamma: float, table_name: str, ax: mplaxes.Axes | None = None
) -> matplotlib.figure.Figure | None:
    fig: matplotlib.figure.Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))

    # Compute global vmin/vmax across all selected altitudes for a shared colorscale
    dfs = df[df["gamma_deg"] == selected_gamma].pivot(index="velocity_m_s", columns="altitude_km", values="range_km")
    vmin = dfs.values.min()
    vmax = dfs.values.max()

    img = ax.imshow(
        dfs.values,
        extent=(float(dfs.columns.min()), float(dfs.columns.max()), float(dfs.index.min()), float(dfs.index.max())),
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
    )

    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"{selected_gamma} deg Gamma")

    if fig is not None:
        fig.suptitle(f"Ballistic Range Table for {table_name}")
        fig.tight_layout(pad=1.2)
        _ = fig.colorbar(img, ax=ax, fraction=0.2, pad=0.04, label="Range (km)")
        return fig

    return None
