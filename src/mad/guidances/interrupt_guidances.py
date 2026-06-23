from mad.guidances.base_guidances import GuidanceInterrupts
import numpy as np


def interrupt_at_t(interrupts: GuidanceInterrupts, t: float) -> bool:
    """Interrupt the current guidance law when simulation time reaches a value (s)."""
    return interrupts.t is not None and interrupts.t > t


def interrupt_at_travelled_distance(interrupts: GuidanceInterrupts, travelled_distance_m: float) -> bool:
    """Interrupt the current guidance law when missile has travelled a distance (m)."""
    return interrupts.travelled_distance_m is not None and interrupts.travelled_distance_m > travelled_distance_m


def interrupt_at_altitude(interrupts: GuidanceInterrupts, reached_altitude_m: float) -> bool:
    """Interrupt the current guidance law when missile reaches a certain altitude (m)."""
    if interrupts.missile is None or interrupts.planet is None:
        raise ValueError("Missile and planet must be provided for this GuidanceInterrupt.")
    altitude = np.linalg.norm(interrupts.missile.position) - interrupts.planet.radius
    return float(altitude) > reached_altitude_m


def interrupt_at_linear_distance_to_target(interrupts: GuidanceInterrupts, reached_distance_m: float) -> bool:
    """Interrupt the current guidance law when missile reaches a certain linear distance to the target (m)."""
    if interrupts.missile is None or interrupts.target is None:
        raise ValueError("Missile and target must be provided for this GuidanceInterrupt.")
    distance_to_target = np.linalg.norm(interrupts.missile.position - interrupts.target.position)
    return float(distance_to_target) < reached_distance_m


def interrupt_at_surface_distance_to_target(interrupts: GuidanceInterrupts, reached_distance_m: float) -> bool:
    """Interrupt the current guidance law when missile reaches a certain surface distance to the target (m)."""
    if interrupts.missile is None or interrupts.target is None or interrupts.planet is None:
        raise ValueError("Missile, target, and planet must be provided for this GuidanceInterrupt.")

    surface_distance = interrupts.planet.surface_distance(interrupts.missile, interrupts.target)
    return surface_distance < reached_distance_m
