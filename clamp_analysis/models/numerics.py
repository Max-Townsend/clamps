def wrap_angle(angle_deg):
    """Wrap angles (degrees) to [-180, 180)."""
    return (angle_deg + 180) % 360 - 180
