from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, Normalize

from clamp_analysis.kinematics.geometry import CLAMPS


def dark_to_light_cmap(name: str, dark: str, light: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [dark, light])


def blend_hex(base: str, target: str, target_weight: float) -> str:
    base = base.lstrip("#")
    target = target.lstrip("#")
    base_rgb = [int(base[idx : idx + 2], 16) for idx in range(0, 6, 2)]
    target_rgb = [int(target[idx : idx + 2], 16) for idx in range(0, 6, 2)]
    mixed = [
        round((1.0 - target_weight) * base_val + target_weight * target_val)
        for base_val, target_val in zip(base_rgb, target_rgb)
    ]
    return "#" + "".join(f"{value:02X}" for value in mixed)


def matched_hue_cmap(name: str, light: str) -> LinearSegmentedColormap:
    dark = blend_hex("#000000", light, 0.12)
    lightened = blend_hex("#FFFFFF", light, 0.78)
    return dark_to_light_cmap(name, dark, lightened)


CLAMP_NORM = Normalize(vmin=min(CLAMPS), vmax=max(CLAMPS))

HUMAN_CMAP = matched_hue_cmap("tracker_human_purple", "#CC79A7")

ONE_D_CMAP = matched_hue_cmap("tracker_calibrated_1d_orange", "#E69F00")

TWO_D_CMAP = matched_hue_cmap("tracker_calibrated_2d_blue", "#56B4E9")

CLAMP_CMAP = HUMAN_CMAP
