"""Shared matplotlib styling for the analysis notebooks.

One palette and one axes-cleaning helper, so every figure across the
analysis notebooks reads as a single system. Import what you need::

    from toolkit.plotting import BLUE, INK, MUTED, NEUTRAL, clean_axes
"""

BLUE = "#2a78d6"  # single hue: one measure, light-to-dark reserved for magnitude
NEUTRAL = "#e2e2e0"
INK, MUTED = "#1a1a19", "#6e6e6a"


def clean_axes(ax):
    """Recessive chart furniture: data in front, scaffolding in back.

    Hides the top/right spines, mutes the remaining spines and ticks,
    and keeps tick labels in ink so the data marks carry the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to restyle, modified in place.

    Returns
    -------
    None
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
