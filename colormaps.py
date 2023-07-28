import matplotlib.colors as mcolors
import numpy as np


def alpha_to_red_colormap():
    # Define the colormap using alpha values from 0 to 255 (fully transparent to fully opaque)
    alphas = np.linspace(0, 128, 256)

    # Create a color map that transitions from transparent to red
    red_colormap = [(1.0, 0.0, 0.0, alpha / 255.0) for alpha in alphas]

    # Create a custom colormap using LinearSegmentedColormap
    colormap = mcolors.LinearSegmentedColormap.from_list(
        'alpha_to_red', red_colormap)

    return colormap


def alpha_to_blue_colormap():
    # Define the colormap using alpha values from 0 to 255 (fully transparent to fully opaque)
    alphas = np.linspace(0, 128, 256)

    # Create a color map that transitions from transparent to red
    red_colormap = [(0.0, 0.0, 1.0, alpha / 255.0) for alpha in alphas]

    # Create a custom colormap using LinearSegmentedColormap
    colormap = mcolors.LinearSegmentedColormap.from_list(
        'alpha_to_red', red_colormap)

    return colormap
