"""
Basic functions for hexgrid operation.
"""
from typing import Union
import numpy as np
import matplotlib
from matplotlib.patches import RegularPolygon


CUBE_DIRECTION = np.array([
    [+1, -1, 0], [+1, 0, -1], [0, +1, -1],
    [-1, +1, 0], [-1, 0, +1], [0, -1, +1]
])


def cube_to_axial(cube):
    """Convert from cube to axial coordinates."""
    ax = np.array(cube)
    if len(ax.shape) == 1:
        ax = np.expand_dims(ax, axis=0)
    q = ax[:, 0]  # x
    r = ax[:, 2]  # z
    return np.vstack([q, r]).T.squeeze()


def axial_to_cube(axial):
    """Convert from axial to cube coordinates."""
    ax = np.array(axial)
    if len(ax.shape) == 1:
        ax = np.expand_dims(ax, axis=0)
    x = ax[:, 0]  # q
    z = ax[:, 1]  # r
    y = -x - z
    return np.vstack([x, y, z]).T.squeeze()


def get_canvas_dim(axial):
    """Get canvas dimension in axial coordinate.

    This function is used to produce the ``canvas_dim`` argument in
    the function ``axial_to_k()``.

    Parameters
    ----------
    axial
        axial coorindates for the entire canvas

    Returns
    -------
    (nx, ny) dimension of the entire canvas in axial coordinate
    """
    return axial.max(axis=0) - axial.min(axis=0) + 1


def axial_to_k(axial, canvas_axial):
    """Convert axial coordinate to flattened linear index.

    Argument ``canvas_dim`` is calculated from ``get_canvas_dim()``.

    Parameters
    ----------
    axial : ndarray
        axial coordinate indices for hexgrid to be converted
    canvas_axial : ndarray
        axial coordinate of the entire hex canvas

    Returns
    -------
    k : ndarray
        flattened indices
    """
    # min of canvas axial coordinate for np.unravel_index operation
    # this is the "lower-left corner" of the coordinate tuple
    canvas_shift = canvas_axial.min(axis=0)

    # numpy.ravel_multi_index: converts a tuple of index arrays into
    # an array of flat indices, applying boundary modes to the multi-index
    return np.ravel_multi_index((axial - canvas_shift).T,
                                get_canvas_dim(canvas_axial))


def k_to_axial(k, canvas_axial):
    """Convert flattened linear index to axial coordinate.

    Parameters
    ----------
    k : ndarray
        flattened indices
    canvas_axial : ndarray
        axial coordinate of the entire hex canvas

    Returns
    -------
    Cleaned array of flattened indices
    """
    # min of canvas axial coordinate for np.unravel_index operation
    canvas_shift = canvas_axial.min(axis=0)

    # np.unravel_indexL converts a flat index or array of flat indices
    # into a tuple of coordinate arrays
    return np.array(np.unravel_index(
        k, get_canvas_dim(canvas_axial))).T + canvas_shift


def cube_to_k(cube, canvas_axial):
    """
    Convert from cube to flattened linear index.

    Parameters
    ----------
    cube : ndarray
        cube coordinate
    canvas_axial : ndarray
        axial coordinate of the entire hex canvas

    Returns
    -------
    Array of flattened indices
    """
    return axial_to_k(cube_to_axial(cube), canvas_axial)


def k_to_cube(k, canvas_axial):
    """
    Convert from flattened linear index to cube.

    Parameters
    ----------
    k : ndarray
        flattened indices
    canvas_axial : ndarray
        axial coordinate of the entire hex canvas

    Returns
    -------
    Array of cube coordinates
    """
    return axial_to_cube(k_to_axial(k, canvas_axial)),


def get_cube_plot_xy(cube):
    """
    get xy coordinates to plot a single hex grid.
    """
    y_plot = -cube[2]            # '-z' in cube coorindate
    x_plot = cube[0] - cube[1]   # 'x-y' in cube coorindate
    x_scale = np.sin(np.pi / 3)
    y_scale = 1 + np.sin(np.pi / 6)
    return np.array([x_plot * x_scale, y_plot * y_scale])


def cube_add(cube, vec):
    """
    Move cube torward the direction and distance specified by vec.

    Reference: https://www.redblobgames.com/grids/hexagons/#neighbors
    """
    return cube + vec


def cube_distance(cube1, cube2):
    """
    Calculate the distance between cube1 and cube2.

    Reference: https://www.redblobgames.com/grids/hexagons/#distances-cube
    """
    return np.abs(cube1 - cube2).max()


def cube_neighbor(cube, direction, distance=1):
    """Get a neighboring hexgrid given the direction and distance.

    Reference: https://www.redblobgames.com/grids/hexagons/#neighbors

    Parameters
    ----------
    cube
        center of current hexgrid in cube coordinate
    direction
        direction of the neighbor (6 choices defined in CUBE_DIRECTION)
    distance
        distance of the neighboring grid

    Returns
    -------
    center of the neighboring grid in cube coorindate
    """
    return cube_add(cube, CUBE_DIRECTION[direction, :] * distance)


def cube_within_radius(cube: Union[list, np.array], radius: int):
    """Get all cube coordinates within a patch with a certain radius.

    Parameters
    ----------
    cube
        cube coordinate of the center of the patch
    r
        radius of the patch

    Returns
    -------
    array of cube coorindates
    """
    if isinstance(cube, list):
        cube = np.array(cube)
    out_shift = []
    for xx in np.arange(-radius, radius + 1):
        for yy in np.arange(max([-radius, -xx - radius]), min([+radius, -xx + radius]) + 1):
            zz = -xx - yy
            out_shift.append([xx, yy, zz])
    return np.array(out_shift) + cube


def plot_one_hexgrid(ax, cube, gridcolor='w', gridalpha=0.3):
    """Plot one hexgrid with specified color and transparency.

    Parameters
    ----------
    ax
        axis handle
    cube : array-like
        cube coorindate of hexagon
    gridcolor, gridalpha
        color and transparency of hexagon
    """
    xy = tuple(get_cube_plot_xy(cube))
    hhex = RegularPolygon(xy,
                          numVertices=6, radius=1,
                          edgecolor='0.5', facecolor=gridcolor,
                          alpha=gridalpha)
    return ax.add_patch(hhex)


def plot_hexgrids(ax, cube_array, cube_color,
                  gridalpha=0.9, cmap='Blues', vmin_vmax=None):
    """Plot multiple hexagons in specified colors.

    Parameters
    ----------
    ax
        axis handle
    cube_array
        hexgrid locations in cube coordinate
    cube_color
        color of hexgrids defined by cube_array
    gridalpha
        transparency of hexagon
    cmap
        colormap
    vmin_vmax : array-like
        min and max color values

    Returns
    -------
    im
        image color mappable
    cbar
        colorbar, use ``set_label()`` to label the colorbar
    """
    if type(cube_color) == str:
        for cube in cube_array:
            plot_one_hexgrid(ax, cube,
                             gridcolor=cube_color, gridalpha=gridalpha)
        im = None
        cbar = matplotlib.cm.ScalarMappable(cmap=cmap)
    else:
        if vmin_vmax is None:
            if np.unique(cube_color).size == 1:  # all grids have the same value
                vmin = cube_color[0] - 0.5
                vmax = cube_color[0] + 0.5
            else:
                vmin = np.nanmin(cube_color)
                vmax = np.nanmax(cube_color)
        else:
            vmin, vmax = vmin_vmax

        ccmap = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = ccmap(norm(cube_color), bytes=True).squeeze().astype('f8') / 255
        for cube, cc in zip(cube_array, colors):
            plot_one_hexgrid(ax, cube,
                             gridcolor=cc, gridalpha=gridalpha)
        im = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = matplotlib.pyplot.colorbar(im, ax=ax)

    return im, cbar
