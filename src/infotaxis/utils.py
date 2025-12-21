"""
Utility functions for misc operations that support
the discrete infotaxis modules.
"""

import numpy as np
from . import hex_ops


def clean_flattened_index(k, k_canvas):
    """Remove flattened index outside of canvas.

    This is especially useful in clean up the hexgrids covered by
    a beam when the beam axis is near the canvas boundary.

    Parameters
    ----------
    k : ndarray
        array of flattened indices
    k_canvas : ndarray
        flattened indices of the entire canvas

    Returns
    -------
    Cleaned array of flattened indices
    """
    return k[np.isin(k, k_canvas)]


def clean_beam_cover_hex(cover_axial, canvas_axial):
    # TODO: this probably works with rectangular grids as well
    in_bnd = np.logical_and(cover_axial >= canvas_axial.min(axis=0),
                            cover_axial <= canvas_axial.max(axis=0))
    idx_want = ~np.any(~in_bnd, axis=1).T  # idx within axial boundary
    k_cover = hex_ops.axial_to_k(cover_axial[idx_want, :], canvas_axial)
    k_cover = clean_flattened_index(k_cover,
                                    hex_ops.axial_to_k(canvas_axial, canvas_axial))
    return hex_ops.k_to_axial(k_cover, canvas_axial)


def beam_shape_hex(cube_ctr, canvas_axial, beam_radius):
    """Generate beam coverage in hexgrid.

    Parameters
    ----------
    cube_ctr
        cube coordinate of beam axis
    canvas_axial
        axial coordinate of the entire canvas
    beam_radius
        radius of beam coverage
    """
    # Generate coverage
    cover_cube = hex_ops.cube_within_radius(cube_ctr, beam_radius)

    # Clean up cells outside of canvas
    cover_axial = clean_beam_cover_hex(hex_ops.cube_to_axial(cover_cube), canvas_axial)

    return hex_ops.axial_to_cube(cover_axial)


def beam_shape(beam_cover_dim=None):
    """Generate beam shape wrt beam axis.
    """
    if beam_cover_dim == 5:
        beam_shape_i = [np.arange(-2, 3)] * 5
        beam_shape_j = np.copy(beam_shape_i).T
    else:
        beam_shape_i = [np.arange(-1, 2)] * 3
        beam_shape_j = np.copy(beam_shape_i).T

    beam_shape_i = np.reshape(beam_shape_i, (-1, 1))
    beam_shape_j = np.reshape(beam_shape_j, (-1, 1))

    return np.hstack((beam_shape_i, beam_shape_j))


def clean_beam_cover(cover_ij, d):
    """Remove beam axis entries outside of search space boundary.
    """
    a = [x >= d.shape[0] or x < 0 for x in cover_ij[:, 0]]
    b = [y >= d.shape[1] or y < 0 for y in cover_ij[:, 1]]
    del_idx = np.where(np.array([x or y for x, y in zip(a, b)]) == True)
    return np.delete(cover_ij, del_idx, 0)


def loge_to_log2(x):
    """Convert from log_e to log_2.
    """
    if x == 0:
        return 0
    else:
        return x/np.log(2)
    

def param_string_to_int(param_dict):
    """Convert dict key from string to integer.
    """
    param_dict_new = {}
    for k, v in param_dict.items():
        if not isinstance(k, int):
            k = int(k)
        param_dict_new[k] = v
    return param_dict_new
