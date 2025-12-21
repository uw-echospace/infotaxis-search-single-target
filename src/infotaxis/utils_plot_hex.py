"""
Plotting utilities for discrete infotaxis in hexagonal grids.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from . import hex_ops
from .utils import beam_shape_hex


PLOT_ATTRS = dict(
    ticklabel_fontsize = 12,
    title_fontsize = 14,
    axislabel_fontsize = 14,
    legend_fontsize = 12,
    symbol_color = 'orangered',
)


def plot_beam_axis(ax, axis_ij):
    """Plot beam axis on the map.
    """
    ax.plot(axis_ij[0], axis_ij[1], 'x', markersize=8,
            markeredgecolor='m', markeredgewidth=2)


def plot_loc(ax, axis_ij, sym='o', color='m', size=10):
    """Plot a symbol at specific location on the map.

    Parameters
    ----------
    ax
        axis to be plotted on
    axis_ij
        location of symbol
    sym
        symbol to be plotted
    color
        color of the symbol
    size
        size of symbol
    """
    ax.plot(axis_ij[0], axis_ij[1],
            markersize=size,
            marker=sym,
            markerfacecolor="None",
            markeredgecolor=color,
            markeredgewidth=2)


def plot_dk_map(ax, map_type, canvas_cube, canvas_vmap,
                beam_cube, echo_type_str, env_attrs,
                plot_attrs=PLOT_ATTRS, cmap="Blues_r", show_title=True):
    """Plot the probability map of target location or map of expected entropy.

    Parameters
    ----------
    ax
        axis to be plotted on
    map_type : str
        type of map: 'entropy' or something else
    canvas_cube : ndarray
        [Nx3] array storing the cube coordinate of the canvas
    canvas_vmap : ndarray
        probability map in corresponding locations as ``canvas_cube``
    beam_cube : ndarray
        [Nx3] array storing all beam axis location until now, in cube coordinate
    echo_type_str : str
        type of echo return from previous ping
    env_attrs : dict
        locations of target, peg, etc. to be shown on the map
    plot_attrs : dict
        plotting options: color of map and symbols, etc.
    """
    im, cbar = hex_ops.plot_hexgrids(ax,
                                     cube_array=canvas_cube,
                                     cube_color=canvas_vmap,
                                     cmap=cmap)
    ax.tick_params(labelsize=plot_attrs['ticklabel_fontsize'])
    ax.axis('equal')

    # Title
    if show_title:
        ttype = ax.text(x=0.05, y=1.05, s=map_type,
                        transform=ax.transAxes,
                        fontsize=plot_attrs['title_fontsize'])
        techo = ax.text(x=0.5, y=1.05, s=echo_type_str,
                        transform=ax.transAxes,
                        fontsize=plot_attrs['title_fontsize'])
        if echo_type_str == 'miss' or echo_type_str == 'false alarm':
            techo.set(**{'text': echo_type_str,
                        'color': 'r',
                        'fontweight': 'bold'})

    # Beam and beam trace
    beam_plotxy = np.array([hex_ops.get_cube_plot_xy(x).squeeze()
                            for x in beam_cube])
    if map_type in ["entropy", "prob target", "prob correct"]:
        # entropy: infotaxis searcher
        # prob target: MAP searcher
        # prob correct: MAP future searcher
        beam_trace = beam_plotxy[-6:, :]    # plot last 5 beam axis loc and the next
        beam_next = beam_plotxy[-1, :]   # next beam axis loc
    else:
        beam_trace = beam_plotxy[-6:-1, :]  # plot last 5 beam axis loc
        beam_next = None

    ax.plot(beam_trace[:, 0], beam_trace[:, 1], marker='.',
            color=plot_attrs['symbol_color'], alpha=0.5, linewidth=3, markersize=8)
    if beam_plotxy.shape[0] > 1:  # if there was a last beam
        plot_loc(ax=ax, axis_ij=beam_plotxy[-2, :],  # plot last beam axis
                 sym='+', color=plot_attrs['symbol_color'], size=10)
    if beam_next is not None:
        plot_loc(ax=ax, axis_ij=beam_next,  # plot last beam axis
                 sym='x', color=plot_attrs['symbol_color'], size=8)

    # Target and peg locs
    if 'target_loc' in env_attrs:
        plot_loc(ax, hex_ops.get_cube_plot_xy(env_attrs['target_loc']),
                 sym='o', color=plot_attrs['symbol_color'])  # target location
    if 'peg_loc' in env_attrs:   # if there is a peg
        plot_loc(ax, hex_ops.get_cube_plot_xy(env_attrs['peg_loc']),
                 sym='^', color=plot_attrs['symbol_color'])  # peg location

    # Colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.15)
    # cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    # # cbar.ax.set_xticklabels(np.linspace(0,1,5))
    # cax.tick_params(labelsize=plot_attrs['ticklabel_fontsize'])
    cbar.ax.tick_params(labelsize=plot_attrs['ticklabel_fontsize'])


def plot_entropy_seq(ax, h_actual_all, h_est_all, plot_attrs):
    """Plot entropy variation across pings.

    Parameters
    ----------
    ax
        axis to be plotted on
    h_actual_all : array_like
        actual entropy at each ping
    h_est_all : array_like
        estimated entropy at each ping
    plot_attrs : dict
        plotting options: color of map and symbols, etc.
    """
    pingnum = len(h_actual_all)   # current ping number
    ax.plot(np.arange(pingnum), h_actual_all,  # actual entropy
            color='k', marker='.', markersize=8, label='h')
    ax.plot(pingnum, h_est_all[pingnum-1].min(),  # minimum h_est after next ping
            marker='x', markeredgecolor=plot_attrs['symbol_color'],
            markersize=8, markeredgewidth=2, linestyle='none',
            label='Next est')
    if pingnum != 1:  # if more than first ping
        ax.plot(pingnum-1, h_est_all[pingnum-2].min(),  # minimum h_est before current ping
                marker='+',
                markeredgecolor=plot_attrs['symbol_color'],
                markerfacecolor=plot_attrs['symbol_color'],
                markersize=10, markeredgewidth=2, linestyle='none',
                label='Last est')
    if pingnum <= 5:
        ax.set_xticks(np.arange(6))
    elif pingnum <= 10:
        ax.set_xticks(np.arange(0, pingnum + 2, 2))
    else:
        ax.set_xticks(np.arange(0, pingnum + 5, 5))
    ax.set_xlabel('Ping number', fontsize=plot_attrs['axislabel_fontsize'])
    ax.set_ylabel('Entropy', fontsize=plot_attrs['axislabel_fontsize'])
    ax.tick_params(labelsize=plot_attrs['ticklabel_fontsize'])
    ax.grid()
    plt.legend(fontsize=plot_attrs['legend_fontsize'])


def get_echo_type_str(echo_value, echo_type):
    """Obtain string for echo type.

    Parameters
    ----------
    echo_value : int
        whether or not an echo is received (0-no, 1-yes)
    echo_type : int
        whether or not it is a true echo or true miss (0-no, 1-yes)
    """
    if echo_value == 1:
        if echo_type == 1:
            echo_type_str = 'echo'
        else:
            echo_type_str = 'false alarm'
    elif echo_value == 0:
        if echo_type == 1:
            echo_type_str = 'no echo'
        else:
            echo_type_str = 'miss'
    else:  # set up for when no previous info available
        echo_type_str = ''
    return echo_type_str


def plot_latest_update(simulation, plot_attrs=PLOT_ATTRS, orientation='horizontal'):
    """Plot maps of target probability, entropy, and entropy variation sequence.

    Parameters
    ----------
    simulation : OneTargetHex object
        an OneTargetHex object
    plot_attrs : dict
        plotting attributes, default:
        plot_attrs = dict(ticklabel_fontsize=12,
                          title_fontsize=14,
                          axislabel_fontsize=14,
                          legend_fontsize=12,
                          cmap='Blues',
                          symbol_color='r')
    orientation : str
        'horizontal' - entropy plot to the right of dk and h_est map (default)
        'vertical' - entropy plot below dk and h_est map

    Returns
    -------
        figure handle
    """
    # Get echo type
    echo_value = simulation.echo_value_all[-1]
    echo_type = simulation.echo_type_all[-1]
    echo_type_str = get_echo_type_str(echo_value=echo_value, echo_type=echo_type)

    pingnum = len(simulation.h_actual_all)

    # Construct env_attrs for plotting
    env_attrs = dict(target_loc=simulation.target_cube)

    # Get beam radius
    beam_r = simulation.radius_last

    if orientation == 'horizontal':
        fig = plt.figure(figsize=(16, 3))
        gs_L = fig.add_gridspec(1, 2, left=0.05, right=0.50)
        gs_R = fig.add_gridspec(1, 1, left=0.56, right=0.95)
    elif orientation == 'vertical':
        fig = plt.figure(figsize=(7.5, 6))
        gs_L = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.55, top=0.95)
        gs_R = fig.add_gridspec(1, 1, left=0.05, right=0.95, bottom=0.05, top=0.42)
    else:
        ValueError("Specified figure orientation not supported.")

    # Target probability map
    ax = fig.add_subplot(gs_L[0, 0])
    plot_dk_map(ax, 'P_T',
                canvas_cube=simulation.canvas_cube,
                canvas_vmap=simulation.d,
                beam_cube=simulation.beam_cube_all,
                echo_type_str=echo_type_str,
                env_attrs=env_attrs,
                plot_attrs=plot_attrs)

    # Expected entropy map
    ax = fig.add_subplot(gs_L[0, 1])
    plot_dk_map(ax, 'entropy',
                canvas_cube=simulation.canvas_cube,
                canvas_vmap=simulation.h_est[beam_r],
                beam_cube=simulation.beam_cube_all,
                echo_type_str=echo_type_str,
                env_attrs=env_attrs,
                plot_attrs=plot_attrs)

    # Next beam coverage
    hex_ops.plot_hexgrids(ax, cube_array=simulation.beam_last_cover_cube,  # plot next beam coverage
                          cube_color='y', gridalpha=0.4)

    # Entropy variation
    ax = fig.add_subplot(gs_R[0, 0])
    plot_entropy_seq(ax, simulation.h_actual_all, simulation.h_est_all, plot_attrs)
    ax.set_title('Ping #%02d     entropy variation' % pingnum, fontsize=14)

    plt.show()

    return fig


def plot_step_from_nc(ds, ping_num, plot_attrs=PLOT_ATTRS, orientation='horizontal'):
    """Plot update after the specified ping number from a .nc file.

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing the run data
    ping_num : int
        ping number after which the update to be plotted
    plot_attrs : dict
        plotting attributes, default:
        plot_attrs = dict(ticklabel_fontsize=12,
                          title_fontsize=14,
                          axislabel_fontsize=14,
                          legend_fontsize=12,
                          cmap='Blues',
                          symbol_color='r')
    orientation : str
        'horizontal' - entropy plot to the right of dk and h_est map (default)
        'vertical' - entropy plot below dk and h_est map

    Returns
    -------
        figure handle
    """
    if ping_num > ds.steps.size:
        ValueError('Requested pingnum larger than numbers in file. Exiting...')
    else:
        # Get echo type
        echo_value = ds['echo_value_all'].isel(steps=ping_num).values
        echo_type = ds['echo_type_all'].isel(steps=ping_num).values
        echo_type_str = get_echo_type_str(echo_value=echo_value, echo_type=echo_type)

        if orientation == 'horizontal':
            fig = plt.figure(figsize=(16, 3))
            gs_L = fig.add_gridspec(1, 2, left=0.05, right=0.50)
            gs_R = fig.add_gridspec(1, 1, left=0.56, right=0.95)
        elif orientation == 'vertical':
            fig = plt.figure(figsize=(7, 5.5))
            gs_L = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.53, top=0.95)
            gs_R = fig.add_gridspec(1, 1, left=0.05, right=0.95, bottom=0.05, top=0.45)
        else:
            ValueError("Specified figure orientation not supported.")

        # Construct env_attrs for plotting
        env_attrs = dict(target_loc=ds.attrs['target_loc'])

        # Get beam radius
        beam_r = ds["radius_all"].isel(steps=ping_num).values

        # Target probability map
        ax = fig.add_subplot(gs_L[0, 0])
        plot_dk_map(
            ax, 'P_T',
            canvas_cube=ds['canvas_cube'].values,
            canvas_vmap=ds['d_all'].isel(steps=ping_num).values,
            beam_cube=ds['beam_cube_all'].isel(steps=slice(0, ping_num + 1)).values,
            echo_type_str=echo_type_str,
            env_attrs=env_attrs,
            plot_attrs=plot_attrs,
            cmap="pink",
        )
        ax.patch.set_visible(False)
        ax.axis('off')

        # Expected entropy map
        ax = fig.add_subplot(gs_L[0, 1])
        plot_dk_map(
            ax, 'entropy',
            canvas_cube=ds['canvas_cube'].values,
            canvas_vmap=ds['h_est_all'].isel(steps=ping_num).sel(beam_radius=beam_r).values,
            beam_cube=ds['beam_cube_all'].isel(steps=slice(0, ping_num + 1)).values,
            echo_type_str=echo_type_str,
            env_attrs=env_attrs,
            plot_attrs=plot_attrs,
            cmap="Blues_r",
        )

        # Next beam coverage
        canvas_axial = hex_ops.cube_to_axial(ds['canvas_cube'])  # the entire canvas in axial coordinate
        last_beam_cover_cube = beam_shape_hex(cube_ctr=ds['beam_cube_all'].isel(steps=ping_num).values,
                                                canvas_axial=canvas_axial,
                                                beam_radius=beam_r)
        hex_ops.plot_hexgrids(ax, cube_array=last_beam_cover_cube,
                                cube_color='y', gridalpha=0.4)

        ax.patch.set_visible(False)
        ax.axis('off')

        # Entropy variation
        ax = fig.add_subplot(gs_R[0, 0])
        plot_entropy_seq(
            ax,
            ds['h_actual_all'].isel(steps=slice(0, ping_num + 1)).values,
            ds['h_est_all'].isel(steps=slice(0, ping_num + 1)).values,
            plot_attrs)
        ax.set_title('Ping #%02d     entropy variation' % (ping_num + 1),
                        fontsize=plot_attrs['title_fontsize'])

        plt.show()

        return fig
