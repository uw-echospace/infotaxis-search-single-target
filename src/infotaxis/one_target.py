"""
Discrete grid infotaxis with precisely one target present in the search space.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import xarray as xr
from scipy.special import xlogy

from . import hex_ops
from .utils import beam_shape_hex, loge_to_log2, param_string_to_int
from .core import PARAM_ANIMAL_DEFAULT, PARAM_ECHO_DEFAULT
from .utils_plot_hex import PLOT_ATTRS, get_echo_type_str, plot_dk_map, plot_entropy_seq


class OneTargetHex(object):
    """Class for infotaxis search in single target search scenario.

    Attributes
    ----------
    param_animal or param_echo : dict of dict
        pm or pfa; default=0.01 (required)
            - simu_param[beam_radius]['pm_const']     
            - simu_param[beam_radius]['pfa_const']

        pm or pfa as a function of distance to beam aim (r) (optional)
        *_beam_sigma_r is the standard deviation of the Gaussian distribution used in beam_dep_curve
            - simu_param[beam_radius]['pm_beam_sigma_r']
            - simu_param[beam_radius]['pfa_beam_sigma_r']
            - simu_param[beam_radius]['pm_beam_scale']
            - simu_param[beam_radius]['pfa_beam_scale']

        pm or pfa as a function of search space (optional)
            - simu_param[beam_radius]['pm_mtx']
            - simu_param[beam_radius]['pfa_mtx']

    search_dict : dict
        dictionary for early stopping criteria and other search params
        None (default): next beam can be anywhere in the canvas
        'neighbor_radius'
            next beam can only be within neighbor_r grids from the current aim
        'move_dir' (not yet implemented)
            next beam can only be within the move_dir section of the canvas wrt current aim
    canvas_radius : int
        radius of hexgrid canvas
    beam_radius : int or list
        choices of beam radius
    target_cube
        target location in cube coordinate, default is (1, -2, 1)
    aim_start_cube
        starting beam location in cube coordinate, default to None and
        will cause random selection based on h_est calculated from the initial d
    beam_radius_start
        starting beam radius, default to the largest of the beam radius choices
    d : ndarray
        probability map of target location
    h_est : ndarray
        expected entropy for all possible next beam axis

    """

    def __init__(
        self,
        search_rule="infotaxis",
        canvas_radius=5,
        beam_radius=[1],
        target_cube=(1, -2, 1),
        aim_start_cube=None,
        beam_radius_start=None,
        param_animal=None,
        param_echo=None,
        search_dict=None
    ):
        # Search space params
        self.search_rule = search_rule
        self.canvas_radius = canvas_radius
        self.target_cube = target_cube
        if not isinstance(beam_radius, list):
            raise ValueError("beam_radius should always be a list, even if it is a 1-element list")
        else:
            self.beam_radius = [int(r) for r in beam_radius]  # convert to int in case of strings
        self.search_dict = search_dict

        # pm/pfa params
        # Echo generation
        if param_echo is None:
            self.param_echo = PARAM_ECHO_DEFAULT
        else:
            self.param_echo = param_echo
        # Animal expectation
        if param_animal is None:
            self.param_animal = PARAM_ANIMAL_DEFAULT
        else:
            self.param_animal = param_animal
        # Convert beam footprint key from string to int
        self.param_animal = param_string_to_int(self.param_animal)
        self.param_echo = param_string_to_int(self.param_echo)

        # Derived params
        self.canvas_cube = hex_ops.cube_within_radius((0, 0, 0), canvas_radius)
        self.canvas_axial = hex_ops.cube_to_axial(self.canvas_cube)   # the entire canvas in axial coordinate
        self.k_canvas = hex_ops.axial_to_k(self.canvas_axial, self.canvas_axial)  # flat indices of the entire canvas
        self.table_X0, self.table_X1 = self.calc_update_table()

        # Initialize update variables
        self.d = np.array([1 / self.canvas_cube.shape[0]] * self.canvas_cube.shape[0])  # indexed by array_idx_*
        self.echo_value = None
        self.echo_type = None
        self.h_actual = None
        self.p_est = None  # p_est is a function of possible next beam axis
        self.h_est = None  # h_est is a function of possible next beam axis
        self.h_X0_est = None
        self.h_X1_est = None
        self.p_X0_est = None
        self.p_X1_est = None

        # Initialize variables that will be overwritten below
        self.aim_last_cube = None        # last beam aim in cube coordinates
        self.beam_cover_last_cube = None  # last beam coverage in cube coorindates; init by get_beam_coverage below
        self.radius_last = None           # last beam radius

        # Initialize params based on starting condition
        self.get_est_ph()     # get the starting p_est, h_est, h_X*_est and p_X*_est
        self.get_h_actual()  # get the starting h_actual
        if aim_start_cube is not None:
            self.aim_last_cube = aim_start_cube
            if beam_radius_start is not None:
                self.radius_last = beam_radius_start
            else:
                print('Initial beam aim is given, but not initial beam radius.')
                self.radius_last = max(self.beam_radius)
                print(f'Set initial beam radius to the largest of beam radius choices: {self.radius_last}')
            self.get_beam_coverage()  # beam aim already selected, so update this separately
                                      # normally updated within get_nex_beam
        else:  # if no initial beam aim given, pick one based on infotaxis rule
            self.radius_last = max(self.beam_radius)  # initialize beam radius in case using MAP searcher
                                                      # (which does not pick beam radius)
            self.get_next_beam()  # get beam aim and radius based on initial h_est

        # Initialize storage variables: lists
        # Values from the last ping
        self.beam_cube_all = [np.copy(self.aim_last_cube)]  # last; set to the starting beam aim
        self.radius_all = [np.copy(self.radius_last)]  # last; set to the initial
        self.echo_value_all = [None]
        self.echo_type_all = [None]
        # Values based on current d, already updated _after_ the last ping
        self.d_all = [np.copy(self.d)]  # current
        self.h_actual_all = [np.copy(self.h_actual)]  # calculated using current d
        self.p_est_all = [np.array(list(self.p_est.values()))]  # max posterior estimate based on current d
        self.h_est_all = [np.array(list(self.h_est.values()))]  # h estimate based on current d

    def update_record(self):
        """Save all updated parameters.
        """
        # Save iteration results
        # Values from the last ping
        self.beam_cube_all.append(np.copy(self.aim_last_cube))
        self.radius_all.append(np.copy(self.radius_last))
        self.echo_value_all.append(self.echo_value)
        self.echo_type_all.append(self.echo_type)
        # Values based on current d, already updated _after_ the last ping
        self.d_all.append(np.copy(self.d))
        self.p_est_all.append(np.array(list(self.p_est.values())))
        self.h_est_all.append(np.array(list(self.h_est.values())))
        self.h_actual_all.append(np.copy(self.h_actual))

    def beam_dep_curve(self, r_cell, beam_sigma_r, beam_scale):
        """
        Producing a pm/pfa curve using the Gaussian distribution
        with standard deviation dep_sigma.

        Parameters
        ----------
        r_cell : number
            grid distance to beam aim
        beam_sigma_r : number
            standard deviation of the Gaussian distribution
        beam_scale : int
            scale factor for the beam-dependent curve

        Returns
        -------
        A function that takes in r and outputs a value between 0 and 1
        according the Gaussian distribution with standard deviation dep_sigma.
        """
        val = np.exp(-0.5 * (r_cell / beam_sigma_r) ** 2)
        return (1-val)*beam_scale + 1  # invert the curve as a multiplier

    def pmpfa_fcn(self, fcn_type, k_axis, k_cell, simu_type, beam_radius, beam_scale_default=10, verbose_opt=False):
        """Generate probability of miss (pm) or probability of false alarm (pfa).

        Parameters
        ----------
        fcn_type : str
            whether it is for calculating pm or pfa
        k_axis : array
            flat index of cell on the beam axis
        k_cell : array
            flat index of cell
        simu_type : str
            'animal' -- for calculating expecations
            'echo' -- for generating simulated echo
            if simu_type contain pm_beam_sigma_r or pfa_beam_sigma_r,
            then compute beam-dependent pm or pfa using beam_dep_curve
            pm_beam_scale and pfa_beam_scale default to 10 unless supplied
        beam_radius : int
            beam radius, used to index self.param_animal or self.param_echo
        verbose_opt : bool
            if to print detailed messages

        Returns
        -------
        Probability of miss (pm) or probability of false alarm (pfa) for all cells in the beam footprint
        """
        # TODO: to allow SNR variation depending where in the beam,
        #  we would need k_axis, k_target, dist_opt
        #  These input variables are not included currently.

        # get simu_param
        if simu_type == 'animal':
            simu_param = self.param_animal[beam_radius]
        elif simu_type == 'echo':
            simu_param = self.param_echo[beam_radius]
        else:
            print('%s is not one of the allowed parameter dictionary type.' % simu_type)

        # load background pm or pfa
        if ('%s_const' % fcn_type) in simu_param:
            p_const = simu_param['%s_const' % fcn_type]
        else:
            if verbose_opt:
                print('%s_const not supplied, default to %s_const=0.001' % (fcn_type, fcn_type))
            p_const = 0.001

        # TODO: this part is not working yet, was in an old version
        #  but removed during class implementation. Need to add back.
        # if depending on location on canvas
        if ('%s_mtx' % fcn_type) in simu_param:
            p_mtx_k_cell = simu_param['%s_mtx' % fcn_type].ravel()[k_cell]
            if verbose_opt:
                print('%s_mtx supplied, %s is location dependent.' % (fcn_type, fcn_type))
        else:
            p_mtx_k_cell = np.ones(k_cell.size) * p_const  # default is uniform with pm_const
            if verbose_opt:
                print('No %s_mtx supplied, %s is not location dependent.' % (fcn_type, fcn_type))

        # beam-dependent pm or pfa
        sigma_key = '%s_beam_sigma_r' % fcn_type
        # check if pm_beam_sigma_r or pfa_beam_sigma_r is supplied in simu_param
        if sigma_key not in simu_param:  # if not supplied, not beam-dependent effects
            p_beam = np.ones(k_cell.size)
        else:  # if supplied, compute beam-dependent effects
            # compute distance from each of k_cell to k__get_p_X0
            r_cell = hex_ops.cube_distance(
                hex_ops.k_to_cube(k_cell, self.canvas_axial),
                hex_ops.k_to_cube(k_axis, self.canvas_axial)
            )
            # compute p_beam using r_cell
            p_beam = self.beam_dep_curve(
                r_cell=r_cell,
                beam_sigma_r=simu_param[sigma_key],
                beam_scale=simu_param.get('%s_beam_scale' % fcn_type, beam_scale_default)
            )
            if verbose_opt:
                print('%s_beam_sigma_r supplied, %s is beam dependent.' % (fcn_type, fcn_type))            

        p_target = np.ones(k_cell.size)  # TODO: add back target-dependent pmpfa generation

        # location-dependent pfa applied as a multiplier
        return p_beam * p_target * p_mtx_k_cell

    def calc_update_table(self) -> Tuple[Dict, Dict]:
        """
        Calculate look-up tables for updating the target probability map (dk).

        These look-up tables only depend on pm, pfa, and beam coverage (hence beam radius).

        Returns
        -------
        (table_X0, table_X1)
            two look-up tables for the update matrix to be multiplied with dk,
            in the form of dictionaries indexed by the beam radius and the beam axis location,
            i.e., table_X0[beam_radius][axis_location]
        """
        k_canvas = self.k_canvas

        table_X0 = defaultdict(list)  # table to store update matrix given X=0
        table_X1 = defaultdict(list)  # table to store update matrix given X=1

        # loop through all beam radius
        for beam_r in self.beam_radius:  

            # loop through all axis locations
            for k_axis in k_canvas:
                # New beam cover
                cover_cube = beam_shape_hex(  # beam coverage in cube coordinate
                    cube_ctr=hex_ops.k_to_cube(k_axis, self.canvas_axial),
                    canvas_axial=self.canvas_axial,
                    beam_radius=beam_r)

                # Get flattened indices
                k_in_beam = hex_ops.cube_to_k(cover_cube, self.canvas_axial)  # cells covered in beam
                k_not_in_beam = np.setdiff1d(k_canvas, k_in_beam, assume_unique=True)  # cells not covered by beam

                # array_idx_* are the indices of k in k_canvas
                # since k is NOT a simple sequence of 0, 1, 2, ...
                # but is the unraveled index given a particular (nx,ny) canvas size
                array_idx_in_beam = np.array([np.argwhere(x == k_canvas) for x in k_in_beam]).squeeze()
                array_idx_not_in_beam = np.setdiff1d(np.arange(k_canvas.size), array_idx_in_beam)

                # ---------------------------------------------
                # X=1
                table_X1_tmp = np.empty(k_canvas.shape)

                # For cells covered by the beam
                for kk, idx in zip(k_in_beam, array_idx_in_beam):  # given target at kk
                    pm_kk = self.pmpfa_fcn(fcn_type='pm', k_axis=k_axis, k_cell=kk, simu_type='animal', beam_radius=beam_r)
                    k_in_beam_not_kk = np.setdiff1d(k_in_beam, kk, assume_unique=True)
                    p_X1_Tk_Bk = 1 - pm_kk * np.prod(1 - self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_in_beam_not_kk,
                                                                        simu_type='animal', beam_radius=beam_r))
                    table_X1_tmp[idx] = p_X1_Tk_Bk

                # For cells not covered by the beam
                for kk_n, idx in zip(k_not_in_beam, array_idx_not_in_beam):  # given target at kk_n
                    p_X1_Tk_B0 = 1 - np.prod(1 - self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_in_beam,
                                                                simu_type='animal', beam_radius=beam_r))
                    table_X1_tmp[idx] = p_X1_Tk_B0

                # add entry to master look-up table
                table_X1[beam_r].append(table_X1_tmp)

                # ---------------------------------------------
                # X=0
                table_X0_tmp = np.empty(k_canvas.shape)

                # For cells covered by the beam
                for kk, idx in zip(k_in_beam, array_idx_in_beam):  # given target at kk
                    pm_kk = self.pmpfa_fcn(fcn_type='pm', k_axis=k_axis, k_cell=kk, simu_type='animal', beam_radius=beam_r)
                    k_in_beam_not_kk = np.setdiff1d(k_in_beam, kk, assume_unique=True)
                    p_X0_Tk_Bk = pm_kk * np.prod(1 - self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_in_beam_not_kk,
                                                                    simu_type='animal', beam_radius=beam_r))
                    table_X0_tmp[idx] = p_X0_Tk_Bk

                # For cells not covered by the beam
                for kk_n, idx in zip(k_not_in_beam, array_idx_not_in_beam):  # given target at kk_n
                    p_X0_Tk_B0 = np.prod(1 - self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_in_beam,
                                                            simu_type='animal', beam_radius=beam_r))
                    table_X0_tmp[idx] = p_X0_Tk_B0

                # add entry to master look-up table
                table_X0[beam_r].append(table_X0_tmp)

        return table_X0, table_X1

    def update_X1(self, beam_radius=None, k_axis=None, k_in_beam=None, update_self_d=True, verbose_opt=False):
        """Update dk when an echo was received (X=1).

        Parameters
        ----------
        beam_radius : int
            beam radius (the actual value, not the index),
            default to the last beam radius
        k_axis : array
            flat index of beam axis,
            default to the last beam axis
        k_in_beam : array
            flat indices of cells covered by beam,
            default to coverage of last beam
        update_self_d : bool
            whether to actually update self.d
            if False, return the updated d mtx
        verbose_opt : bool
            whether to print out detailed messages

        Returns
        -------
            if update_self_d=False, return the updated d mtx
        """
        # Use last beam if not specified
        if k_axis is None:
            k_axis = hex_ops.cube_to_k(self.aim_last_cube, self.canvas_axial)
        if k_in_beam is None:
            k_in_beam = hex_ops.cube_to_k(self.beam_cover_last_cube, self.canvas_axial)

        # if beam radius not specified
        if beam_radius is None:
            beam_radius = self.radius_last

        # Get corresponding array indices
        axis_array_idx = np.argwhere(k_axis == self.k_canvas).squeeze()
        cover_array_idx = np.array([np.argwhere(x == self.k_canvas) for x in k_in_beam]).squeeze()

        if np.all(self.d[cover_array_idx] == 0):  # if dk in all cells are 0
            if verbose_opt:
                print('All cells in beam has probability 0. No updates performed.')
            dk_new = self.d
        else:
            dk_new = self.d * self.table_X1[beam_radius][axis_array_idx]  # update dk
            dk_new = dk_new / np.sum(dk_new)  # normalize

        # Return value if update_self_d=False
        # This is necessary, because in get_h_est we need to "pretend" to update map
        # to get dk map after receiving the next echo outcome
        if update_self_d:
            self.d = dk_new
        else:
            return dk_new

    def update_X0(self, beam_radius=None, k_axis=None, k_in_beam=None, update_self_d=True, verbose_opt=False):
        """Update dk when NO echo was received (X=0).

        Parameters
        ----------
        beam_radius : int
            beam radius (the actual value, not the index),
            default to the last beam radius
        k_axis : array
            flat index of beam axis,
            default to the last beam axis
        k_in_beam : array
            flat indices of cells covered by beam,
            default to coverage of last beam
        update_self_d : bool
            whether to actually update self.d
            if False, return the updated d mtx
        verbose_opt : bool
            whether to print out detailed messages

        Returns
        -------
            if update_self_d=False, return the updated d mtx
        """
        # Use last beam if not specified
        if k_axis is None:
            k_axis = hex_ops.cube_to_k(self.aim_last_cube, self.canvas_axial)
        if k_in_beam is None:
            k_in_beam = hex_ops.cube_to_k(self.beam_cover_last_cube, self.canvas_axial)

        # if beam radius not specified
        if beam_radius is None:
            beam_radius = self.radius_last

        # Get corresponding array indices
        axis_array_idx = np.argwhere(k_axis == self.k_canvas).squeeze()
        cover_array_idx = np.array([np.argwhere(x == self.k_canvas) for x in k_in_beam]).squeeze()

        if np.sum(self.d[cover_array_idx]) == 1:  # if dk in all cells covered by beam sum to 1
            # TODO: review this behavior, since if have within-beam-location-dependent pmpfa, update may still be needed
            if verbose_opt:
                print('Probability of all cells in beam sums to 1. No updates performed.')
            dk_new = self.d
        else:
            dk_new = self.d * self.table_X0[beam_radius][axis_array_idx]  # update dk

            # Normalize and reshape to get dk map
            # The if statement is to avoid the case when all cells outside the beam are 0
            # This case is problematic in the updating only when PM is 0, since in all other
            # cases it is possible that no echo is returned when the target is in the beam
            if np.sum(dk_new) != 0:
                dk_new = dk_new / np.sum(dk_new)

        # Return value if update_self_d=False
        # This is necessary, because in get_h_est we need to "pretend" to update map
        # to get dk map after receiving the next echo outcome
        if update_self_d:
            self.d = dk_new
        else:
            return dk_new

    def get_h_actual(self):
        """Calculate the actual h after the last update.
        """
        ha = -np.sum(xlogy(self.d, self.d))  # entropy after update
        self.h_actual = loge_to_log2(ha)    # convert to log base 2

    def get_beam_coverage(self):
        """Calculate coverage of the last beam in cube coordinate.
        """
        self.beam_cover_last_cube = beam_shape_hex(cube_ctr=self.aim_last_cube,
                                                   canvas_axial=self.canvas_axial,
                                                   beam_radius=self.radius_last)

    def _get_p_X0(self, k_idx, k_in_beam, beam_r):
        """Calculate p_X0 given the beam axis and all grids in beam.

        Parameters
        ----------
        k_idx
            beam axis
        k_in_beam
            all other grids in beam
        """
        if k_idx in k_in_beam:
            # if idx in beam: miss at idx and no false alarm at all other grids
            k_others = np.setdiff1d(k_in_beam, k_idx, assume_unique=True)
            return (
                # self.pmpfa_fcn("pm", np.array(k_idx), "animal", beam_r)
                # * np.prod(1 - self.pmpfa_fcn("pfa", k_others, "animal", beam_r))                
                self.pmpfa_fcn(fcn_type="pm", k_axis=k_idx, k_cell=k_idx, simu_type="animal", beam_radius=beam_r)
                * np.prod(1 - self.pmpfa_fcn(fcn_type="pfa", k_axis=k_idx, k_cell=k_others, simu_type="animal", beam_radius=beam_r))
            )
        else:
            # return np.prod(1 - self.pmpfa_fcn("pfa", k_in_beam, "animal", beam_r))            
            return np.prod(1 - self.pmpfa_fcn(fcn_type="pfa", k_axis=k_idx, k_cell=k_in_beam, simu_type="animal", beam_radius=beam_r))

    def get_est_ph(self):
        """
        Loop through all possible axis location to calculate
        - the expected entropy (infotaxis)
        - the expected probability of correct with MAP after next ping (MAP_future)
        """
        dk_curr = np.copy(self.d)  # current dk map

        # Get grids to be searched given spatial constraint
        # h_est is only calculated for these grids and not the entire canvas
        if self.search_dict is not None:
            # Get sequence index for grids to be searched
            k_search = self._get_grid_to_search()
        else:
            k_search = self.k_canvas

        # Get arary_idx_* used to index dk map
        array_idx_search = np.array([np.argwhere(x == self.k_canvas) for x in k_search]).squeeze()

        p_correct = defaultdict(lambda: np.ones(dk_curr.shape)*np.nan)    # expected prob correct
        h_next = defaultdict(lambda: np.ones(dk_curr.shape)*np.nan)    # estimated entropy
        h_X0_est = defaultdict(lambda: np.ones(dk_curr.shape)*np.nan)  # estimated entropy if next X=0
        h_X1_est = defaultdict(lambda: np.ones(dk_curr.shape)*np.nan)  # estimated entropy if next X=1
        p_X0_est = defaultdict(lambda: np.ones(dk_curr.shape)*np.nan)  # estimated next P(X=0)
        p_X1_est = defaultdict(lambda: np.ones(dk_curr.shape)*np.nan)  # estimated next P(X=1)

        # loop through all possible beam radius
        for beam_r in self.beam_radius:

            # loop through all possible beam axis
            # array_idx is used to index dk
            for array_idx, k_axis in zip(array_idx_search, k_search):

                # New beam cover
                cover_cube = beam_shape_hex(  # beam coverage in cube coordinate
                    hex_ops.k_to_cube(k_axis, self.canvas_axial),
                    canvas_axial=self.canvas_axial,
                    beam_radius=beam_r)

                # Get flattened indices
                k_in_beam = hex_ops.cube_to_k(cover_cube, self.canvas_axial)  # cells covered in beam

                # array_idx_* are the indices of k in k_canvas
                # since k is NOT a simple sequence of 0, 1, 2, ...
                # but is the unraveled index given a particular (nx,ny) canvas size
                array_idx_in_beam = np.array([np.argwhere(x == self.k_canvas) for x in k_in_beam]).squeeze()

                # Estimate d and h
                d_est_X1 = self.update_X1(beam_radius=beam_r,
                                          k_axis=k_axis, k_in_beam=k_in_beam, update_self_d=False)
                d_est_X0 = self.update_X0(beam_radius=beam_r,
                                          k_axis=k_axis, k_in_beam=k_in_beam, update_self_d=False)
                
                ######## EXPECTED ENTROPY ########

                h_X1 = -np.sum(xlogy(d_est_X1, d_est_X1))  # avoid 0*log0 error
                h_X0 = -np.sum(xlogy(d_est_X0, d_est_X0))
                h_X1 = loge_to_log2(h_X1)  # convert to base 2
                h_X0 = loge_to_log2(h_X0)

                # Probability of receiving an echo or not -----------
                # Use array_idx_* to index dk map
                p_tB = np.sum(dk_curr[array_idx_in_beam])  # prob of target in beam

                # Probability of receiving an echo: p_X1
                # --------------------------
                # p(X=1 | target in B)
                p_X1_tB_tmp = np.zeros((k_in_beam.size, 1))
                # target in beam: loop through all possible k location
                for seq, kk in enumerate(k_in_beam):
                    k_in_beam_not_kk = np.setdiff1d(k_in_beam, kk, assume_unique=True)
                    if dk_curr[array_idx_in_beam[seq]] == 0:
                        p_X1_tB_tmp[seq] = 0
                    else:
                        p_X1_tB_tmp[seq] = (
                            1 -
                            self.pmpfa_fcn(fcn_type='pm', k_axis=k_axis, k_cell=kk,
                                            simu_type='animal', beam_radius=beam_r) *
                            np.prod(1 - self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_in_beam_not_kk,
                                                        simu_type='animal', beam_radius=beam_r))
                        ) * \
                        (dk_curr[array_idx_in_beam[seq]] / np.sum(dk_curr[array_idx_in_beam]))  # prob of target in k given k in beam
                p_X1_tB = np.sum(p_X1_tB_tmp)
                # --------------------------
                # p(X=1 | target not in B)
                # does not need to consider where exactly k is in the beam,
                # so no scaling related to dk like in the p_X1_tB_tmp loop
                p_X1_tnB = 1 - np.prod(1 - self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_in_beam,
                                                          simu_type='animal', beam_radius=beam_r))
                #--------------------------
                # Put everything together
                p_X1 = p_X1_tB * p_tB + p_X1_tnB * (1-p_tB)
                # ---------------------------------------------------

                # expected max posterior and entropy
                p_correct[beam_r][array_idx] = p_X1 * d_est_X1.max() + (1-p_X1) * d_est_X0.max()
                h_next[beam_r][array_idx] = p_X1 * h_X1 + (1 - p_X1) * h_X0
                h_X0_est[beam_r][array_idx] = h_X0
                h_X1_est[beam_r][array_idx] = h_X1
                p_X0_est[beam_r][array_idx] = 1 - p_X1
                p_X1_est[beam_r][array_idx] = p_X1

                ######## EXPECTED PROB CORRECT ########

                # Only process grids that are max of d_est_X0 and d_est_X1
                # Grids at max of d_est_X0
                array_idx_max_X0 = np.argwhere(d_est_X0 == d_est_X0.max()).squeeze()
                tmp_X0 = []
                if array_idx_max_X0.size == 1:
                    k_idx = self.k_canvas[array_idx_max_X0]  # a_idx is used to index dk map
                    tmp_X0.append(
                        (dk_curr[array_idx_max_X0] * (1-self._get_p_X0(k_idx, k_in_beam, beam_r))).squeeze()
                    )
                else:
                    for a_idx in array_idx_max_X0:
                        k_idx = self.k_canvas[a_idx]  # a_idx is used to index dk map
                        tmp_X0.append(
                            (dk_curr[a_idx] / len(array_idx_max_X0) * self._get_p_X0(k_idx, k_in_beam, beam_r)).squeeze()
                        )

                # Grids at max of d_est_X1
                array_idx_max_X1 = np.argwhere(d_est_X1 == d_est_X1.max()).squeeze()
                tmp_X1 = []
                if array_idx_max_X1.size == 1:
                    k_idx = self.k_canvas[array_idx_max_X1]  # a_idx is used to index dk map
                    tmp_X1.append(
                        (dk_curr[array_idx_max_X1] * (1-self._get_p_X0(k_idx, k_in_beam, beam_r))).squeeze()
                    )
                else:
                    for a_idx in array_idx_max_X1:
                        k_idx = self.k_canvas[a_idx]  # a_idx is used to index dk map
                        tmp_X1.append(
                            (dk_curr[a_idx] / len(array_idx_max_X1) * (1-self._get_p_X0(k_idx, k_in_beam, beam_r))).squeeze()
                        )

                p_correct[beam_r][array_idx] = np.sum(tmp_X0 + tmp_X1)

                ########################################

        self.p_est = p_correct
        self.h_est = h_next
        self.h_X0_est = h_X0_est
        self.h_X1_est = h_X1_est
        self.p_X0_est = p_X0_est
        self.p_X1_est = p_X1_est

    def _get_grid_to_search(self) -> np.array:
        """
        Get k for the grids to be searched given the last beam aim and beam radius.
        """
        search_r = self.search_dict['neighbor_radius']
        if self.aim_last_cube is None:  # if no last beam aim (e.g. first ping)
            return self.k_canvas  # serach the entire canvas
        else:
            neighbor_cube = beam_shape_hex(self.aim_last_cube, self.canvas_axial, beam_radius=search_r)
            return hex_ops.cube_to_k(neighbor_cube, self.canvas_axial)
        # # TODO: add directional beam movement constraints
        # search_dir = search_opt['movement_direction']
        # direction_k = ...

    def _search_est_map(self, vv: np.ndarray, array_idx_search: List, verbose_opt=False):
        """
        Select grid in map with estimated quantities.

        infotaxis: select grid with the smallest expected h after next ping (self.h_est)
        MAP_future: select grid with the largest expected posterior after next ping (self.p_est)

        Parameters
        ----------
        vv : np.ndarray
            an array containing h_est or p_est to be selected from
        array_idx_search : list
            a list containing all array indices to consider (axial coordinate)
        verbose_opt : bool
            if to print stuff for debugging, default = False: do not print

        Returns
        -------
        pick_idx
            selected index out of the input array_idx_search
        argmin_idx
            all possible sequence indices with identical or very similar h_est
            to select the next beam from
        """

        if np.sum(vv - vv.min()) == 0:  # all cells being considered have the same value
            print('All values within search constraints are the same, randomly selecting one cell.')
            pick_idx = np.random.randint(0, array_idx_search.size)  # randomly select one argmin point
            argmin_idx = array_idx_search

        else:
            vv_norm = (vv - vv.min()) / (vv.max() - vv.min())
            vv_norm_sort = np.sort(vv_norm)
            vv_sort = np.sort(vv)

            # argmin_seq_idx below is sequence index in vv
            if np.diff(np.unique(vv_norm_sort))[0] <= 0.01 and \
                    np.diff(np.unique(vv_sort))[0] <= 1e-12:
                print('Challenging case: many similar minimum values, need careful selection.')

                # Indices that fit both conditions
                argmin_seq_idx_1 = np.where(
                    np.isclose(vv_norm, np.min(vv_norm), rtol=0, atol=0.01)
                )[0]
                argmin_seq_idx_2 = np.where(vv - np.min(vv) <= 1e-12)[0]
                argmin_seq_idx = np.intersect1d(argmin_seq_idx_1, argmin_seq_idx_2, assume_unique=True)
                if verbose_opt:
                    print('Cells <=0.01 region from min(value) (normalized values)')
                    print(argmin_seq_idx_1)
                    print('Cells <= 10e-12 region from min(value) (raw values)')
                    print(argmin_seq_idx_2)
                    print('Joint of the above')
                    print(argmin_seq_idx)
            else:
                print('Simple case: there is a single minimum value.')
                argmin_seq_idx = np.where(vv == np.min(vv))[0]  # find all argmin locs

            if verbose_opt:
                print('Minimum values (displaying at most 10 values):')
                [print('  %d  %.15f' % (x_seq, x)) for x_seq, x in enumerate(vv[argmin_seq_idx[:10]])]
                print('Sorted first 10 values:')
                [print('  %d  %.15f' % (x_seq, x)) for x_seq, x in enumerate(np.sort(vv)[:10])]

            # Default printout
            print('Number of minimum choice: %02d' % argmin_seq_idx.size)
            print('Possible next beam sequence indices (displaying at most 10 values):')
            print(argmin_seq_idx[:10])

            # randomly select one argmin point
            pick_idx = argmin_seq_idx[np.random.randint(0, argmin_seq_idx.size)]
            argmin_idx = array_idx_search[argmin_seq_idx]

        return pick_idx, argmin_idx


    def _search_MAP(self, dd: np.ndarray, array_idx_search: List,verbose_opt=False):
        """
        MAP: select grid with the maximum posterior probability in the current map (self.d)

        Parameters
        ----------
        dd : np.ndarray
            an array containing probability (dk) to be selected from
        array_idx_search : list
            a list containing all array indices to consider (axial coordinate)
        verbose_opt : bool
            if to print stuff for debugging, default = False: do not print

        Returns
        -------
        """
        # Use _search_est_map by inverting the sign of dd
        # to get max of dd (instead of min of vv under infotaxis)
        return self._search_est_map(
            vv=-dd,  # MAP needs the select grid with the max probability in dk
            array_idx_search=array_idx_search,
            verbose_opt=verbose_opt
        )        


    def get_next_beam(self, search_rule=None, verbose_opt=False, return_values=False):
        """Select next beam axis among those with lowest entropy.

        Parameters
        ----------
        search_rule : string
            search rule to use, options: "infotaxis", "MAP_future", "MAP"
            default to self.search_rule, which can be overwritten
            by setting this argument explicitly
        verbose_opt : bool
            if to print stuff for debugging, default = False: do not print
        return_values : bool
            whether to return selected beam aim and all equally good locations

        Returns
        -------
        pick_seq_idx
            sequence index of the selected next beam out of the canvas
        argmin_seq_idx
            all possible sequence indices to select the next beam from
        beam_r_sel
            the selected beam radius

        (Note the sequence index is different from the flattened axial indices k,
         it is the actual element index that can be used to select a row from canvas_cube.)
        """

        # Subset vv if there are spatial constraints in next beam location
        if self.search_dict is not None:
            # Get sequence index for grids to be searched
            k_search = self._get_grid_to_search()
            array_idx_search = np.array([np.argwhere(x == self.k_canvas) for x in k_search]).squeeze()
        else:
            array_idx_search = np.arange(self.k_canvas.size)

        # Set search_rule if not given
        if search_rule is None:
            search_rule = self.search_rule


        # Select next beam aim and beam radius based on search strategy

        # ------ infotaxis or MAP_future -------------------
        if search_rule in ["infotaxis", "MAP_future"]:
            # Get pick_seq_idx and argmin_seq_idx from each beam_r
            pick_idx_dict = dict()    # index to pick out from global vv
            argmin_idx_dict = dict()  # index to consider from global vv
            vv_dict = dict()      # minimum h_est or maximum p_est

            for beam_r in self.beam_radius:  # loop through all beam radius
                # Perform infotaxis or MAP_future search
                vv = (
                    self.h_est[beam_r][array_idx_search]  # need min
                    if search_rule == "infotaxis"
                    else -self.p_est[beam_r][array_idx_search]  # need max
                )
                pick_idx, argmax_idx = self._search_est_map(vv, array_idx_search, verbose_opt)

                # Save selection for this beam_r
                pick_idx_dict[beam_r] = array_idx_search[pick_idx]
                argmin_idx_dict[beam_r] = argmax_idx
                vv_dict[beam_r] = vv[pick_idx]

            # Select which beam radius and beam aim to use
            beam_r_sel = (
                min(vv_dict, key=vv_dict.get)  # the radius with min h_est
                if search_rule == "infotaxis"
                else max(vv_dict, key=vv_dict.get)  # the radius with max p_est
            )
            self.radius_last = beam_r_sel
            self.aim_last_cube = self.canvas_cube[pick_idx_dict[beam_r_sel], :]

            if return_values:
                return pick_idx_dict[beam_r_sel], argmin_idx_dict[beam_r_sel], beam_r_sel

        # ------ MAP --------------------------------------
        elif search_rule == "MAP":
            pick_idx, argmax_idx = self._search_MAP(
                dd=self.d[array_idx_search],
                array_idx_search=array_idx_search,
                verbose_opt=verbose_opt
            )
            # Update beam aim, but not beam radius
            self.aim_last_cube = self.canvas_cube[array_idx_search[pick_idx], :]

            if return_values:
                return array_idx_search[pick_idx], argmax_idx, None

        # -------------------------------------------------
        else:
            raise NotImplementedError(f"Search strategy {search_rule} has not been implemented!")
        
        # Update beam coverage
        self.get_beam_coverage()

    def get_echo(self):
        """Generate an echo outcome.
        """

        # Flattened index
        k_axis = hex_ops.cube_to_k(self.aim_last_cube, self.canvas_axial)
        k_in_beam = hex_ops.cube_to_k(self.beam_cover_last_cube, self.canvas_axial)
        k_target = hex_ops.cube_to_k(self.target_cube, self.canvas_axial)
        k_others = np.setdiff1d(k_in_beam, k_target, assume_unique=True)  # cells in beam that does not contain target

        # Find out if an echo was received
        if np.isin(k_target, k_in_beam):   # target covered in beam
            echo_k_others = np.random.uniform(size=k_others.size) < \
                           self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_others, simu_type='echo', beam_radius=self.radius_last)
            echo_k_target = np.random.uniform(size=1) > \
                           self.pmpfa_fcn(fcn_type='pm', k_axis=k_axis, k_cell=k_target, simu_type='echo', beam_radius=self.radius_last)
            echo_val = bool(np.any(echo_k_others) or np.any(echo_k_target))
            if echo_val:
                print('ECHO!')
                echo_type = True
            else:
                print('MISSING!')
                echo_type = False
        else:   # target not covered in beam
            echo_k_others = np.random.uniform(size=k_in_beam.size) < \
                           self.pmpfa_fcn(fcn_type='pfa', k_axis=k_axis, k_cell=k_others, simu_type='echo', beam_radius=self.radius_last)
            echo_val = bool(np.any(echo_k_others))
            if echo_val:
                print('FALSE ALARM!')
                echo_type = False
            else:
                print('NO ECHO!')
                echo_type = True

        self.echo_value = echo_val
        self.echo_type = echo_type

    def check_stopping(self, criteria: Dict =None):
        """Checking if the stopping condition is met.

        Parameters
        ----------
        criteria : Dict
            A dictionary containing stopping criteria.
            The search stops when one of the criteria is met.

            - max_ping_num: maximum number of pings (steps)
            - prob_th: threshold for target probability to stop the search

        Returns
        -------
        bool
            Whether or not the stopping condition has been met
        """
        # Set default max_ping_num if it does not exist as a criterion
        if criteria is None:
            criteria = {}

        if "max_ping_num" not in criteria:
            criteria["max_ping_num"] = 100

        # Check max probability in map
        if "p_max_th" in criteria:
            if np.any(self.d > criteria["pmax_th"]):
                print("Search stopped: p_max exceeding threshold!")
                return True

        # Check if repeate same probability N times
        if "pmax_repeat_N" in criteria:
            p_max = np.max(np.vstack(self.d_all), axis=1)  # max target probability for all pings
            p_max_diff = np.diff(p_max[-criteria["pmax_repeat_N"]:])  # diff of the last N p_max
            if len(p_max_diff) > 1 and np.all(abs(p_max_diff) < criteria["pmax_diff_threshold"]):  # p_max values are similar
                print(f"Search stopped: the last {criteria["pmax_repeat_N"]} p_max are similar!")
                return True

        # Check if already reached max number of pings
        if len(self.d_all) >= criteria["max_ping_num"]:
            print("Search stopped: Reached the maximum allowable number of pings!")
            return True

        return False

    def save_to_nc(self, save_fname, time_spent_perf=None, time_spent_proc=None):
        """ Save simulation results into a .nc file.

        Parameters
        ----------
        save_fname : str
            path/filename to save results to
        time_spent_perf : float or None
            time spent to run the search measured by time.perf_counter()
        time_spent_proc : float or None
            time spent to run the search measured by time.process_time()
        """
        canvas = ('canvas', np.arange(len(self.d)))      # dimension of canvas
        steps = ('steps', np.arange(len(self.d_all)))    # number of simulation steps
        beam_radius = ('beam_radius', self.beam_radius)  # beam radius
        cube = ('cube', ['x', 'y', 'z'])                 # for cube coordinate
        simu_type = ('simu_type', ['animal', 'echo'])    # simulation parameter type

        # creating DataArrays
        da_canvas = xr.DataArray(self.canvas_cube, coords=[canvas, cube])
        da_d_all = xr.DataArray(np.array(self.d_all), coords=[steps, canvas])
        da_beam_cube_all = xr.DataArray(self.beam_cube_all, coords=[steps, cube])
        da_radius_all = xr.DataArray(self.radius_all, coords=[steps])
        da_p_est_all = xr.DataArray(np.array(self.p_est_all), coords=[steps, beam_radius, canvas])
        da_h_est_all = xr.DataArray(np.array(self.h_est_all), coords=[steps, beam_radius, canvas])
        da_h_actual_all = xr.DataArray(np.array(self.h_actual_all), coords=[steps])
        da_echo_value_all = xr.DataArray(np.array(self.echo_value_all), coords=[steps])
        da_echo_type_all = xr.DataArray(np.array(self.echo_type_all), coords=[steps])
        da_pm_const = xr.DataArray(np.array([[x['pm_const'] for x in list(self.param_animal.values())],
                                             [x['pm_const'] for x in list(self.param_echo.values())]]),
                                   coords=[simu_type, beam_radius])
        da_pfa_const = xr.DataArray(np.array([[x['pfa_const'] for x in list(self.param_animal.values())],
                                              [x['pfa_const'] for x in list(self.param_echo.values())]]),
                                    coords=[simu_type, beam_radius])

        # construct Dataset
        ds = xr.Dataset(
            {
                'canvas_cube': da_canvas,
                'd_all': da_d_all,
                'beam_cube_all': da_beam_cube_all,
                'radius_all': da_radius_all,
                'p_est_all': da_p_est_all,
                'h_actual_all': da_h_actual_all,
                'h_est_all': da_h_est_all,
                'echo_value_all': da_echo_value_all,
                'echo_type_all': da_echo_type_all,
                'pm_const': da_pm_const,
                'pfa_const': da_pfa_const,
            }
        )

        # add attributes
        ds.attrs['target_loc'] = self.target_cube
        if hasattr(self, 'peg_loc'):  # if peg_cube exists  TODO: this is not currently included
            ds.attrs['peg_loc'] = self.peg_cube

        ds.attrs['canvas_radius'] = self.canvas_radius

        if self.search_dict is not None:
            for k, v in self.search_dict.items():
                ds.attrs['search_' + k] = v

        if time_spent_perf is not None:
            ds.attrs['search_time_perf'] = time_spent_perf
        if time_spent_proc is not None:
            ds.attrs['search_time_proc'] = time_spent_proc

        # convert to .nc
        ds.to_netcdf(save_fname)

    def plot_latest_update(self, search_rule=None, plot_attrs=PLOT_ATTRS, orientation='horizontal'):
        """Plot maps of target probability, entropy, and entropy variation sequence.

        Parameters
        ----------
        search_rule : string
            search rule to use, options: "infotaxis", "MAP_future", "MAP"
            default to self.search_rule, which can be overwritten
            by setting this argument explicitly
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
        # Set search_rule if not given
        if search_rule is None:
            search_rule = self.search_rule

        # Get echo type
        echo_value = self.echo_value_all[-1]
        echo_type = self.echo_type_all[-1]
        echo_type_str = get_echo_type_str(echo_value=echo_value, echo_type=echo_type)

        pingnum = len(self.h_actual_all)

        # Construct env_attrs for plotting
        env_attrs = dict(target_loc=self.target_cube)

        # Get beam radius
        beam_r = self.radius_last

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
        plot_dk_map(ax,
            "P_T",
            canvas_cube=self.canvas_cube,
            canvas_vmap=self.d,
            beam_cube=self.beam_cube_all,
            echo_type_str=echo_type_str,
            env_attrs=env_attrs,
            plot_attrs=plot_attrs,
            cmap="pink"
        )

        # Infotaxis: expected entropy map 
        # MAP_future: expected posterior map after next ping
        # MAP: current posterior map
        ax = fig.add_subplot(gs_L[0, 1])
        if search_rule == "infotaxis":
            plot_dk_map(
                ax,
                "entropy",
                canvas_cube=self.canvas_cube,
                canvas_vmap=self.h_est[beam_r],
                beam_cube=self.beam_cube_all,
                echo_type_str=echo_type_str,
                env_attrs=env_attrs,
                plot_attrs=plot_attrs
            )
        elif search_rule == "MAP":
            plot_dk_map(ax,
                "prob target",
                canvas_cube=self.canvas_cube,
                canvas_vmap=self.d,
                beam_cube=self.beam_cube_all,
                echo_type_str=echo_type_str,
                env_attrs=env_attrs,
                plot_attrs=plot_attrs
            )
        elif search_rule == "MAP_future":
            plot_dk_map(
                ax,
                "prob correct",
                canvas_cube=self.canvas_cube,
                canvas_vmap=self.p_est[beam_r],
                beam_cube=self.beam_cube_all,
                echo_type_str=echo_type_str,
                env_attrs=env_attrs,
                plot_attrs=plot_attrs
            )

        # Next beam coverage
        hex_ops.plot_hexgrids(ax, cube_array=self.beam_cover_last_cube,  # plot next beam coverage
                            cube_color='y', gridalpha=0.4)

        # Entropy variation
        ax = fig.add_subplot(gs_R[0, 0])
        plot_entropy_seq(ax, self.h_actual_all, self.h_est_all, plot_attrs)
        ax.set_title(f'Ping #{pingnum:02d}     entropy variation', fontsize=14)

        plt.show()

        return fig
