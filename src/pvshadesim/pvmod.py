# -*- coding: utf-8 -*-
"""Generate the physical and electrical model of the module/ system."""

import ast
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from v_pvmismatch.pvmismatch import pvconstants, pvmodule, pvcell, pvsystem

from .plotting import plot_mod_idx, print_idx_map
from .utils import find_middle

# Main function


def create_pvmod_dict(pvmod_params, sim_config, pvcell_params,
                      cell_idx_xls, cell_pos_xls, gen_mod_idx=False,
                      NPTS=1500, Tcell=298.15):
    """
    Generate the simulation dictionary containing the physical and electrical models of the module for all sims.

    Parameters
    ----------
    pvmod_params : pandas.DataFrame
        Dataframe containing the PV module database.
    sim_config : pandas.DataFrame
        Dataframe containing the modules, cells, shade type, system info to simulate.
    pvcell_params : pandas.DataFrame
        Dataframe containing the PV cell database.
    cell_idx_xls : str
        Path to the "User_cell_index_maps.xlsx" file.
    cell_pos_xls : str
        Path to the "User_cell_pos.xlsx" file.
    gen_mod_idx : bool, optional
        Generate images of the cell position index within the module. The default is False.
    NPTS : int, optional
        Number of points in the IV curve. This is an input to PVMismatch. The default is 1500.
    Tcell : float, optional
        Cell temperature in kelvin (K). This is an input to PVMismatch. The default is 298.15.

    Returns
    -------
    mods_sys_dict : dict
        Dictionary containing the physical and electrical models of modules in the simulation.

    """
    sim_config_T = sim_config.T
    if 'GCR' not in sim_config_T:
        sim_config_T['GCR'] = 0.
    # Extract module names
    mod_list = sim_config_T['Module'].tolist()
    cell_list = sim_config_T['Cell'].tolist()
    is_ls_list = sim_config_T['is_Landscape'].tolist()
    shade_list = sim_config_T['Shade_Definition'].tolist()
    str_len_list = sim_config_T['Str_Length'].tolist()
    num_str_list = sim_config_T['Num_Str'].tolist()
    num_mods_shade_list = sim_config_T['Num_Mods_Shade'].tolist()
    is_AC_Mod_list = sim_config_T['is_AC_Mod'].tolist()
    plot_label_list = sim_config_T.index.tolist()
    MSX_list = sim_config_T['Mod_Space_X'].tolist()
    MSY_list = sim_config_T['Mod_Space_Y'].tolist()
    MEX_list = sim_config_T['Mod_Edge_X'].tolist()
    MEY_list = sim_config_T['Mod_Edge_Y'].tolist()
    Tilt_list = sim_config_T['Tilt'].tolist()
    Azimuth_list = sim_config_T['Azimuth'].tolist()
    str_tilt_list = sim_config_T['str_tilt'].tolist()
    numStrTrack_list = sim_config_T['num_str_tracker'].tolist()
    mkt_list = sim_config_T['Market'].tolist()
    gcr_list = sim_config_T['GCR'].tolist()

    mods_sys_dict = {}
    for idx_sim, mod_name in enumerate(mod_list):
        if isinstance(Tcell, list):
            Tc = Tcell[idx_sim]
        else:
            Tc = Tcell
        cell_name = cell_list[idx_sim]
        if cell_name == 'Default':
            cell_name = [pvmod_params[mod_name]['Cell_type']]
        elif '[' in cell_name:
            cell_name = ast.literal_eval(cell_name)
        else:
            cell_name = [cell_name]
        is_Landscape = is_ls_list[idx_sim]
        is_AC_Mod = is_AC_Mod_list[idx_sim]
        shade_sce = shade_list[idx_sim]
        if mod_name not in mods_sys_dict:
            mods_sys_dict[mod_name] = {}
        cname = cell_name[0]
        if cname not in mods_sys_dict[mod_name]:
            mods_sys_dict[mod_name][cname] = {}
        if is_Landscape:
            orientation = 'Landscape'
        else:
            orientation = 'Portrait'
        if orientation not in mods_sys_dict[mod_name][cname]:
            mods_sys_dict[mod_name][cname][orientation] = {}
        if is_AC_Mod:
            mod_type = 'AC'
        else:
            mod_type = 'DC'
        sim_info = [str_len_list[idx_sim], num_str_list[idx_sim],
                    num_mods_shade_list[idx_sim], is_AC_Mod_list[idx_sim],
                    plot_label_list[idx_sim], MSX_list[idx_sim],
                    MSY_list[idx_sim], MEX_list[idx_sim],
                    MEY_list[idx_sim], Tilt_list[idx_sim], Azimuth_list[idx_sim],
                    str_tilt_list[idx_sim], numStrTrack_list[idx_sim],
                    mkt_list[idx_sim], gcr_list[idx_sim]]
        mods_sys_dict[mod_name][cname][orientation][mod_type] = create_pv_mod(pvmod_params, pvcell_params,
                                                                              cell_idx_xls, cell_pos_xls,
                                                                              mod_name, cell_name,
                                                                              is_Landscape, shade_sce, sim_info,
                                                                              NPTS=NPTS, Tcell=Tc)
    if gen_mod_idx:
        plot_mod_idx(mods_sys_dict)
    return mods_sys_dict


def create_pv_mod(pvmod_params, pvcell_params, cell_idx_xls, cell_pos_xls, mod_name, cell_name, is_Landscape,
                  shade_sce, sim_info,
                  NPTS=1500, Tcell=298.15,
                  idx_map=None):
    """
    Generate the physical and electrical model of the PV module.

    Parameters
    ----------
    pvmod_params : pandas.DataFrame
        Dataframe containing the PV module database.
    pvcell_params : pandas.DataFrame
        Dataframe containing the PV cell database.
    cell_idx_xls : str
        Path to the "User_cell_index_maps.xlsx" file.
    cell_pos_xls : str
        Path to the "User_cell_pos.xlsx" file.
    mod_name : str
        Unique name of module in the database.
    cell_name : str
        Unique name of cell in the database.
    is_Landscape : bool
        Is the module orientation in Landscape?
    shade_sce : str
        Filter based on specific column in the shade database.
    sim_info : list
        List containing other simulation information about the system.
    NPTS : int, optional
        Number of points in the IV curve. This is an input to PVMismatch. The default is 1500.
    Tcell : float, optional
        Cell temperature in kelvin (K). This is an input to PVMismatch. The default is 298.15.
    idx_map : numpy.ndarray, optional
        User provided cell index map. This will be deprecated in a later version. Don't use. The default is None.

    Returns
    -------
    maxsys_dict : dict
        Dictionary containing the module model.

    """
    pvconst = pvconstants.PVconstants(npts=NPTS)

    # Sim info
    str_len, num_str, num_mods_shade, is_AC_Mod, plot_label, Mod_Space_X, Mod_Space_Y, Mod_Edge_X, Mod_Edge_Y, Tilt, Azimuth, str_tilt, num_str_tracker, mkt, gcr = sim_info
    num_mods_shade = ast.literal_eval(num_mods_shade)

    # Convert some of the variable to required type
    str_len = int(str_len)
    num_str = int(num_str)
    Mod_Space_X = float(Mod_Space_X)
    Mod_Space_Y = float(Mod_Space_Y)
    Mod_Edge_X = float(Mod_Edge_X)
    Mod_Edge_Y = float(Mod_Edge_Y)
    gcr = float(gcr)
    num_str_tracker = int(num_str_tracker)

    # Extract module information
    mod_params = pvmod_params[mod_name]
    # User cell idx and cell_poss
    if 'user_cell_idx' in mod_params:
        user_sht_name = mod_params['user_cell_idx']
        if user_sht_name not in ['True', 'False', 'TRUE', 'FALSE']:
            user_cell_idx = pd.read_excel(
                cell_idx_xls, sheet_name=user_sht_name).to_numpy()
            user_cell_pos_df = pd.read_excel(
                cell_pos_xls, sheet_name=user_sht_name)
            user_cell_pos = user_cell_pos_df[user_cell_pos_df.columns.to_list()[0]].to_list()
            use_user = True
            if 'series' in user_cell_pos_df.columns.to_list()[0]:
                user_series = True
            else:
                user_series = False
        else:
            use_user = False
    else:
        use_user = False
    # Extract cell name

    # Extract Module info
    num_cells_x = int(mod_params['Num_cells_X'])
    num_cells_y = int(mod_params['Num_cells_Y'])
    num_parallel = int(mod_params['Num_parallel'])
    layout_type = mod_params['parallel_type']
    is_Series = mod_params['is_Series']
    Cell_rotated = mod_params['Cell_rotated']
    VBYPASS = mod_params['VBYPASS']
    num_diodes = int(mod_params['Num_diodes'])
    outer_circuit = mod_params['outer_circuit']

    # Check if diode voltage is single value or list
    if isinstance(VBYPASS, str) and '[' in VBYPASS:
        VBYPASS_ls = ast.literal_eval(VBYPASS)
    else:
        VBYPASS_ls = [np.float64(VBYPASS)]*num_diodes

    # Extract cell info
    CELLAREA = []
    cell_X = []
    cell_Y = []
    for cn in cell_name:
        CELLAREA.append(pvcell_params.loc['cellArea', cn])
        if Cell_rotated:
            cell_X.append(pvcell_params[cn]['cell_width'])
            cell_Y.append(pvcell_params[cn]['cell_length'])
        else:
            cell_X.append(pvcell_params[cn]['cell_length'])
            cell_Y.append(pvcell_params[cn]['cell_width'])

    # PHYSICAL INFO
    # Extract Module info
    # Module Map
    if idx_map is None:
        if use_user:
            idx_map = user_cell_idx
        else:
            idx_map = create_idx_map(num_cells_x, num_cells_y, num_parallel,
                                     layout_type, is_Series, Cell_rotated)
    # System Map
    sys_idx_map = create_sys_idx_map(str_len, num_str)
    sys_idx_map_formatted = print_idx_map(sys_idx_map)

    # Create Module Coordinates Array
    # Module vertex coordinates
    Mod_X, Mod_Y = mod_params['Mod_X'], mod_params['Mod_Y']
    if gcr > 0:
        if is_Landscape:
            Mod_Space_X = (Mod_X/gcr) - Mod_X*np.cos(np.deg2rad(Tilt))
        else:
            Mod_Space_Y = (Mod_Y/gcr) - Mod_Y*np.cos(np.deg2rad(Tilt))
    mod_coord_array = create_cell_coordinates(sys_idx_map, [Mod_X], [Mod_Y],
                                              Mod_Space_X, Mod_Space_Y,
                                              Mod_Edge_X, Mod_Edge_Y, Tilt, str_tilt,
                                              is_Landscape=is_Landscape, is_Mod=True)
    # Module polygons
    mod_poly_df = cell_ploygons(mod_coord_array)

    # Create Cell polygons for entire system
    CellSpace_X = mod_params['CellSpace_X']
    CellSpace_Y = mod_params['CellSpace_Y']
    EdgeSpace_X = mod_params['EdgeSpace_X']
    EdgeSpace_Y = mod_params['EdgeSpace_Y']
    # Create Cell coordinates array
    syscell_poly_dict = {}
    for idx_row in range(mod_poly_df.shape[0]):
        syscell_poly_dict[idx_row] = {}
        for idx_col in range(mod_poly_df.shape[1]):
            mod_poly = mod_poly_df.iloc[idx_row, idx_col]
            mx, my = mod_poly.exterior.xy
            Mod_orig_X = mx[0]
            Mod_orig_Y = my[0]
            cell_coord_array = create_cell_coordinates(idx_map, cell_X, cell_Y,
                                                       CellSpace_X, CellSpace_Y,
                                                       EdgeSpace_X, EdgeSpace_Y,
                                                       Tilt, str_tilt,
                                                       Mod_orig_X, Mod_orig_Y,
                                                       is_Landscape=is_Landscape)
            if (idx_row == mod_poly_df.shape[0] - 1) and (idx_col == 0):
                cell_coord_arr = cell_coord_array
            cell_poly_df = cell_ploygons(cell_coord_array)
            syscell_poly_dict[idx_row][idx_col] = cell_poly_df

    # ELECTRICAL CIRCUIT INFO
    # Extract Module info
    tct_flag = mod_params['TCT']
    add_crstie = mod_params['more_crossties']
    # cellpos calculation
    # number columns or cells per diode (depends on series or parallel)
    numcells_per_diode = calc_numcols_diode(num_cells_x, num_cells_y,
                                            num_diodes, is_Series,
                                            num_parallel, layout_type,
                                            Cell_rotated)
    if Cell_rotated:
        num_cells_x, num_cells_y = num_cells_y, num_cells_x

    if use_user:
        cell_pos = pvmodule.crosstied_cellpos_pat(user_cell_pos,
                                                  num_parallel,
                                                  partial=not tct_flag)
        if user_series:
            cell_pos = set_crosstie(cell_pos, value_crosstie=False)
        else:
            add_crstie = ast.literal_eval(add_crstie)
            if add_crstie is not None:
                # Add additional crossties if specified

                for i in add_crstie:
                    cell_pos = set_idx(cell_pos, i, value_crosstie=True)
    else:
        if is_Series:
            # If series use standard cell position function
            cell_pos = pvmodule.standard_cellpos_pat(num_cells_y,
                                                     numcells_per_diode)
        else:
            # If parallel use crosstied cell position function
            cell_pos = pvmodule.crosstied_cellpos_pat(numcells_per_diode,
                                                      num_parallel,
                                                      partial=not tct_flag)
            add_crstie = ast.literal_eval(add_crstie)
            if add_crstie is not None:
                # Add additional crossties if specified

                for i in add_crstie:
                    cell_pos = set_idx(cell_pos, i, value_crosstie=True)
    # Create PVMM Cell class
    cell_types = list(range(len(cell_name)))
    if len(cell_types) < idx_map.shape[1]:
        cell_types = np.array(cell_types * idx_map.shape[1])
    cell_types = np.tile(cell_types, (idx_map.shape[0], 1))
    ordered_cell_types = cell_types.flatten()[np.argsort(idx_map, axis=None)]
    pvcs = []
    for cn in cell_name:
        pvc = create_pvcell(pvcell_params, cn, pvconst)
        pvc.Ee = 1.0
        pvc.Tcell = Tcell
        pvcs.append(pvc)
    pvc_list = []
    for ctype in ordered_cell_types.tolist():
        pvc_list.append(pvcs[ctype])
    # Create PVMM Module class
    maxmodule = pvmodule.PVmodule(pvconst=pvconst,
                                  cell_pos=cell_pos,
                                  pvcells=pvc_list,
                                  Vbypass=VBYPASS_ls,
                                  cellArea=CELLAREA[0])
    # Build PV system
    maxsys = pvsystem.PVsystem(pvconst=pvconstants.PVconstants(npts=NPTS),
                               pvmods=maxmodule, numberStrs=num_str, numberMods=str_len)
    unique_cell_types = cell_types.copy()
    if len(cell_name) < idx_map.shape[1]:
        cell_name = cell_name * idx_map.shape[1]
    u_cell_names = list(set(cell_name))
    for idx_ucn, ucn in enumerate(u_cell_names):
        indices = [i for i, x in enumerate(cell_name) if x == ucn]
        for idx_cn in indices:
            unique_cell_types[:, idx_cn] = idx_ucn
    # Set irradiance
    # irrad_suns = 1.0
    # maxsys.setSuns(irrad_suns)
    # Create Cross tie array
    idx_crosstie = np.ones(idx_map.shape, dtype=bool)
    for flist in cell_pos:
        for slist in flist:
            for tlist in slist:
                itemindex = np.where(idx_map == tlist['idx'])
                idx_crosstie[itemindex[0][0],
                             itemindex[1][0]] = tlist['crosstie']
    # Format the Module Map
    idx_map_formatted = print_idx_map(idx_map, idx_crosstie)

    if is_Landscape:
        idx_map = np.rot90(idx_map)
        unique_cell_types = np.rot90(unique_cell_types)
    # Store all information in a dictionary
    maxsys_dict = dict()
    maxsys_dict['Physical_Info'] = dict()
    maxsys_dict['Physical_Info']['Index_Map'] = idx_map
    maxsys_dict['Physical_Info']['Cell_type'] = unique_cell_types
    maxsys_dict['Physical_Info']['System Index_Map'] = sys_idx_map
    maxsys_dict['Physical_Info']['Crosstie_Map'] = idx_crosstie
    maxsys_dict['Physical_Info']['Formatted_Idx_Map'] = idx_map_formatted
    maxsys_dict['Physical_Info']['Formatted_Sys_Idx_Map'] = sys_idx_map_formatted
    maxsys_dict['Physical_Info']['Cell_Coordinates'] = cell_coord_arr
    maxsys_dict['Physical_Info']['Cell_Polygons'] = syscell_poly_dict
    maxsys_dict['Physical_Info']['Module_Coordinates'] = mod_coord_array
    maxsys_dict['Physical_Info']['Module_Polygon'] = mod_poly_df
    maxsys_dict['Physical_Info']['is_Landscape'] = is_Landscape
    maxsys_dict['Physical_Info']['Mod_Space_X'] = Mod_Space_X
    maxsys_dict['Physical_Info']['Mod_Space_Y'] = Mod_Space_Y
    maxsys_dict['Physical_Info']['Glass_thickness'] = mod_params['Glass_thick']
    maxsys_dict['Physical_Info']['Edge_to_glass'] = mod_params['Frame_2_glass']
    maxsys_dict['Physical_Info']['Lip_thickness'] = mod_params['Lip_thick']
    maxsys_dict['Physical_Info']['Lip_length'] = mod_params['Lip_length']
    maxsys_dict['Electrical_Circuit'] = dict()
    maxsys_dict['Electrical_Circuit']['Cell_Postion'] = cell_pos
    maxsys_dict['Electrical_Circuit']['PV_Module'] = maxmodule
    maxsys_dict['Electrical_Circuit']['PV_System'] = maxsys
    maxsys_dict['Electrical_Circuit']['outer_circuit'] = outer_circuit
    maxsys_dict['shade_list'] = shade_sce
    maxsys_dict['Sim_info'] = dict()
    maxsys_dict['Sim_info']['str_len'] = str_len
    maxsys_dict['Sim_info']['num_str'] = num_str
    maxsys_dict['Sim_info']['num_mods_shade'] = num_mods_shade
    maxsys_dict['Sim_info']['is_AC_Mod'] = is_AC_Mod
    maxsys_dict['Sim_info']['plot_label'] = plot_label
    maxsys_dict['Sim_info']['tilt'] = Tilt
    maxsys_dict['Sim_info']['azimuth'] = Azimuth
    maxsys_dict['Sim_info']['str_tilt'] = str_tilt
    maxsys_dict['Sim_info']['num_str_tracker'] = num_str_tracker
    maxsys_dict['Sim_info']['Market'] = mkt

    return maxsys_dict


def create_idx_map(num_cells_x, num_cells_y, num_parallel=2,
                   layout_type='snake_together', is_Series=True,
                   Cell_rotated=False):
    """
    Generate the cell index map of a module given the number of cells in 2D for some standard modules.

    Options for layout_type are "snake_together", "LR_half", "TB_half", and "all_parallel".

    Parameters
    ----------
    num_cells_x : int
        Number of columns in a module.
    num_cells_y : int
        Number of rows in a module.
    num_parallel : int, optional
        Number of parallel substrings in a module. The default is 2.
    layout_type : str, optional
        Layout type for a module with parallel substrings. The default is 'snake_together'.
    is_Series : bool, optional
        Is it a series only module? The default is True.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Returns
    -------
    idx_map : numpy.ndarray
        Cell index map of the module.

    """
    if is_Series:
        idx_map = create_series_map(num_cells_x, num_cells_y, Cell_rotated)
    else:
        idx_map = create_parallel_map(num_cells_x, num_cells_y, num_parallel,
                                      layout_type, Cell_rotated)
    return idx_map


def set_crosstie(cell_pos_in, value_crosstie=False):
    """
    # Find cell with idx, then set crosstie to input True/False value.

    # Return cell_pos_out
    #
    # cell_pos is a list of lists of lists, so this function first finds the cell
    # item with the indicated idx

    Parameters
    ----------
    cell_pos_in : List
        List of lists of lists containing the electrical circuit architecture for
        a PVMM PV module with indices and cross tie information.
    idx : int
        Index for which Crosstie needs to be set to value_crosstie in the cell_pos.
    value_crosstie : Boolean, optional
        Flag indicating whether to set the crossite of idx to True or False.
        The default is True.

    Returns
    -------
    cell_pos_out : List
        List of lists of lists containing the electrical circuit architecture for
        a PVMM PV module with indices and cross tie information. The crossties
        are updated based on input.

    """
    cell_pos_out = cell_pos_in.copy()

    for i, substr in enumerate(cell_pos_in):  # each substring
        # each column of cells in substring
        for j, cell_col in enumerate(substr):
            for k, cell in enumerate(cell_col):  # each cell
                cell_pos_out[i][j][k]['crosstie'] = value_crosstie

    return cell_pos_out


def create_series_map(num_cells_x, num_cells_y, Cell_rotated=False):
    """
    Create topological index map for a Series only module.

    The number of cells in the X & Y directions are specified.

    Parameters
    ----------
    num_cells_x : Int
        Number of cells in X direction or number of cell columns.
    num_cells_y : Int
        Number of cells in Y direction or number of cell rows.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Returns
    -------
    idx_map : numpy.ndarray
        Returns an array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    """
    # Create regular matrix with arange, reshape to required matrix size.
    if Cell_rotated:
        inp_map = np.arange(int(num_cells_y*num_cells_x),
                            dtype=int).reshape(int(num_cells_y), int(num_cells_x))
        idx_map = inp_map.copy()
        # Create snake pattern in matrix
        # For every second row, order of numbers is reversed.
        idx_map[1::2, :] = inp_map[1::2, ::-1]
    else:
        inp_map = np.arange(int(num_cells_x*num_cells_y),
                            dtype=int).reshape(int(num_cells_x), int(num_cells_y)).T
        idx_map = inp_map.copy()
        # Create snake pattern in matrix
        # For every second column, order of numbers is reversed.
        idx_map[:, 1::2] = inp_map[::-1, 1::2]
    return idx_map


def create_parallel_map(num_cells_x, num_cells_y, num_parallel=2,
                        layout_type='snake_together', Cell_rotated=False):
    """
    Create topological index map for a Parallel circuit module.

    The number of cells in the X & Y direction, number of parallel substrings, and the layout
    type are specified.

    Parameters
    ----------
    num_cells_x : Int
        Number of cells in X direction or number of cell columns.
    num_cells_y : Int
        Number of cells in Y direction or number of cell rows.
    num_parallel : Int, optional
        Number of parallel substrings. The default is 2.
    layout_type : string, optional
        Layout Type. The default is 'snake_together'. All options include:
            1) snake_together
            2) LR_half --> Parallel substring start in the middle and cut module along Y direction.
            3) TB_half --> Parallel substring start in the middle and cut module along X direction.
            4) all_parallel --> All cell columns are parallel substrings. Eg. P-series modules.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Raises
    ------
    ValueError
        1) Incorrect layout_type is inputted.
        2) Number of cell columns can't be divided by the number of parallel
            substrings to get a 0 remainder. This ensures that all indices are
            correctly filled out in module array.

    Returns
    -------
    idx_map : numpy.ndarray
        Returns an array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    """
    # Check if number of columns can be divided by number of parallel
    if layout_type == 'snake_together':
        # Call snake together function
        idx_map = create_snake_together(num_cells_x, num_cells_y,
                                        num_parallel, Cell_rotated)
    elif layout_type == 'LR_half':
        # Call LR Half function
        idx_map = create_LR_half(num_cells_x, num_cells_y,
                                 num_parallel, Cell_rotated)
    elif layout_type == 'TB_half':
        # Call TB Half function
        idx_map = create_TB_half(num_cells_x, num_cells_y,
                                 num_parallel, Cell_rotated)
    elif layout_type == 'all_parallel':
        # Call all parallel function
        idx_map = create_all_parallel(
            num_cells_x, num_cells_y, Cell_rotated)
    else:
        raise ValueError(
            'Incorrect layout type inputted. Possible values are: snake_together, LR_half, & TB_half')
    return idx_map


def create_snake_together(num_cells_x, num_cells_y, num_parallel=2,
                          Cell_rotated=False):
    """
    Create topological index map for a Snaked Together Parallel circuit module.

    The number of cells in the X & Y direction, and number of parallel substrings,
    are specified.

    Parameters
    ----------
    num_cells_x : Int
        Number of cells in X direction or number of cell columns.
    num_cells_y : Int
        Number of cells in Y direction or number of cell rows.
    num_parallel : Int, optional
        Number of parallel substrings. The default is 2.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Returns
    -------
    idx_map : numpy.ndarray
        Returns an array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    """
    # For the parallel case, it is similar to the series but need to split based
    # on number of parallel strings.
    if Cell_rotated:
        mat_list = []
        start_list = np.arange(0, int(num_cells_y*num_cells_x),
                               int(num_cells_y*num_cells_x/num_parallel))
        for idx_p in range(num_parallel):
            # Create regular matrix with arange, reshape to required matrix size.
            inp_map = np.arange(int(start_list[idx_p]),
                                int(num_cells_y*num_cells_x /
                                    num_parallel)+start_list[idx_p],
                                1, dtype=int)
            inp_map = inp_map.reshape(int(num_cells_y/num_parallel),
                                      int(num_cells_x))
            out_map = inp_map.copy()
            # Create snake pattern in matrix
            # For every second column, order of numbers is reversed.
            out_map[1::2, :] = inp_map[1::2, ::-1]
            mat_list.append(out_map)
        idx_map = np.zeros((int(num_cells_y), int(num_cells_x)), dtype=int)
        for idx_p in range(num_parallel):
            idx_map[idx_p::num_parallel, :] = mat_list[idx_p]
    else:
        mat_list = []
        start_list = np.arange(0, int(num_cells_x*num_cells_y),
                               int(num_cells_x*num_cells_y/num_parallel))
        for idx_p in range(num_parallel):
            # Create regular matrix with arange, reshape to required matrix size.
            inp_map = np.arange(int(start_list[idx_p]),
                                int(num_cells_x*num_cells_y /
                                    num_parallel)+start_list[idx_p],
                                1, dtype=int)
            inp_map = inp_map.reshape(int(num_cells_x/num_parallel),
                                      int(num_cells_y)).T
            out_map = inp_map.copy()
            # Create snake pattern in matrix
            # For every second column, order of numbers is reversed.
            out_map[:, 1::2] = inp_map[::-1, 1::2]
            mat_list.append(out_map)
        idx_map = np.zeros((int(num_cells_y), int(num_cells_x)), dtype=int)
        for idx_p in range(num_parallel):
            idx_map[:, idx_p::num_parallel] = mat_list[idx_p]
    return idx_map


def create_LR_half(num_cells_x, num_cells_y, num_parallel=2,
                   Cell_rotated=False):
    """
    Create topological index map for a Left-Right Half Parallel circuit module.

    The number of cells in the X & Y direction, and number of parallel substrings,
    are specified.

    Parameters
    ----------
    num_cells_x : Int
        Number of cells in X direction or number of cell columns.
    num_cells_y : Int
        Number of cells in Y direction or number of cell rows.
    num_parallel : Int, optional
        Number of parallel substrings. The default is 2.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Raises
    ------
    ValueError
        Raised if number of parallel substrings is not equal to 2.

    Returns
    -------
    idx_map : numpy.ndarray
        Returns an array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    """
    if num_parallel == 2:
        if Cell_rotated:
            mat_list = []
            start_list = np.arange(0, int(num_cells_y*num_cells_x),
                                   int(num_cells_y*num_cells_x/num_parallel))
            for idx_p in range(num_parallel):
                # Create regular matrix with arange, reshape to required matrix size.
                inp_map = np.arange(int(start_list[idx_p]),
                                    int(num_cells_y*num_cells_x /
                                        num_parallel)+start_list[idx_p],
                                    1, dtype=int)
                inp_map = inp_map.reshape(int(num_cells_y/num_parallel),
                                          int(num_cells_x))
                out_map = inp_map.copy()
                # Create snake pattern in matrix
                # For every second column, order of numbers is reversed.
                out_map[1::2, :] = inp_map[1::2, ::-1]
                mat_list.append(out_map)
            mat1 = mat_list[0]
            mat2 = mat_list[1].copy()
            out_mat1 = mat1.copy()
            out_mat1 = mat1[::-1, :]
            idx_map = np.concatenate((out_mat1, mat2), axis=0)
        else:
            mat_list = []
            start_list = np.arange(0, int(num_cells_x*num_cells_y),
                                   int(num_cells_x*num_cells_y/num_parallel))
            for idx_p in range(num_parallel):
                # Create regular matrix with arange, reshape to required matrix size.
                inp_map = np.arange(int(start_list[idx_p]),
                                    int(num_cells_x*num_cells_y /
                                        num_parallel)+start_list[idx_p],
                                    1, dtype=int)
                inp_map = inp_map.reshape(int(num_cells_x/num_parallel),
                                          int(num_cells_y)).T
                out_map = inp_map.copy()
                # Create snake pattern in matrix
                # For every second column, order of numbers is reversed.
                out_map[:, 1::2] = inp_map[::-1, 1::2]
                mat_list.append(out_map)
            mat1 = mat_list[0]
            mat2 = mat_list[1].copy()
            out_mat1 = mat1.copy()
            out_mat1 = mat1[:, ::-1]
            idx_map = np.concatenate((out_mat1, mat2), axis=1)

    else:
        raise ValueError(
            'Snaking Left & Right halves is not possible with this many parallel strings')
    return idx_map


def create_TB_half(num_cells_x, num_cells_y, num_parallel,
                   Cell_rotated=False):
    """
    Create topological index map for a Top-Bottom Half Parallel circuit module.

    The number of cells in the X & Y direction, and number of parallel substrings,
    are specified.

    Parameters
    ----------
    num_cells_x : Int
        Number of cells in X direction or number of cell columns.
    num_cells_y : Int
        Number of cells in Y direction or number of cell rows.
    num_parallel : Int, optional
        Number of parallel substrings. The default is 2.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Raises
    ------
    ValueError
        Raised if number of parallel substrings is not equal to 2.

    Returns
    -------
    idx_map : numpy.ndarray
        Returns an array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    """
    if num_parallel == 2:
        if Cell_rotated:
            inp_map1 = np.arange(int(num_cells_y*num_cells_x*0.5),
                                 dtype=int)
            inp_map1 = inp_map1.reshape(int(num_cells_y),
                                        int(num_cells_x*0.5))
            out_map1 = inp_map1.copy()
            # Create snake pattern in matrix
            # For every second column, order of numbers is reversed.
            out_map1[::2, :] = inp_map1[::2, ::-1]
            inp_map2 = np.arange(int(num_cells_y*num_cells_x*0.5),
                                 int(num_cells_y*num_cells_x),
                                 dtype=int)
            inp_map2 = inp_map2.reshape(int(num_cells_y),
                                        int(num_cells_x*0.5))
            out_map2 = inp_map2.copy()
            # Create snake pattern in matrix
            # For every second column, order of numbers is reversed.
            out_map2[1::2, :] = inp_map2[1::2, ::-1]
            idx_map = np.concatenate((out_map1, out_map2), axis=1)
        else:
            inp_map1 = np.arange(int(num_cells_x*num_cells_y*0.5),
                                 dtype=int)
            inp_map1 = inp_map1.reshape(int(num_cells_x),
                                        int(num_cells_y*0.5)).T
            out_map1 = inp_map1.copy()
            # Create snake pattern in matrix
            # For every second column, order of numbers is reversed.
            out_map1[:, ::2] = inp_map1[::-1, ::2]
            inp_map2 = np.arange(int(num_cells_x*num_cells_y*0.5),
                                 int(num_cells_x*num_cells_y),
                                 dtype=int)
            inp_map2 = inp_map2.reshape(int(num_cells_x),
                                        int(num_cells_y*0.5)).T
            out_map2 = inp_map2.copy()
            # Create snake pattern in matrix
            # For every second column, order of numbers is reversed.
            out_map2[:, 1::2] = inp_map2[::-1, 1::2]
            idx_map = np.concatenate((out_map1, out_map2), axis=0)
    else:
        raise ValueError(
            'Snaking Top & Bottom halves is not possible with this many parallel strings')
    return idx_map


def create_all_parallel(num_cells_x, num_cells_y,
                        Cell_rotated=False):
    """
    Create topological index map for an All Parallel circuit module.

    The number of cells in the X & Y direction are specified.

    Parameters
    ----------
    num_cells_x : Int
        Number of cells in X direction or number of cell columns.
    num_cells_y : Int
        Number of cells in Y direction or number of cell rows.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Returns
    -------
    idx_map : numpy.ndarray
        Returns an array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    """
    if Cell_rotated:
        # Create regular matrix with arange, reshape to required matrix size.
        inp_map = np.arange(int(num_cells_y*num_cells_x),
                            dtype=int).reshape(int(num_cells_y), int(num_cells_x)).T
    else:
        # Create regular matrix with arange, reshape to required matrix size.
        inp_map = np.arange(int(num_cells_x*num_cells_y),
                            dtype=int).reshape(int(num_cells_x), int(num_cells_y)).T
    return inp_map


def create_sys_idx_map(str_len, num_str):
    """
    Generate the system index map containing the modules in a string and the number of strings.

    Parameters
    ----------
    str_len : int
        Module string length.
    num_str : int
        Number of strings in the system.

    Returns
    -------
    sys_idx_map : numpy.ndarray
        Returns an array of size num_str X str_len with indices for
        each module in the system.

    """
    sys_idx_map = np.tile(np.arange(str_len), (num_str, 1))
    return sys_idx_map


def create_cell_coordinates(idx_map, cell_X, cell_Y,
                            CellSpace_X, CellSpace_Y,
                            EdgeSpace_X, EdgeSpace_Y,
                            Tilt, str_tilt,
                            Mod_orig_X=0, Mod_orig_Y=0,
                            is_Mod=False,
                            is_Landscape=False):
    """
    Calculate cell coordinates and store in a numpy array, given the cell and module dimensions.

    Parameters
    ----------
    idx_map : numpy.ndarray
        An array of size num_cells_y X num_cells_x with indices for
        each cell in the module.
    cell_X : float
        size of cell in X-direction in cm.
    cell_Y : float
        size of cell in Y-direction in cm.
    CellSpace_X : float
        Spacing between cells in X-direction in cm.
    CellSpace_Y : float
        Spacing between cells in Y-direction in cm.
    EdgeSpace_X : float
        Spacing between cell and module edge in X-direction in cm.
    EdgeSpace_Y : float
        Spacing between cell and module edge in Y-direction in cm.
    Tilt : float
        Tilt angle in degrees.
    str_tilt : bool
        Is each string tilted separately?
    Mod_orig_X : float, optional
        X-coordinate of Module in system plane.
    Mod_orig_Y : float, optional
        Y-coordinate of Module in system plane.
    is_Mod : bool, optional
        Do you need to generate the module coordinates instead of cells?
    is_Landscape : bool, optional
        Is the module orientation in Landscape?

    Returns
    -------
    cell_coord_array : numpy.ndarray
        Numpy array of size cell_y X cell_x X 4 edges X 2 axis (x&Y). This is a
        4D array.

    """
    # Check if landscape
    if is_Landscape and not is_Mod:
        idx_map = np.rot90(idx_map)
        cell_X, cell_Y = cell_Y, cell_X
        CellSpace_X, CellSpace_Y = CellSpace_Y, CellSpace_X
        EdgeSpace_X, EdgeSpace_Y = EdgeSpace_Y, EdgeSpace_X
    elif is_Landscape and is_Mod:
        cell_X, cell_Y = cell_Y, cell_X
        CellSpace_X, CellSpace_Y = CellSpace_Y, CellSpace_X
        EdgeSpace_X, EdgeSpace_Y = EdgeSpace_Y, EdgeSpace_X

    # Update CellSpace_Y if it is string tilt & Module
    # if str_tilt and is_Mod:
    #     CellSpace_Y = CellSpace_Y - (cell_Y - cell_Y*np.cos(np.deg2rad(Tilt)))
    # Define cell coordinates

    # 1. Arrangement of vertices in array: (BL, TL, TR, BR)
    # 2. Arrangement of axis in array: (X, Y)
    cell_coord_array = np.zeros((idx_map.shape[0],
                                 idx_map.shape[1], 4, 2))

    # X-coordinates
    # The coordinates for these points can be calculated using the
    # simple formula (utilizing the column indexes),

    # $$ X = EdgeSpace\_X + idx\_X (cell\_X + CellSpace\_X) $$

    col_idx_array = np.arange(idx_map.shape[1])
    col_idx_array = np.tile(col_idx_array, (idx_map.shape[0], 1))

    # Bottom Left
    if is_Landscape:
        if len(cell_X) < col_idx_array.shape[0]:
            cell_X = np.array(cell_X * col_idx_array.shape[0])
            cell_Y = np.array(cell_Y * col_idx_array.shape[0])
        else:
            cell_X = np.array(cell_X)
            cell_Y = np.array(cell_Y)
        cell_X = cell_X[..., None]
        cell_Y = cell_Y[..., None]
        cell_X = np.tile(cell_X, (1, col_idx_array.shape[1]))
        cell_Y = np.tile(cell_Y, (1, col_idx_array.shape[1]))
    else:
        if len(cell_X) < col_idx_array.shape[1]:
            cell_X = np.array(cell_X * col_idx_array.shape[1])
            cell_Y = np.array(cell_Y * col_idx_array.shape[1])
        else:
            cell_X = np.array(cell_X)
            cell_Y = np.array(cell_Y)
        cell_X = np.tile(cell_X, (col_idx_array.shape[0], 1))
        cell_Y = np.tile(cell_Y, (col_idx_array.shape[0], 1))
    for col_idx in range(col_idx_array.shape[1]):
        if col_idx == 0:
            cell_coord_array[:, col_idx, 0, 0] = Mod_orig_X + EdgeSpace_X
        else:
            cell_coord_array[:, col_idx, 0, 0] = cell_coord_array[:,
                                                                  col_idx-1, 0, 0] + cell_X[:, col_idx-1] + CellSpace_X
    # Top Left
    cell_coord_array[:, :, 1, 0] = cell_coord_array[:, :, 0, 0].copy()
    # Top Right

    # The coordinates for the right points can be calculated using the simple formula

    # $$ X = EdgeSpace\_X + cell\_X + idx\_X (CellSpace\_X + cell\_X)
    for col_idx in range(col_idx_array.shape[1]):
        if col_idx == 0:
            cell_coord_array[:, col_idx, 2, 0] = Mod_orig_X + EdgeSpace_X + cell_X[:, col_idx]
        else:
            cell_coord_array[:, col_idx, 2, 0] = cell_coord_array[:, col_idx-1, 2, 0] + CellSpace_X + cell_X[:, col_idx]
    # Bottom Right
    cell_coord_array[:, :, 3, 0] = cell_coord_array[:, :, 2, 0].copy()

    # Y-coordinates
    # Assumption: All cells have the same width.

    # The Y coordinates for each of the points can be similarly calculated
    # using the same formula.
    row_idx_array = np.flip(np.arange(idx_map.shape[0]))
    row_idx_array = row_idx_array[..., None]
    row_idx_array = np.tile(row_idx_array, (1, idx_map.shape[1]))

    # Bottom Left
    cell_coord_array[:, :, 0, 1] = Mod_orig_Y + \
        EdgeSpace_Y + row_idx_array*(cell_Y + CellSpace_Y)
    # Top Left
    cell_coord_array[:, :, 1, 1] = Mod_orig_Y + EdgeSpace_Y + \
        cell_Y + row_idx_array*(cell_Y + CellSpace_Y)
    # Top Right
    cell_coord_array[:, :, 2, 1] = cell_coord_array[:, :, 1, 1].copy()
    # Bottom Right
    cell_coord_array[:, :, 3, 1] = cell_coord_array[:, :, 0, 1].copy()

    return cell_coord_array


def cell_ploygons(cell_array):
    """
    Generate the shapely polygons for the cells in the module.

    Parameters
    ----------
    cell_array : numpy.ndarray
        Cell coordinates in the module.

    Returns
    -------
    cell_poly_df : pandas.DataFrame
        Dataframe containing the cell polygons.

    """
    cell_poly_array = np.zeros((cell_array.shape[0], cell_array.shape[1]))
    cell_poly_df = pd.DataFrame(cell_poly_array)
    # Run for loop through cells and generate the polygons
    for idx_row in range(cell_array.shape[0]):
        for idx_col in range(cell_array.shape[1]):
            lst_tup = []
            for idx_vrt in range(cell_array.shape[2]):
                lst_tup.append(
                    (cell_array[idx_row, idx_col, idx_vrt, 0], cell_array[idx_row, idx_col, idx_vrt, 1]))
            cell_poly = Polygon(lst_tup)
            cell_poly_df.iloc[idx_row, idx_col] = cell_poly
    return cell_poly_df


def calc_numcols_diode(num_x, num_y, num_diodes, is_series, num_parallel,
                       par_lyt_type='snake_together', Cell_rotated=False):
    """
    Calculate the number cells or cell columns in parallel to a diode.

    Parameters
    ----------
    num_cells_x : int
        Number of cells in X direction or number of cell columns.
    num_cells_y : int
        Number of cells in Y direction or number of cell rows.
    num_diodes : int
        Number of diodes in the module electric circuit.
    is_series : Boolean
        Flag to indicate if the cirsuit is a Series or Parallel. True for Series.
    num_parallel : Int, optional
        Number of parallel substrings. The default is 2.
    par_lyt_type : string, optional
        Layout Type. The default is 'snake_together'. All options include:
            1) snake_together
            2) LR_half --> Parallel substring start in the middle and cut module along Y direction.
            3) TB_half --> Parallel substring start in the middle and cut module along X direction.
            4) all_parallel --> All cell columns are parallel substrings. Eg. P-series modules.
    Cell_rotated : bool, optional
        Are the cells rotated by 90 deg (a.k.a horizontal stringing)? The default is False.

    Returns
    -------
    list
        For series modules returns a list of number of cell columns per diode.
        For parallel modules returns a list of number of cells in a parallel
        substring per diode.

    """
    if Cell_rotated:
        num_x, num_y = num_y, num_x
    if is_series:
        # If series
        # Check if number of cell columns are divisable by num of diodes.
        # If yes, return a list of equal number of cell columns.
        if num_x % num_diodes == 0:
            return [int(num_x/num_diodes)] * num_diodes
        else:
            # First find the base factor and the remainder
            factor = int(num_x/num_diodes)
            remainder = num_x % num_diodes
            # Create a list of base factor only
            numcols_diode = [factor] * num_diodes
            # Find remainder of middle columns
            rem_mid = (factor + remainder) % (num_diodes-2)
            # If remainder is zero,
            if rem_mid == 0:
                # Split the remaining number columns equally
                for idx_col in range(1, num_diodes-1):
                    numcols_diode[idx_col] = int(
                        (factor + remainder) / (num_diodes-2))
            else:
                # Split equally by factor first.
                for idx_col in range(1, num_diodes-1):
                    numcols_diode[idx_col] = int(rem_mid)
                # Add remainder to middle of the list
                numcols_diode[find_middle(numcols_diode)] = int(
                    (factor + remainder) / (num_diodes-2)) + int(rem_mid)
            return numcols_diode
    else:
        # Parallel Cases except ALL PARRALLEL: A little more complicated.
        # Need to use cells in X & Y direction.
        if par_lyt_type != 'all_parallel':
            # Check if number of cell columns are divisable by num of diodes.
            # If yes, return a list of equal number of cell columns times row cells by parallel strings.
            if num_x % num_diodes == 0:
                return [int(num_x*num_y/num_diodes/num_parallel)] * num_diodes
            else:
                # First find the base factor and the remainder
                factor = int(num_x/num_diodes)
                remainder = num_x % num_diodes
                # Create a list of base factor only
                numcols_diode = [int(factor*num_y/num_parallel)] * num_diodes
                # Find remainder of middle columns
                rem_mid = ((num_diodes-2)*factor + remainder) % (num_diodes-2)
                if rem_mid == 0:
                    # Split the remaining number columns equally
                    for idx_col in range(1, num_diodes-1):
                        numcols_diode[idx_col] = int(
                            (factor + remainder)*num_y / (num_diodes-2) / num_parallel)
                else:
                    # Split equally by factor first.
                    for idx_col in range(1, num_diodes-1):
                        numcols_diode[idx_col] = factor + rem_mid
                    numcols_diode[find_middle(numcols_diode)] = 0
                    # Calculate remaining number of cells.
                    remaining_cells = num_y*num_x / \
                        num_parallel - sum(numcols_diode)
                    # Add remaining cells to middle of the list
                    numcols_diode[find_middle(numcols_diode)] = remaining_cells
                return numcols_diode
        else:
            # ALL PARALLEL Case" Easier. Need to use cells in Y direction only.
            # Check if number of cell columns are divisable by num of diodes.
            # If yes, return a list of equal number of cell columns times row cells by parallel strings.
            if num_y % num_diodes == 0:
                return [int(num_y/num_diodes)] * num_diodes
            else:
                # First find the base factor and the remainder
                factor = int(num_y/num_diodes)
                remainder = num_y % num_diodes
                # Create a list of base factor only
                numcols_diode = [int(factor)] * num_diodes
                # Trivial case of just 2 diodes: Just add additional cells to last diode
                if num_diodes == 2:
                    numcols_diode[1] = numcols_diode[1] + remainder
                else:
                    # Find remainder of middle columns
                    rem_mid = ((num_diodes-2)*factor +
                               remainder) % (num_diodes-2)
                    if rem_mid == 0:
                        # Split the remaining number columns equally
                        for idx_col in range(1, num_diodes-1):
                            numcols_diode[idx_col] = int(
                                (factor + remainder) / (num_diodes-2))
                    else:
                        # Split equally by factor first.
                        for idx_col in range(1, num_diodes-1):
                            numcols_diode[idx_col] = factor + rem_mid
                        numcols_diode[find_middle(numcols_diode)] = 0
                        # Calculate remaining number of cells.
                        remaining_cells = num_y - sum(numcols_diode)
                        # Add remaining cells to middle of the list
                        numcols_diode[find_middle(
                            numcols_diode)] = remaining_cells
                return numcols_diode


def set_crosstie(cell_pos_in, value_crosstie=False):
    """
    # Find cell with idx, then set crosstie to input True/False value.

    # Return cell_pos_out
    #
    # cell_pos is a list of lists of lists, so this function first finds the cell
    # item with the indicated idx

    Parameters
    ----------
    cell_pos_in : List
        List of lists of lists containing the electrical circuit architecture for
        a PVMM PV module with indices and cross tie information.
    idx : int
        Index for which Crosstie needs to be set to value_crosstie in the cell_pos.
    value_crosstie : Boolean, optional
        Flag indicating whether to set the crossite of idx to True or False.
        The default is True.

    Returns
    -------
    cell_pos_out : List
        List of lists of lists containing the electrical circuit architecture for
        a PVMM PV module with indices and cross tie information. The crossties
        are updated based on input.

    """
    cell_pos_out = cell_pos_in.copy()

    for i, substr in enumerate(cell_pos_in):  # each substring
        # each column of cells in substring
        for j, cell_col in enumerate(substr):
            for k, cell in enumerate(cell_col):  # each cell
                cell_pos_out[i][j][k]['crosstie'] = value_crosstie

    return cell_pos_out


def set_idx(cell_pos_in, idx, value_crosstie=True):
    """
    # Find cell with idx, then set crosstie to input True/False value.

    # Return cell_pos_out
    #
    # cell_pos is a list of lists of lists, so this function first finds the cell
    # item with the indicated idx

    Parameters
    ----------
    cell_pos_in : List
        List of lists of lists containing the electrical circuit architecture for
        a PVMM PV module with indices and cross tie information.
    idx : int
        Index for which Crosstie needs to be set to value_crosstie in the cell_pos.
    value_crosstie : Boolean, optional
        Flag indicating whether to set the crossite of idx to True or False.
        The default is True.

    Returns
    -------
    cell_pos_out : List
        List of lists of lists containing the electrical circuit architecture for
        a PVMM PV module with indices and cross tie information. The crossties
        are updated based on input.

    """
    cell_pos_out = cell_pos_in.copy()

    for i, substr in enumerate(cell_pos_in):  # each substring
        # each column of cells in substring
        for j, cell_col in enumerate(substr):
            for k, cell in enumerate(cell_col):  # each cell
                if cell['idx'] == idx:
                    cell_pos_out[i][j][k]['crosstie'] = value_crosstie

    return cell_pos_out


def create_pvcell(pvcell_params, cell_name, pvconst, Ee=1., Tcell=298.15):
    """
    # Create PVMismatch pvcell from pvcell_params for specified cell.

    # Default Tcell = 25C

    Parameters
    ----------
    pvcell_params : Dataframe
        Cell Parameter DB.
    cell_name : Cell Technology
        Eg. P6_v01.
    pvconst : Class
        PVMM class containing constants used for the simulation.
    Tcell : Float, optional
        Cell Temperature in K. The default is 298.15.

    Returns
    -------
    pvc : PVMM Class
        PVMM cell based on inputs.

    """
    # parameters for specified cell_name
    pvc_params = pvcell_params[cell_name]

    # create pvcell
    pvc = pvcell.PVcell(pvconst=pvconst, Rs=pvc_params['Rs'],
                        Rsh=pvc_params['Rsh'],
                        Isat1_T0=pvc_params['Isat1_T0'],
                        Isat2_T0=pvc_params['Isat2_T0'],
                        Isc0_T0=pvc_params['Isc0_T0'],
                        aRBD=pvc_params['aRBD'],
                        bRBD=pvc_params['bRBD'],
                        VRBD=pvc_params['VRBD'],
                        nRBD=pvc_params['nRBD'],
                        Eg=pvc_params['Eg'],
                        alpha_Isc=pvc_params['alpha_Isc'],
                        Tcell=Tcell, Ee=Ee)

    return pvc
