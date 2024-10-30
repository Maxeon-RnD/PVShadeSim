# -*- coding: utf-8 -*-
"""Generate physical and cell level irradiance levels of shading and system."""

import os
from pathlib import Path
import ast
from random import randrange, uniform
import itertools
import random
import re

import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.affinity import scale, rotate, translate

from .utils import ranges, load_pickle, save_pickle
from .plotting import plot_shade_module, plot_shade_array


def gen_shade_scenarios(mods_sys_dict, pvshade_params, search_idx_name,
                        gen_sh_sce=False, gen_sh_arr=False):
    """
    Generate all shade scenarios in simulations. If required, plot them too.

    Parameters
    ----------
    mods_sys_dict : dict
        Dict containing physical and electric models of modules in simulation.
    pvshade_params : pandas.DataFrame
        Dataframe containing the PV shade database.
    search_idx_name : str
        Which index of database to search for the filter.
    gen_sh_sce : bool, optional
        Generate plot of shade scenarios? The default is False.
    gen_sh_arr : bool, optional
        Generate plot of cell intersection arrays with shade scenarios.
        The default is False.

    Returns
    -------
    mods_sys_dict : dict
        Dict containing the physical and electrical models of modules in the
        simulation + shading data.

    """
    # Get current working directory (old)
    cw = os.getcwd()
    # Create new folder
    newpath = os.path.join(cw, r'shade_scenarios')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    col_list = ['Scenario Definition', 'Scenario Type', 'Scenario Variation',
                'Shade Array', 'Shade Polygon',
                'Module Shaded Area', 'Module Shaded Area Percentage'
                ]
    wp_cnt = 0
    ch_cnt = 0
    vp_cnt = 0
    mod_sys_keys = list(mods_sys_dict.keys())
    for mod_name in mod_sys_keys:
        cell_mod_keys = list(mods_sys_dict[mod_name].keys())
        for cell_name in cell_mod_keys:
            orient_keys = list(mods_sys_dict[mod_name][cell_name].keys())
            for orient in orient_keys:
                ec_keys = list(
                    mods_sys_dict[mod_name][cell_name][orient].keys())
                for ec_type in ec_keys:
                    maxsys_dict = mods_sys_dict[mod_name][cell_name][orient][ec_type]
                    df_shd_sce = pd.DataFrame(columns=col_list)
                    # Create No shade scenario
                    df_new_row = pd.DataFrame(data=[['Standard',
                                                     'No Shade',
                                                     'Base Case',
                                                     np.zeros((maxsys_dict['Physical_Info']['Cell_Coordinates'].shape[0],
                                                               maxsys_dict['Physical_Info']['Cell_Coordinates'].shape[1])),
                                                     create_rectangle(
                                                         (-10000, -10000),
                                                         1e-3, 1e-3,
                                                         rot_ang=0),
                                                     0,
                                                     0
                                                     ]], columns=col_list)
                    df_shd_sce = pd.concat(
                        [df_shd_sce, df_new_row], ignore_index=True)
                    shade_list = maxsys_dict['shade_list']
                    shade_list = shade_list.split(", ")
                    # Search shade_db for specified shade list
                    for shade_case in shade_list:
                        # If search is not column name
                        if search_idx_name != 'uniqueName':
                            filt = (pvshade_params == shade_case).any()
                            sub_df = pvshade_params.loc[:, filt]
                        else:
                            sub_df = sub_df[shade_list]
                        # Get column names
                        sce_list = list(sub_df.columns.values)
                        for scen in sce_list:
                            # Check which function needs to be used
                            func_name = sub_df.loc['shade_type', [scen]][0]
                            if func_name == '1cell':
                                # Extract info
                                shd_cell_prop_lst = ast.literal_eval(
                                    sub_df.loc['shade_cell_prop', [scen]][0])
                                shd_cell_prop = np.arange(
                                    shd_cell_prop_lst[0],
                                    shd_cell_prop_lst[1]+1e-3,
                                    shd_cell_prop_lst[2])
                                shd_cell_idx = sub_df.loc['shade_cell_idx',
                                                          [scen]][0]
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                shd_dir = sub_df.loc['cen_pt', [scen]][0]
                                if shd_dir == '[]':
                                    shd_dir = 'Y'
                                # Generate shade case
                                df_shd_sce = shade_1cell(
                                    maxsys_dict, df_shd_sce, shd_cell_prop,
                                    shd_cell_idx, shd_dir,
                                    translucence, dir_diff_ratio)
                            elif func_name == 'ncells':
                                # Extract info
                                num_cells = int(sub_df.loc['num_obj',
                                                           [scen]][0])
                                shd_cell_idx = ast.literal_eval(
                                    sub_df.loc['shade_cell_idx', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                # Generate shade case
                                df_shd_sce = shade_ncells(maxsys_dict,
                                                          df_shd_sce,
                                                          num_cells,
                                                          shd_cell_idx_lst=shd_cell_idx,
                                                          translucence=translucence,
                                                          dir_diff_ratio=dir_diff_ratio)
                            elif func_name == 'substring':
                                # Extract info
                                substr_diodes = ast.literal_eval(sub_df.loc['shade_cell_idx',
                                                                            [scen]][0])
                                mod_prop_vec = ast.literal_eval(sub_df.loc['shade_mod_prop',
                                                                           [scen]][0])
                                translucence = float(sub_df.loc['translucence',
                                                                [scen]][0])
                                dir_diff_ratio = float(sub_df.loc['dir_diff_ratio',
                                                                  [scen]][0])
                                df_shd_sce = shade_substr_per_diode(maxsys_dict,
                                                                    df_shd_sce,
                                                                    substr_diodes,
                                                                    mod_prop_vec=mod_prop_vec,
                                                                    translucence=translucence,
                                                                    dir_diff_ratio=dir_diff_ratio)
                            elif func_name == 'leaves_birddroppings':
                                num_obj = int(sub_df.loc['num_obj', [scen]][0])
                                maj_dia = ast.literal_eval(
                                    sub_df.loc['x_len', [scen]][0])
                                min_dia = ast.literal_eval(
                                    sub_df.loc['y_len', [scen]][0])
                                rot_ang = ast.literal_eval(
                                    sub_df.loc['rot_ang', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                use_std = sub_df.loc['use_std', [scen]][0]
                                pickle_fn = sub_df.loc['pickle_file', [
                                    scen]][0]
                                df_shd_sce = shade_leaves_birddroppings(maxsys_dict,
                                                                        df_shd_sce,
                                                                        num_obj,
                                                                        maj_dia,
                                                                        min_dia,
                                                                        rot_ang,
                                                                        translucence,
                                                                        dir_diff_ratio,
                                                                        use_std,
                                                                        pickle_fn)
                            elif func_name == 'row' or func_name == 'col':
                                rowcol_type = sub_df.loc['rowcol_type', [
                                    scen]][0]
                                mod_prop_vec = ast.literal_eval(
                                    sub_df.loc['shade_mod_prop', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                cen_pt = sub_df.loc['cen_pt', [scen]][0]
                                df_shd_sce = shade_row_col(maxsys_dict,
                                                           df_shd_sce,
                                                           func_name,
                                                           rowcol_type,
                                                           mod_prop_vec,
                                                           translucence,
                                                           dir_diff_ratio,
                                                           cen_pt)
                            elif func_name == 'bottom_edge_soiling_row' or func_name == 'bottom_edge_soiling_col':
                                rowcol_type = sub_df.loc['rowcol_type', [
                                    scen]][0]
                                shd_type = func_name.replace(
                                    'bottom_edge_soiling_', '')
                                mod_prop_vec = ast.literal_eval(
                                    sub_df.loc['shade_mod_prop', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = bottom_edge_soiling(maxsys_dict,
                                                                 df_shd_sce,
                                                                 bes_shape='rectangle',
                                                                 mod_prop_vec=mod_prop_vec,
                                                                 translucence=translucence,
                                                                 dir_diff_ratio=dir_diff_ratio,
                                                                 shd_type=shd_type,
                                                                 rowcol_type=rowcol_type)
                            elif func_name == 'bottom_edge_soiling_row_singtriang' or func_name == 'bottom_edge_soiling_col_singtriang':
                                rowcol_type = sub_df.loc['rowcol_type', [
                                    scen]][0]
                                shd_type = func_name.replace(
                                    'bottom_edge_soiling_', '')
                                shd_type = shd_type.replace('_singtriang', '')
                                mod_prop_vec = ast.literal_eval(
                                    sub_df.loc['shade_mod_prop', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = bottom_edge_soiling(maxsys_dict,
                                                                 df_shd_sce,
                                                                 bes_shape='single_triangle',
                                                                 mod_prop_vec=mod_prop_vec,
                                                                 translucence=translucence,
                                                                 dir_diff_ratio=dir_diff_ratio,
                                                                 shd_type=shd_type,
                                                                 rowcol_type=rowcol_type)
                            elif func_name == 'bottom_edge_soiling_row_doubtriang' or func_name == 'bottom_edge_soiling_col_doubtriang':
                                rowcol_type = sub_df.loc['rowcol_type', [
                                    scen]][0]
                                shd_type = func_name.replace(
                                    'bottom_edge_soiling_', '')
                                shd_type = shd_type.replace('_doubtriang', '')
                                mod_prop_vec = ast.literal_eval(
                                    sub_df.loc['shade_mod_prop', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = bottom_edge_soiling(maxsys_dict,
                                                                 df_shd_sce,
                                                                 bes_shape='double_triangle',
                                                                 mod_prop_vec=mod_prop_vec, translucence=translucence,
                                                                 dir_diff_ratio=dir_diff_ratio, shd_type=shd_type,
                                                                 rowcol_type=rowcol_type)
                            elif func_name == 'Wire Pole' or func_name == 'Chimney':
                                rot_ang = ast.literal_eval(
                                    sub_df.loc['rot_ang', [scen]][0])
                                rot_ang_vec = np.arange(
                                    rot_ang[0], rot_ang[1], rot_ang[2])
                                sh_width = ast.literal_eval(
                                    sub_df.loc['y_len', [scen]][0])
                                sh_width_vec = np.arange(
                                    sh_width[0], sh_width[1], sh_width[2])
                                mod_prop_vec = ast.literal_eval(
                                    sub_df.loc['shade_mod_prop', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                use_std = sub_df.loc['use_std', [scen]][0]
                                pickle_fn = sub_df.loc['pickle_file', [
                                    scen]][0]
                                cen_pt = sub_df.loc['cen_pt', [scen]][0]
                                if '[' in cen_pt:
                                    cen_pt = ast.literal_eval(cen_pt)
                                df_shd_sce, sh_width_vec = shade_rot_rectangle(maxsys_dict,
                                                                               df_shd_sce,
                                                                               rot_ang_vec,
                                                                               sh_width_vec,
                                                                               translucence,
                                                                               dir_diff_ratio,
                                                                               use_std,
                                                                               pickle_fn,
                                                                               func_name,
                                                                               cen_pt)
                                if func_name == 'Wire Pole' and wp_cnt == 0:
                                    width_lst = ['Width {}'.format(
                                        i) for i in range(1,
                                                          len(sh_width_vec)+1)]
                                    wp_wid_col_list = [
                                        'Module', 'Cell',
                                        'Orientation'] + width_lst
                                    wire_pole_df = pd.DataFrame(
                                        columns=wp_wid_col_list)
                                    wp_cnt += 1

                                if func_name == 'Chimney' and ch_cnt == 0:
                                    width_lst = ['Width {}'.format(
                                        i) for i in range(1,
                                                          len(sh_width_vec)+1)]
                                    ch_wid_col_list = [
                                        'Module', 'Cell',
                                        'Orientation'] + width_lst
                                    chimney_pole_df = pd.DataFrame(
                                        columns=ch_wid_col_list)
                                    ch_cnt += 1

                                if func_name == 'Wire Pole':
                                    df_new_row = pd.DataFrame(data=[[mod_name,
                                                                     cell_name,
                                                                     orient
                                                                     ] + sh_width_vec.tolist()],
                                                              columns=wp_wid_col_list)
                                    wire_pole_df = pd.concat(
                                        [wire_pole_df, df_new_row],
                                        ignore_index=True)

                                else:
                                    df_new_row = pd.DataFrame(data=[[mod_name,
                                                                     cell_name,
                                                                     orient
                                                                     ] + sh_width_vec.tolist()],
                                                              columns=ch_wid_col_list)
                                    chimney_pole_df = pd.concat(
                                        [chimney_pole_df, df_new_row],
                                        ignore_index=True)

                            elif func_name == 'Pipe':
                                rot_ang = ast.literal_eval(
                                    sub_df.loc['rot_ang', [scen]][0])
                                rot_ang_vec = np.arange(
                                    rot_ang[0], rot_ang[1]+1, rot_ang[2])
                                # print(rot_ang_vec)
                                sh_len = ast.literal_eval(
                                    sub_df.loc['x_len', [scen]][0])
                                if isinstance(sub_df.loc['shade_mod_prop',
                                                         [scen]][0],
                                              (int, float)):
                                    use_len = sub_df.loc['shade_mod_prop',
                                                         [scen]][0]
                                else:
                                    use_len = ast.literal_eval(sub_df.loc['shade_mod_prop',
                                                                          [scen]][0])
                                if use_len:
                                    sh_len_vec = np.arange(
                                        sh_len[0], sh_len[1]+1, sh_len[2])
                                else:
                                    sh_len_vec = np.arange(25, 25, 1)
                                sh_width = ast.literal_eval(
                                    sub_df.loc['y_len', [scen]][0])
                                sh_width_vec = np.arange(
                                    sh_width[0], sh_width[1]+1, sh_width[2])
                                # print(sh_width_vec)
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                use_std = sub_df.loc['use_std', [scen]][0]
                                pickle_fn = sub_df.loc['pickle_file', [
                                    scen]][0]
                                cen_pt = sub_df.loc['cen_pt', [scen]][0]
                                if '[' in cen_pt:
                                    cen_pt = ast.literal_eval(cen_pt)
                                df_shd_sce, sh_width_vec = shade_pipe(maxsys_dict,
                                                                      df_shd_sce,
                                                                      rot_ang_vec,
                                                                      sh_width_vec,
                                                                      sh_len_vec,
                                                                      translucence,
                                                                      dir_diff_ratio,
                                                                      use_std,
                                                                      use_len,
                                                                      pickle_fn,
                                                                      func_name,
                                                                      cen_pt)
                                if vp_cnt == 0:
                                    width_lst = ['Width {}'.format(
                                        i) for i in range(1,
                                                          len(sh_width_vec)+1)]
                                    wid_col_list = [
                                        'Module', 'Cell',
                                        'Orientation'] + width_lst
                                    pipe_pole_df = pd.DataFrame(
                                        columns=wid_col_list)
                                    vp_cnt += 1
                                df_new_row = pd.DataFrame(data=[[mod_name,
                                                                 cell_name,
                                                                 orient
                                                                 ] + sh_width_vec.tolist()],
                                                          columns=wid_col_list)
                                pipe_pole_df = pd.concat(
                                    [pipe_pole_df, df_new_row],
                                    ignore_index=True)

                            elif func_name == 'mixed1':
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = shade_mixed1(
                                    maxsys_dict, df_shd_sce, translucence,
                                    dir_diff_ratio)
                            elif func_name == 'mixed2':
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = shade_mixed2(
                                    maxsys_dict, df_shd_sce, translucence,
                                    dir_diff_ratio)
                            elif func_name == 'tree1':
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = shade_tree1(
                                    maxsys_dict, df_shd_sce, translucence,
                                    dir_diff_ratio)
                            elif func_name == 'tree2':
                                rot_ang = ast.literal_eval(
                                    sub_df.loc['rot_ang', [scen]][0])
                                cen_pt = sub_df.loc['cen_pt', [scen]][0]
                                if '[' in cen_pt:
                                    cen_pt = ast.literal_eval(cen_pt)
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = shade_tree2(maxsys_dict,
                                                         df_shd_sce,
                                                         rot_ang, cen_pt,
                                                         translucence,
                                                         dir_diff_ratio)
                            elif func_name == 'tree3':
                                rot_ang = ast.literal_eval(
                                    sub_df.loc['rot_ang', [scen]][0])
                                cen_pt = sub_df.loc['cen_pt', [scen]][0]
                                if '[' in cen_pt:
                                    cen_pt = ast.literal_eval(cen_pt)
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                df_shd_sce = shade_tree3(maxsys_dict,
                                                         df_shd_sce,
                                                         rot_ang, cen_pt,
                                                         translucence,
                                                         dir_diff_ratio)
                            elif func_name == 'user_defined':
                                obj_shape = ast.literal_eval(
                                    sub_df.loc['shade_mod_prop', [scen]][0])
                                x_len = ast.literal_eval(
                                    sub_df.loc['x_len', [scen]][0])
                                y_len = ast.literal_eval(
                                    sub_df.loc['y_len', [scen]][0])
                                rot_ang = ast.literal_eval(
                                    sub_df.loc['rot_ang', [scen]][0])
                                cen_pt = ast.literal_eval(
                                    sub_df.loc['cen_pt', [scen]][0])
                                translucence = float(
                                    sub_df.loc['translucence', [scen]][0])
                                dir_diff_ratio = float(
                                    sub_df.loc['dir_diff_ratio', [scen]][0])
                                scen_name = sub_df.loc['scenario_type',
                                                       [scen]][0]
                                df_shd_sce = shade_user_define_objects(maxsys_dict,
                                                                       df_shd_sce,
                                                                       cen_pt,
                                                                       x_len,
                                                                       y_len,
                                                                       rot_ang,
                                                                       obj_shape,
                                                                       translucence,
                                                                       dir_diff_ratio,
                                                                       scen_name)
                            else:
                                raise ValueError(
                                    'No matching shading functions found: ' + func_name)
                    mods_sys_dict[mod_name][cell_name][orient][ec_type]['Shade Scenarios'] = df_shd_sce
                    # Plot Shade Scenarios
                    if gen_sh_sce:
                        # Change directory to new
                        os.chdir(newpath)

                        plot_label = mods_sys_dict[mod_name][cell_name][orient][ec_type]['Sim_info']['plot_label']
                        pdf_fname = plot_label + '_Shade_Scenarios.pdf'
                        plot_shade_module(
                            maxsys_dict, df_shd_sce, plot_file=pdf_fname)

                        # Change current working directly to old
                        os.chdir(cw)
                    # Plot Shade Arrays
                    if gen_sh_arr:
                        is_Landscape = mods_sys_dict[mod_name][cell_name][orient][ec_type]['Physical_Info']['is_Landscape']
                        plot_label = mods_sys_dict[mod_name][cell_name][orient][ec_type]['Sim_info']['plot_label']
                        plot_shade_array(df_shd_sce, plot_label, is_Landscape)
                    # if 'wire_pole_df' in locals():
                    #     # dfi.export(wire_pole_df, 'Wire_Pole_Widths.png')
                    # if 'chimney_pole_df' in locals():
                    #     # dfi.export(chimney_pole_df, 'Chimney_Widths.png')
                    # if 'pipe_pole_df' in locals():
                        # dfi.export(pipe_pole_df, 'Vent_Pipe_Widths.png')
    return mods_sys_dict


def create_rectangle(center_pt, xlen, ylen, rot_ang):
    """
    Generate a rectangle shapely along with rotation.

    Parameters
    ----------
    center_pt : list
        Coordinates of the center point of the rectangle.
    xlen : float
        Length of the rectangle.
    ylen : float
        Width of the rectangle.
    rot_ang : float
        Rotation angle of the rectangle in degrees.

    Returns
    -------
    rect_shply : shapely
        Rectangle shapely.

    """
    # Convert center point to numpy array
    center_pt = np.array([[center_pt[0]], [center_pt[1]]])
    # Define Rotation Matrix
    rot_mat = np.array([
                       [np.cos(np.deg2rad(rot_ang)), -
                        np.sin(np.deg2rad(rot_ang))],
                       [np.sin(np.deg2rad(rot_ang)),
                        np.cos(np.deg2rad(rot_ang))]
                       ])
    # Define vertices of rectangle in rotated
    # dimensions (x', y' coordinate axis)
    rect_rot_vert = np.array([[-0.5*xlen, -0.5*xlen,  0.5*xlen,  0.5*xlen],
                              [-0.5*ylen,  0.5*ylen,  0.5*ylen, -0.5*ylen]
                              ])
    # Calculate vertices in unrotated dimensions (x,y coordinate axis)
    rect_vert = np.round(np.tile(center_pt, 4) +
                         np.matmul(rot_mat, rect_rot_vert), decimals=2)
    rect_shply = Polygon([(rect_vert[0, 0], rect_vert[1, 0]),
                          (rect_vert[0, 1], rect_vert[1, 1]),
                          (rect_vert[0, 2], rect_vert[1, 2]),
                          (rect_vert[0, 3], rect_vert[1, 3])])
    return rect_shply


def cell_overlap_area(cell_poly_df, mod_overlap, cell_area_overlap=[]):
    """
    Calculate the cell overlap area of shade polygon with all cells in module.

    Parameters
    ----------
    cell_poly_df : pandas.DataFrame
        Dataframe containing the cell polygons.
    mod_overlap : shapely.Polygon
        Intersected shade polygon with module.
    cell_area_overlap : numpy.array, optional
        Cell area overlaps for all cells in the module. The default is [].

    Returns
    -------
    cell_area_overlap : numpy.array, optional
        Cell area overlaps for all cells in the module.

    """
    if cell_area_overlap == []:
        cell_area_overlap = np.zeros(cell_poly_df.shape)
    num_cells_dim = cell_poly_df.shape
    max_dim_idx = num_cells_dim.index(max(num_cells_dim))
    if max_dim_idx == 1:
        for idx_row in range(cell_poly_df.shape[0]):
            # Get String Coordinates from Shapely
            Cellstrtb = cell_poly_df.iloc[idx_row, 0]
            Cellendb = cell_poly_df.iloc[idx_row, -1]
            Cellstrtt = cell_poly_df.iloc[idx_row, 0]
            Cellendt = cell_poly_df.iloc[idx_row, -1]
            mx, my = Cellstrtt.exterior.xy
            mp0x = mx[1]
            mp0y = my[1]
            # Get left bottom most point
            mx, my = Cellstrtb.exterior.xy
            mp1x = mx[0]
            mp1y = my[0]
            # Get right top most point
            mx, my = Cellendt.exterior.xy
            mp2x = mx[2]
            mp2y = my[2]
            # Get right bottom most point
            mx, my = Cellendb.exterior.xy
            mp3x = mx[3]
            mp3y = my[3]
            cell_str_poly = Polygon([[mp1x, mp1y], [mp0x, mp0y], [mp2x, mp2y],
                                     [mp3x, mp3y]])
            if mod_overlap.intersects(cell_str_poly):
                split_num = cell_poly_df.shape[1] // 10
                if split_num == 0:
                    split_num = 1
                col_rngs = ranges(cell_poly_df.shape[1]-1, split_num)
                for col_rng in col_rngs:
                    # Get String Coordinates from Shapely
                    Cellstrtb = cell_poly_df.iloc[idx_row, col_rng[0]]
                    Cellendb = cell_poly_df.iloc[idx_row, col_rng[1]]
                    Cellstrtt = cell_poly_df.iloc[idx_row, col_rng[0]]
                    Cellendt = cell_poly_df.iloc[idx_row, col_rng[1]]
                    mx, my = Cellstrtt.exterior.xy
                    mp0x = mx[1]
                    mp0y = my[1]
                    # Get left bottom most point
                    mx, my = Cellstrtb.exterior.xy
                    mp1x = mx[0]
                    mp1y = my[0]
                    # Get right top most point
                    mx, my = Cellendt.exterior.xy
                    mp2x = mx[2]
                    mp2y = my[2]
                    # Get right bottom most point
                    mx, my = Cellendb.exterior.xy
                    mp3x = mx[3]
                    mp3y = my[3]
                    cell_str_poly = Polygon([[mp1x, mp1y], [mp0x, mp0y],
                                             [mp2x, mp2y],
                                             [mp3x, mp3y]])
                    if mod_overlap.intersects(cell_str_poly):
                        for idx_col in range(col_rng[0], col_rng[1]+1):
                            cell_poly = cell_poly_df.iloc[idx_row, idx_col]
                            if mod_overlap.intersects(cell_poly):
                                ovrlp_poly_area = mod_overlap.intersection(
                                    cell_poly).area
                                if cell_area_overlap[idx_row, idx_col] < ovrlp_poly_area:
                                    cell_area_overlap[idx_row,
                                                      idx_col] = ovrlp_poly_area
    else:
        for idx_col in range(cell_poly_df.shape[1]):
            # Get String Coordinates from Shapely
            Cellstrtb = cell_poly_df.iloc[-1, idx_col]
            Cellendb = cell_poly_df.iloc[-1, idx_col]
            Cellstrtt = cell_poly_df.iloc[0, idx_col]
            Cellendt = cell_poly_df.iloc[0, idx_col]
            mx, my = Cellstrtt.exterior.xy
            mp0x = mx[1]
            mp0y = my[1]
            # Get left bottom most point
            mx, my = Cellstrtb.exterior.xy
            mp1x = mx[0]
            mp1y = my[0]
            # Get right top most point
            mx, my = Cellendt.exterior.xy
            mp2x = mx[2]
            mp2y = my[2]
            # Get right bottom most point
            mx, my = Cellendb.exterior.xy
            mp3x = mx[3]
            mp3y = my[3]
            cell_str_poly = Polygon([[mp1x, mp1y], [mp0x, mp0y], [mp2x, mp2y],
                                     [mp3x, mp3y]])
            if mod_overlap.intersects(cell_str_poly):
                split_num = cell_poly_df.shape[0] // 5
                if split_num == 0:
                    split_num = 1
                row_rngs = ranges(cell_poly_df.shape[0]-1, split_num)
                for row_rng in row_rngs:
                    Cellstrtb = cell_poly_df.iloc[row_rng[1], idx_col]
                    Cellendb = cell_poly_df.iloc[row_rng[1], idx_col]
                    Cellstrtt = cell_poly_df.iloc[row_rng[0], idx_col]
                    Cellendt = cell_poly_df.iloc[row_rng[0], idx_col]
                    mx, my = Cellstrtt.exterior.xy
                    mp0x = mx[1]
                    mp0y = my[1]
                    # Get left bottom most point
                    mx, my = Cellstrtb.exterior.xy
                    mp1x = mx[0]
                    mp1y = my[0]
                    # Get right top most point
                    mx, my = Cellendt.exterior.xy
                    mp2x = mx[2]
                    mp2y = my[2]
                    # Get right bottom most point
                    mx, my = Cellendb.exterior.xy
                    mp3x = mx[3]
                    mp3y = my[3]
                    cell_str_poly = Polygon([[mp1x, mp1y], [mp0x, mp0y],
                                             [mp2x, mp2y],
                                             [mp3x, mp3y]])
                    if mod_overlap.intersects(cell_str_poly):
                        for idx_row in range(row_rng[0], row_rng[1]+1):
                            cell_poly = cell_poly_df.iloc[idx_row, idx_col]
                            if mod_overlap.intersects(cell_poly):
                                ovrlp_poly_area = mod_overlap.intersection(
                                    cell_poly).area
                                if cell_area_overlap[idx_row, idx_col] < ovrlp_poly_area:
                                    cell_area_overlap[idx_row,
                                                      idx_col] = ovrlp_poly_area
    return cell_area_overlap


def calc_cell_shading(cell_area_overlap, CELLAREA, translucence=1,
                      dir_diff_ratio=1, inp_shade=[]):
    """
    Calculate the final shade array.

    Parameters
    ----------
    cell_area_overlap : numpy.array
        Cell area overlap array for each cell in the module.
    CELLAREA : float
        Cell area.
    translucence : float, optional
        Opacity of the shade scenario. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse ratio. The default is 1.
    inp_shade : numpy.array, optional
        Shade array. The default is [].

    Returns
    -------
    op_shade_array : numpy.array
        Shade array.

    """
    shading_array = dir_diff_ratio*translucence*cell_area_overlap/(CELLAREA)
    if not inp_shade:
        inp_shade = shading_array.copy()
    op_shade_array = np.maximum(inp_shade, shading_array)
    op_shade_array[op_shade_array >= 0.9999] = 0.9999
    op_shade_array[op_shade_array <= 0.0001] = 0
    return op_shade_array


def shade_1cell(maxsys_dict, df_shd_sce,
                shd_cell_prop=np.arange(0.2, 1.1, 0.2),
                shd_cell_idx=[], shd_dir='Y',
                translucence=1,
                dir_diff_ratio=1):
    """
    Generate the single cell shading scenario.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    shd_cell_prop : numpy.ndarray, optional
        Array of cell proportions to shade by.
        The default is np.arange(0.2, 1.1, 0.2).
    shd_cell_idx : int or empty list, optional
        Index of cell to shade. If empty, a random cell is shaded.
        The default is [].
    shd_dir : str, optional
        Direction of shading. The default is 'Y'.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    if shd_cell_idx == '[]':
        shd_cell_idx = randrange(
            0, maxsys_dict['Physical_Info']['Index_Map'].max() + 1, 1)
    # Find idx of shaded cell
    shd_row, shd_col = np.where(
        maxsys_dict['Physical_Info']['Index_Map'] == shd_cell_idx)
    # Extract cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len = cell_coords[shd_row[0], shd_col[0], 3,
                           0] - cell_coords[shd_row[0], shd_col[0], 0, 0]
    cell_wid = cell_coords[shd_row[0], shd_col[0], 1,
                           1] - cell_coords[shd_row[0], shd_col[0], 0, 1]
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    for sh_prp in shd_cell_prop:
        # Calculate shade dimension
        sh_prp = round(sh_prp, 6)
        if shd_dir == 'Y':
            shd_wid = cell_wid * sh_prp
            shd_len = cell_len
            shd_cp = [cell_coords[shd_row[0], shd_col[0], 0, 0] + 0.5*shd_len,
                      cell_coords[shd_row[0], shd_col[0], 0, 1] + 0.5*shd_wid
                      ]
        elif shd_dir == 'X':
            shd_wid = cell_wid
            shd_len = cell_len * sh_prp
            shd_cp = [cell_coords[shd_row[0], shd_col[0], 0, 0] + 0.5*shd_len,
                      cell_coords[shd_row[0], shd_col[0], 0, 1] + 0.5*shd_wid
                      ]
        elif shd_dir == 'MidY':
            shd_wid = cell_wid * sh_prp
            shd_len = cell_len
            shd_cp = [cell_coords[shd_row[0], shd_col[0], 0, 0] + 0.5*cell_len,
                      cell_coords[shd_row[0], shd_col[0], 0, 1] + 0.5*cell_wid
                      ]
        elif shd_dir == 'MidX':
            shd_wid = cell_wid
            shd_len = cell_len * sh_prp
            shd_cp = [cell_coords[shd_row[0], shd_col[0], 0, 0] + 0.5*cell_len,
                      cell_coords[shd_row[0], shd_col[0], 0, 1] + 0.5*cell_wid
                      ]
        else:
            raise ValueError(
                'Incorrect shade direction inputted. Valid inputs: Y, X, MidY, MidX')
        # Create shade polygon (Rectangle)
        shd_poly = create_rectangle(shd_cp, shd_len, shd_wid, 0)
        # Check overlap against module
        mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                          0].intersection(shd_poly)
        mod_overlap_area = mod_overlap.area
        mod_overlap_area_perc = 100*mod_overlap.area / \
            maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
        # Calculate overlap area for each cell
        max_key = max(
            list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
        cell_area_overlap = cell_overlap_area(
            maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
            mod_overlap)
        # Calculate shading matrix
        op_shade_array = calc_cell_shading(
            cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
        # Add to shade dataframe
        col_list = list(df_shd_sce.columns.values)
        df_new_row = pd.DataFrame(data=[['Standard',
                                         'One Cell ' + shd_dir,
                                         'Cell Proportion ' + str(sh_prp),
                                         op_shade_array,
                                         shd_poly,
                                         mod_overlap_area,
                                         mod_overlap_area_perc
                                         ]], columns=col_list)
        df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def gen_cells_shd_list(maxsys_dict, num_cells):
    """
    Generate random cell indices but with some order.

    Each diode section --> each parallel substring.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    num_cells : int
        Number of cells to shade.

    Returns
    -------
    shd_cell_list : list
        List of cell indices to shade.

    """
    cell_pos = maxsys_dict['Electrical_Circuit']['Cell_Postion']
    num_diodes = len(cell_pos)
    num_par = len(cell_pos[0])
    tot_1cell = num_diodes * num_par
    diode_list = list(range(num_diodes))
    par_list = list(range(num_par))
    diode_var1 = list(itertools.repeat(diode_list, num_par))
    diode_var = []
    for dv1 in diode_var1:
        for dv2 in dv1:
            diode_var.append(dv2)
    par_var1 = []
    rev_p = False
    for idx_p in range(num_diodes):
        if rev_p:
            par_var1.append(list(reversed(par_list)))
        else:
            par_var1.append(par_list)
        rev_p = not rev_p
    par_var = []
    for dv1 in par_var1:
        for dv2 in dv1:
            par_var.append(dv2)
    # Generate the cell indices to shade
    shd_cell_list = []
    num_iter, rem = divmod(num_cells, tot_1cell)
    if num_iter <= 0:
        diode_idx = diode_var
        par_idx = par_var
    else:
        diode_idx = diode_var * (num_iter+1)
        par_idx = par_var * (num_iter+1)
    diode_idx = diode_idx[:num_cells]
    par_idx = par_idx[:num_cells]
    for idx_num in range(num_cells):
        idx_list = gen_idx_list(cell_pos, diode_idx[idx_num], par_idx[idx_num])
        idx_list = remove_old_cells(shd_cell_list, idx_list)
        idx_cell = random.choice(idx_list)
        shd_cell_list.append(idx_cell)
    return shd_cell_list


def gen_idx_list(cell_pos, diode_idx, par_idx):
    """
    Generate list of cell idxs for shading based on diode and parallel substr.

    Parameters
    ----------
    cell_pos : dict
        Cell position from PVMismatch.
    diode_idx : int
        Diode subsection index.
    par_idx : int
        Parallel substring index.

    Returns
    -------
    idx_list : List
        List of cell indices for shading.

    """
    idx_list = []
    for idx_k in range(len(cell_pos[diode_idx][par_idx])):
        idx_list.append(cell_pos[diode_idx][par_idx][idx_k]['idx'])
    return idx_list


def remove_old_cells(shd_cell_list, idx_list):
    """
    Remove cells that have already been added previously.

    Parameters
    ----------
    shd_cell_list : list
        List of cell indices to shade.
    idx_list : List
        List of cell indices for shading.

    Returns
    -------
    idx_list : List
        List of cell indices for shading.

    """
    if shd_cell_list:
        for shd_cell in shd_cell_list:
            try:
                idx_list.remove(shd_cell)
            except ValueError:
                continue
    return idx_list


def shade_ncells(maxsys_dict, df_shd_sce, num_cells=2, shd_cell_idx_lst=[],
                 translucence=1,
                 dir_diff_ratio=1):
    """
    Generate full shading of n number of cells.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    num_cells : int, optional
        Number of cells to shade. The default is 2.
    shd_cell_idx_lst : list, optional
        List of cell indices to shade. The default is [].
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Extract cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    full_poly = []
    shd_cell_list = gen_cells_shd_list(maxsys_dict, num_cells)
    for idx_cell in range(num_cells):
        if shd_cell_idx_lst == []:
            # Random cell index
            # shd_cell_idx = randrange(
            #     0, maxsys_dict['Physical_Info']['Index_Map'].max() + 1, 1)
            shd_cell_idx = shd_cell_list[idx_cell]
        else:
            shd_cell_idx = shd_cell_idx_lst[idx_cell]
        # Find idx of shaded cell
        shd_row, shd_col = np.where(
            maxsys_dict['Physical_Info']['Index_Map'] == shd_cell_idx)
        # Cell dimensions
        cell_len = cell_coords[shd_row[0], shd_col[0], 3,
                               0] - cell_coords[shd_row[0], shd_col[0], 0, 0]
        cell_wid = cell_coords[shd_row[0], shd_col[0], 1,
                               1] - cell_coords[shd_row[0], shd_col[0], 0, 1]
        CELLAREA = cell_len * cell_wid
        cell_cp = [cell_coords[shd_row[0], shd_col[0], 0, 0] + 0.5*cell_len,
                   cell_coords[shd_row[0], shd_col[0], 0, 1] + 0.5*cell_wid
                   ]
        # Create shade polygon (Rectangle)
        shd_poly = create_rectangle(cell_cp, cell_len, cell_wid, 0)
        full_poly.append(shd_poly)
    full_poly = unary_union(full_poly)
    # Check overlap against module
    mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                      0].intersection(full_poly)
    mod_overlap_area = mod_overlap.area
    mod_overlap_area_perc = 100*mod_overlap.area / \
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
    # Calculate overlap area for each cell
    max_key = max(list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
    cell_area_overlap = cell_overlap_area(
        maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0], mod_overlap)
    # Calculate shading matrix
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    op_shade_array = calc_cell_shading(
        cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
    # Add to shade dataframe
    col_list = list(df_shd_sce.columns.values)
    df_new_row = pd.DataFrame(data=[['Standard',
                                     'N Cells',
                                     'Num Cells ' + str(num_cells),
                                     op_shade_array,
                                     full_poly,
                                     mod_overlap_area,
                                     mod_overlap_area_perc
                                     ]], columns=col_list)
    df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def shade_substr_per_diode(maxsys_dict, df_shd_sce, substr_diodes,
                           mod_prop_vec=[10, 90, 10], translucence=1,
                           dir_diff_ratio=1):
    """
    Generate proportional shading of a module substring.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    substr_diodes : list
        List specifying which parallel substrings and diode sections to shade.
    mod_prop_vec : list, optional
        List containing the proportions by which to shade the substring.
        The default is [10, 90, 10].
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    idx_map = maxsys_dict['Physical_Info']['Index_Map']
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    mod_prop_vec = 0.01 * np.arange(mod_prop_vec[0],
                                    mod_prop_vec[1]+mod_prop_vec[2],
                                    mod_prop_vec[2])
    # Get cell position
    cell_pos = maxsys_dict['Electrical_Circuit']['Cell_Postion']
    num_str = len(cell_pos[0])
    # Run through each substring and diode section for shading
    for mod_prop in mod_prop_vec:
        mod_prop = round(mod_prop, 3)
        full_poly = []
        for sim_num in substr_diodes:
            diode_num = sim_num[0]
            substr_nums = sim_num[1]
            if substr_nums == -1:
                substr_nums = list(range(num_str))
            else:
                substr_nums = [substr_nums]
            for substr_num in substr_nums:
                cells = cell_pos[diode_num][substr_num]
                cell_idxs = []
                for cell in cells:
                    cell_idxs.append(cell['idx'])
                cell_idxs = np.array(cell_idxs)
                cells_mask = np.isin(idx_map, cell_idxs)
                # Length and width of the substring
                BL_X = cell_coords[cells_mask, 0, 0].min()
                TR_Y = cell_coords[cells_mask, 2, 1].max()
                BR_X = cell_coords[cells_mask, 3, 0].min()
                BR_Y = cell_coords[cells_mask, 3, 1].min()
                SX = BR_X - BL_X
                SY = TR_Y - BR_Y
                # Shade length & width
                shd_x = SX
                shd_y = mod_prop * SY
                cen_pt = [BL_X + 0.5*shd_x, BR_Y + 0.5*shd_y]
                # Create shade polygon (Rectangle)
                shd_poly = create_rectangle(cen_pt, shd_x, shd_y, 0)
                full_poly.append(shd_poly)
        full_poly = unary_union(full_poly)
        # Check overlap against module
        mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                          0].intersection(full_poly)
        mod_overlap_area = mod_overlap.area
        mod_overlap_area_perc = 100*mod_overlap_area / \
            maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
        # Calculate overlap area for each cell
        cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
        max_key = max(
            list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
        cell_area_overlap = cell_overlap_area(maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
                                              mod_overlap)
        # Calculate shading matrix
        # Cell dimensions
        CELLAREA = cell_len_vec*cell_wid_vec
        op_shade_array = calc_cell_shading(cell_area_overlap, CELLAREA,
                                           translucence, dir_diff_ratio)
        col_list = list(df_shd_sce.columns.values)
        shd_scn = 'Shade Substring'
        shd_var = 'Substring Proportion ' + str(mod_prop)
        df_new_row = pd.DataFrame(data=[['Standard',
                                         shd_scn,
                                         shd_var,
                                         op_shade_array,
                                         full_poly,
                                         mod_overlap_area,
                                         mod_overlap_area_perc
                                         ]], columns=col_list)
        df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def shade_leaves_birddroppings(maxsys_dict, df_shd_sce, num_obj=100,
                               maj_dia=[5, 50], min_dia=[5, 50],
                               rot_ang=[-90, 90], translucence=1,
                               dir_diff_ratio=1, use_std=True,
                               pickle_fn='Leaves_BirdDroppings_Data.pickle'):
    """
    Generate the leaves and bird droppings shade scenario.

    This is achieved by randomly placing N ellipses with varying angles inside
    the module area.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    num_obj : int, optional
        Number of ellipses to place inside module area. The default is 100.
    maj_dia : list, optional
        Range of the major diameter of ellipses. The default is [5, 50].
    min_dia : list, optional
        Range of the minor diameter of the ellipses. The default is [5, 50].
    rot_ang : list, optional
        Range of the rotation angle of the ellipses. The default is [-90, 90].
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.
    use_std : bool, optional
        If True, use a standard scenario and scale based on Module dimensions.
        The default is True.
    pickle_fn : str, optional
        Name of the standard scenario pickle file.
        The default is 'Leaves_BirdDroppings_Data.pickle'.

    Returns
    -------
    df_shd_sce : TYPE
        DESCRIPTION.

    """
    if use_std:
        pickle_path = pickle_fn
        # Check if pickle file with standard random data is available
        try:
            leaf_bird_data = load_pickle(pickle_path)
        except FileNotFoundError:
            # If not, Generate random data
            cps, xdiams, ydiams, rotangs = gen_rand_coord_data(
                maxsys_dict, num_obj, maj_dia, min_dia, rot_ang)
            MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                    2, 0]
            MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                    2, 1]
            leaf_bird_data = [cps, np.asarray(xdiams), np.asarray(
                ydiams), np.asarray(rotangs), MX, MY]
            save_pickle(pickle_path, leaf_bird_data)
        # Extract standard module data
        std_cps = leaf_bird_data[0]
        std_xdiams = leaf_bird_data[1]
        std_ydiams = leaf_bird_data[2]
        std_rotangs = leaf_bird_data[3]
        std_MX = leaf_bird_data[4]
        std_MY = leaf_bird_data[5]
        # Re-generate data based on Pickle Module proportions
        std_cps_props_x = np.asarray([pts.x for pts in std_cps]) / std_MX
        std_cps_props_y = np.asarray([pts.y for pts in std_cps]) / std_MY
        std_xdiams_props = std_xdiams / std_MX
        std_ydiams_props = std_ydiams / std_MY
        # Generate data for inputted Module
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        cps_x = std_cps_props_x * MX
        cps_y = std_cps_props_y * MY
        xdiams = std_xdiams_props * MX
        ydiams = std_ydiams_props * MY
        rotangs = std_rotangs
    else:
        cps, xdiams, ydiams, rotangs = gen_rand_coord_data(
            maxsys_dict, num_obj, maj_dia, min_dia, rot_ang)
        cps_x = np.asarray([pts.x for pts in cps])
        cps_y = np.asarray([pts.y for pts in cps])
    # Run for loop to generate shade data
    full_poly = []
    for idx_obj in range(num_obj):
        x_diam = xdiams[idx_obj]
        y_diam = ydiams[idx_obj]
        ra = rotangs[idx_obj]
        cp = [cps_x[idx_obj], cps_y[idx_obj]]
        # Create shade polygon (Ellipse)
        shd_poly = create_ellipse(cp, x_diam, y_diam, ra)
        full_poly.append(shd_poly)
    full_poly = unary_union(full_poly)
    # Check overlap against module
    mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                      0].intersection(full_poly)
    mod_overlap_area = mod_overlap.area
    mod_overlap_area_perc = 100*mod_overlap.area / \
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
    # Calculate overlap area for each cell
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    max_key = max(list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
    cell_area_overlap = cell_overlap_area(
        maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0], mod_overlap)
    # Calculate shading matrix
    # Cell dimensions
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    op_shade_array = calc_cell_shading(
        cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
    # Add to shade dataframe
    col_list = list(df_shd_sce.columns.values)
    df_new_row = pd.DataFrame(data=[['Standard',
                                     'Leaves Bird Droppings',
                                     'Num Objects ' + str(num_obj),
                                     op_shade_array,
                                     full_poly,
                                     mod_overlap_area,
                                     mod_overlap_area_perc
                                     ]], columns=col_list)
    df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def gen_rand_coord_data(maxsys_dict, num_obj, maj_dia, min_dia, rot_ang):
    """
    Generate random dimensions for N number of ellipses.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    num_obj : int, optional
        Number of ellipses to place inside module area. The default is 100.
    maj_dia : list, optional
        Range of the major diameter of ellipses. The default is [5, 50].
    min_dia : list, optional
        Range of the minor diameter of the ellipses. The default is [5, 50].
    rot_ang : list, optional
        Range of the rotation angle of the ellipses. The default is [-90, 90].

    Returns
    -------
    cps : list
        List of random center points within the module area.
    xdiams : list
        List of random ellipse major diameters.
    ydiams : list
        List of random ellipse mino diameters.
    rotangs : list
        List of random ellipse rotation angles.

    """
    cps = Random_Points_in_Polygon(
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0], num_obj)
    xdiams = []
    ydiams = []
    rotangs = []
    for idx_cp in range(num_obj):
        # Random major diameter
        x_diam = uniform(maj_dia[0], maj_dia[1] + 1)
        xdiams.append(x_diam)
        # Random minor diameter
        y_diam = uniform(min_dia[0], min_dia[1] + 1)
        ydiams.append(y_diam)
        # Random rotation angle
        ra = uniform(rot_ang[0], rot_ang[1] + 1)
        rotangs.append(ra)
    return cps, xdiams, ydiams, rotangs


def Random_Points_in_Polygon(polygon, number):
    """
    Generate N number of points within a polygon randomly.

    Parameters
    ----------
    polygon : shapely polygon
        Shapely polygon (PV Module in this application).
    number : int
        Number of points to generate.

    Returns
    -------
    points : list
        N number of shapely points.

    """
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(uniform(minx, maxx), uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def create_ellipse(center_pt, x_diam, y_diam, rot_ang):
    """
    Create ellipse given the dimensions.

    Parameters
    ----------
    center_pt : float
        Center point of ellipse.
    x_diam : float
        Major diameter of ellipse.
    y_diam : float
        Minor diameter of ellipse.
    rot_ang : float
        Rotation angle of ellipse.

    Returns
    -------
    ellipse : shapely polygon
        Shapely polygon shaped like an ellipse.

    """
    # Let create a circle of radius 1 around center point:
    circ = Point(center_pt).buffer(1)
    # Let create the ellipse along x and y:
    ellipse = scale(circ, x_diam, y_diam)
    # Let rotate the ellipse (clockwise, x axis pointing right):
    ellipse = rotate(ellipse, rot_ang)
    return ellipse


def shade_row_col(maxsys_dict, df_shd_sce, shd_type='row',
                  rowcol_type='mod_prop', mod_prop_vec=[10, 90, 10],
                  translucence=1, dir_diff_ratio=1, cen_pt_d='forward'):
    """
    Generate short (row for portrait) or long edge (col for portrait) shading.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    shd_type : str, optional
        Specify whether it is row or columnwise shading. The default is 'row'.
    rowcol_type : str, optional
        Specify whether the shading needs to be done cellwise, proportionally
        or by actual dimensions.
        The default is 'mod_prop'.
    mod_prop_vec : list, optional
        List containing the proportions or actual dimensions by which to shade
        the module. The default is [10, 90, 10].
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.
    cen_pt_d : str, optional
        Specify the direction of shading. 'forward' or 'reverse'.
        'forward' is bottom to top for row, and left to right for col.
        The default is 'forward'.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len = cell_coords[0, 0, 3, 0] - cell_coords[0, 0, 0, 0]
    cell_width = cell_coords[0, 0, 1, 1] - cell_coords[0, 0, 0, 1]
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    if rowcol_type == 'cells':
        # Get number of rows
        cells_shape = cell_coords.shape
        num_rows = cells_shape[0]
        for idx_row in range(num_rows-1):
            # Define center point
            if shd_type == 'row':
                cen_pt = [
                    0.5 *
                    maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                       2, 0],
                    0.5*cell_width*(idx_row+1)
                ]
            elif shd_type == 'col':
                cen_pt = [
                    0.5*cell_len*(idx_row+1),
                    0.5 *
                    maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                       2, 1]
                ]
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            # Create shade polygon (Rectangle)
            if shd_type == 'row':
                full_poly = create_rectangle(
                    cen_pt, maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0], cell_width*(idx_row+1), 0)
            elif shd_type == 'col':
                full_poly = create_rectangle(
                    cen_pt, cell_len*(idx_row+1),
                    maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                       2, 1],
                    0)
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            # Check overlap against module
            mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                              0].intersection(full_poly)
            mod_overlap_area = mod_overlap.area
            mod_overlap_area_perc = 100*mod_overlap.area / \
                maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
            # Calculate overlap area for each cell
            max_key = max(
                list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
            cell_area_overlap = cell_overlap_area(
                maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
                mod_overlap)
            # Calculate shading matrix
            op_shade_array = calc_cell_shading(
                cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
            # Add to shade dataframe
            shd_var = 'Num Cells' + str(idx_row + 1)
            if shd_type == 'row':
                shd_scn = 'Shade Row'
            elif shd_type == 'col':
                shd_scn = 'Shade Column'
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            col_list = list(df_shd_sce.columns.values)
            df_new_row = pd.DataFrame(data=[['Standard',
                                             shd_scn,
                                             shd_var,
                                             op_shade_array,
                                             full_poly,
                                             mod_overlap_area,
                                             mod_overlap_area_perc
                                             ]], columns=col_list)
            df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
    elif rowcol_type == 'mod_prop':
        # Module dimensions
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        mod_prop_vec = 0.01 * \
            np.arange(mod_prop_vec[0], mod_prop_vec[1]+mod_prop_vec[2],
                      mod_prop_vec[2])
        for mod_prop in mod_prop_vec:
            mod_prop = round(mod_prop, 3)
            # Define shade coordinates
            if shd_type == 'row':
                shd_x = MX
                shd_y = mod_prop * MY
            elif shd_type == 'col':
                shd_x = mod_prop * MX
                shd_y = MY
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            if cen_pt_d == 'reverse':
                if shd_type == 'row':
                    cen_pt = [0.5*shd_x, MY - 0.5*shd_y]
                elif shd_type == 'col':
                    cen_pt = [MX - 0.5*shd_x, 0.5*shd_y]
                else:
                    raise ValueError(
                        'Incorrect shade type. Valid values: row or col.')
            else:
                cen_pt = [0.5*shd_x, 0.5*shd_y]
            # Create shade polygon (Rectangle)
            full_poly = create_rectangle(cen_pt, shd_x, shd_y, 0)
            # Check overlap against module
            mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                              0].intersection(full_poly)
            mod_overlap_area = mod_overlap.area
            mod_overlap_area_perc = 100*mod_overlap.area / \
                maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
            # Calculate overlap area for each cell
            max_key = max(
                list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
            cell_area_overlap = cell_overlap_area(
                maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
                mod_overlap)
            # Calculate shading matrix
            op_shade_array = calc_cell_shading(
                cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
            # Add to shade dataframe
            shd_var = 'Module Proportion ' + str(mod_prop)
            if shd_type == 'row':
                shd_scn = 'Shade Row'
            elif shd_type == 'col':
                shd_scn = 'Shade Column'
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            if cen_pt_d == 'reverse':
                shd_scn += ' reverse'
            col_list = list(df_shd_sce.columns.values)
            df_new_row = pd.DataFrame(data=[['Standard',
                                             shd_scn,
                                             shd_var,
                                             op_shade_array,
                                             full_poly,
                                             mod_overlap_area,
                                             mod_overlap_area_perc
                                             ]], columns=col_list)
            df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
    elif rowcol_type == 'actual':
        # Module dimensions
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        mod_prop_vec = np.arange(mod_prop_vec[0], mod_prop_vec[1]+1,
                                 mod_prop_vec[2])
        for mod_prop in mod_prop_vec:
            mod_prop = round(mod_prop, 3)
            # Define shade coordinates
            if shd_type == 'row':
                shd_x = MX
                shd_y = mod_prop
            elif shd_type == 'col':
                shd_x = mod_prop
                shd_y = MY
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            cen_pt = [0.5*shd_x, 0.5*shd_y]
            # Create shade polygon (Rectangle)
            full_poly = create_rectangle(cen_pt, shd_x, shd_y, 0)
            # Check overlap against module
            mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                              0].intersection(full_poly)
            mod_overlap_area = mod_overlap.area
            mod_overlap_area_perc = 100*mod_overlap.area / \
                maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
            # Calculate overlap area for each cell
            max_key = max(
                list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
            cell_area_overlap = cell_overlap_area(
                maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
                mod_overlap)
            # Calculate shading matrix
            op_shade_array = calc_cell_shading(
                cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
            # Add to shade dataframe
            shd_var = 'Width ' + str(mod_prop)
            if shd_type == 'row':
                shd_scn = 'Shade Row'
            elif shd_type == 'col':
                shd_scn = 'Shade Column'
            else:
                raise ValueError(
                    'Incorrect shade type. Valid values: row or col.')
            col_list = list(df_shd_sce.columns.values)
            df_new_row = pd.DataFrame(data=[['Standard',
                                             shd_scn,
                                             shd_var,
                                             op_shade_array,
                                             full_poly,
                                             mod_overlap_area,
                                             mod_overlap_area_perc
                                             ]], columns=col_list)
            df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
    else:
        print('Incorrect row shade type inputted. Possible inputs for rowcol_type: mod_prop or cells.')
    return df_shd_sce


def bottom_edge_soiling(maxsys_dict, df_shd_sce, bes_shape='rectangle',
                        mod_prop_vec=[1, 10, 1], translucence=1,
                        dir_diff_ratio=1, shd_type='row',
                        rowcol_type='mod_prop'):
    """
    Generate bottom edge soiling scenario.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    bes_shape : str, optional
        Shape of the shading. Options are 'rectangle', 'single_triangle',
        'double_triangle'. The default is 'rectangle'.
    mod_prop_vec : list, optional
        List containing the proportions or actual dimensions by which to shade
        the module. The default is [1, 10, 1].
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.
    shd_type : str, optional
        Specify whether it is row or columnwise shading. The default is 'row'.
    rowcol_type : str, optional
        Specify whether the shading needs to be done cellwise, proportionally
        or by actual dimensions.
        The default is 'mod_prop'.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    # Module dimensions
    MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
    MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
    if rowcol_type == 'mod_prop':
        mod_prop_vec = 0.01 * \
            np.arange(mod_prop_vec[0], mod_prop_vec[1]+1, mod_prop_vec[2])
    elif rowcol_type == 'actual':
        mod_prop_vec = np.arange(
            mod_prop_vec[0], mod_prop_vec[1]+1, mod_prop_vec[2])
    else:
        raise ValueError(
            'Incorrect rowcol type. Valid values: mod_prop or actual.')
    for mod_prop in mod_prop_vec:
        mod_prop = round(mod_prop, 2)
        if bes_shape == 'rectangle':
            # Define shade coordinates
            if shd_type == 'row':
                shd_x = MX
                if rowcol_type == 'mod_prop':
                    shd_y = mod_prop * MY
                elif rowcol_type == 'actual':
                    shd_y = mod_prop
            elif shd_type == 'col':
                if rowcol_type == 'mod_prop':
                    shd_x = mod_prop * MX
                elif rowcol_type == 'actual':
                    shd_x = mod_prop
                shd_y = MY
            cen_pt = [0.5*shd_x, 0.5*shd_y]
            if rowcol_type == 'mod_prop':
                shd_var = 'Rectangle Mod. Prop. ' + str(mod_prop)
            elif rowcol_type == 'actual':
                shd_var = 'Rectangle ' + str(mod_prop) + ' mm'
            # Create shade polygon (Rectangle)
            full_poly = create_rectangle(cen_pt, shd_x, shd_y, 0)
        elif bes_shape == 'single_triangle':
            # Define shade coordinates
            if shd_type == 'row':
                shd_x = MX
                if rowcol_type == 'mod_prop':
                    shd_y = mod_prop * MY
                elif rowcol_type == 'actual':
                    shd_y = mod_prop
            elif shd_type == 'col':
                if rowcol_type == 'mod_prop':
                    shd_x = mod_prop * MX
                elif rowcol_type == 'actual':
                    shd_x = mod_prop
                shd_y = MY
            cen_pt = [0*shd_x, 0*shd_y]
            if rowcol_type == 'mod_prop':
                shd_var = 'Single Triangle Mod. Prop. ' + str(mod_prop)
            elif rowcol_type == 'actual':
                shd_var = 'Single Triangle ' + str(mod_prop) + ' mm'
            # Create shade polygon (Rectangle)
            full_poly = create_rtang_triangle(cen_pt, shd_x, shd_y, 0)
        elif bes_shape == 'double_triangle':
            # Define shade coordinates
            if shd_type == 'row':
                shd_x = MX*0.5
                if rowcol_type == 'mod_prop':
                    shd_y = mod_prop * MY
                elif rowcol_type == 'actual':
                    shd_y = mod_prop
                cen_pt2 = [MX, 0]
                full_poly2 = create_rtang_triangle(cen_pt2, -shd_x, shd_y, 0)
            elif shd_type == 'col':
                if rowcol_type == 'mod_prop':
                    shd_x = mod_prop * MX
                elif rowcol_type == 'actual':
                    shd_x = mod_prop
                shd_y = MY*0.5
                cen_pt2 = [0, MY]
                full_poly2 = create_rtang_triangle(cen_pt2, shd_x, -shd_y, 0)
            cen_pt1 = [0*shd_x, 0*shd_y]
            if rowcol_type == 'mod_prop':
                shd_var = 'Double Triangle Mod. Prop. ' + str(mod_prop)
            elif rowcol_type == 'actual':
                shd_var = 'Double Triangle ' + str(mod_prop) + ' mm'
            # Create shade polygon (Rectangle)
            full_poly1 = create_rtang_triangle(cen_pt1, shd_x, shd_y, 0)
            full_poly = unary_union([full_poly1, full_poly2])
        else:
            raise ValueError(
                'Incorrect shape. Inputs for bes_shape: rectangle, single_triangle, or double_triangle')
        # Check overlap against module
        mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                          0].intersection(full_poly)
        mod_overlap_area = mod_overlap.area
        mod_overlap_area_perc = 100*mod_overlap.area / \
            maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
        # Calculate overlap area for each cell
        max_key = max(
            list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
        cell_area_overlap = cell_overlap_area(
            maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
            mod_overlap)
        # Calculate shading matrix
        op_shade_array = calc_cell_shading(
            cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
        # Add to shade dataframe
        shade_type = 'Bottom Edge Soiling ' + bes_shape + ' ' + shd_type
        col_list = list(df_shd_sce.columns.values)
        df_new_row = pd.DataFrame(data=[['Standard',
                                         shade_type,
                                         shd_var,
                                         op_shade_array,
                                         full_poly,
                                         mod_overlap_area,
                                         mod_overlap_area_perc
                                         ]], columns=col_list)
        df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def create_rtang_triangle(bh_pt, xlen, ylen, rot_ang):
    """
    Generate a right angled triangle shapely polygon.

    Parameters
    ----------
    bh_pt : float
        Right angle point coordinate.
    xlen : float
        Length of x.
    ylen : float
        Length of y.
    rot_ang : float
        Rotation angle for the triangle.

    Returns
    -------
    trg_shply : shapely polygon
        Right angled triangle polygon.

    """
    # Convert base-height point to numpy array
    center_pt = np.array([[bh_pt[0]], [bh_pt[1]]])
    # Define Rotation Matrix
    rot_mat = np.array([
                       [np.cos(np.deg2rad(rot_ang)), -
                        np.sin(np.deg2rad(rot_ang))],
                       [np.sin(np.deg2rad(rot_ang)),
                        np.cos(np.deg2rad(rot_ang))]
                       ])
    # Define vertices of rectangle in
    # rotated dimensions (x', y' coordinate axis)
    trg_rot_vert = np.array([[0, 0, xlen],
                             [0, ylen, 0]
                             ])
    # Calculate vertices in unrotated dimensions (x,y coordinate axis)
    trg_vert = np.round(np.tile(center_pt, 3) +
                        np.matmul(rot_mat, trg_rot_vert), decimals=2)
    trg_shply = Polygon([(trg_vert[0, 0], trg_vert[1, 0]),
                         (trg_vert[0, 1], trg_vert[1, 1]),
                         (trg_vert[0, 2], trg_vert[1, 2])])
    return trg_shply


def shade_rot_rectangle(maxsys_dict, df_shd_sce,
                        rot_ang_vec=np.arange(25, 81, 25),
                        sh_width_vec=np.arange(25, 115, 35),
                        translucence=1, dir_diff_ratio=1, use_std=True,
                        pickle_fn='Wire_Pole_Data.pickle',
                        stype='Wire Pole', cen_pt='middle'):
    """
    Generate Wire, Pole, or Chimney shading.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    rot_ang_vec : numpy.array, optional
        Rotation angle of the shading rectangle.
        The default is np.arange(25, 81, 25).
    sh_width_vec : numpy.array, optional
        Width of the shading rectangle. The default is np.arange(25, 115, 35).
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.
    use_std : bool, optional
        If True, use a standard scenario and scale based on Module dimensions.
        The default is True.
    pickle_fn : str, optional
        Name of the standard scenario pickle file.
        The default is 'Wire_Pole_Data.pickle'.
    stype : str, optional
        Shading type. The default is 'Wire Pole'.
    cen_pt : str, optional
        Which location of the module to shade from.
        Options are middle, BR, BL, TR, TL, diagonal.
        The default is 'middle'.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    sh_width_vec : numpy.array, optional
        Width of the shading rectangle. The default is np.arange(25, 115, 35).

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
    MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
    if cen_pt == 'middle':
        cen_pt = [
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.5*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'BR':
        cen_pt = [
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0
        ]
    elif cen_pt == 'BL':
        cen_pt = [
            0,
            0
        ]
    elif cen_pt == 'TR':
        cen_pt = [
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'TL':
        cen_pt = [
            0,
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'diagonal':
        ra = np.arctan(MY / MX)
        rotang = np.degrees(ra)
        act_ra = np.radians(180 - 90 - rotang)
        rot_ang_vec = rotang * np.ones(rot_ang_vec.shape)
        del_CX = 0.5 * sh_width_vec * np.cos(act_ra)
        del_CY = 0.5 * sh_width_vec * np.sin(act_ra)
        cen_pt = [
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1,
                                                               0, 2, 0] - del_CX[0],
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1,
                                                               0, 2, 1] + del_CY[0]
        ]
    elif isinstance(cen_pt, str):
        raise ValueError(
            'Incorrect center point. Pptions are middle, BL, BR, TL, TR, or list of corrdinates, eg. cen_pt = [x, y].')
    if use_std:
        pickle_path = pickle_fn
        # Check if pickle file with standard random data is available
        try:
            wire_pole_data = load_pickle(pickle_path)
        except FileNotFoundError:
            MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                    2, 0]
            MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                    2, 1]
            poly_coords = []
            for idx_wid, sh_width in enumerate(sh_width_vec):
                sh_width = round(sh_width, 2)
                for rot_ang in rot_ang_vec:
                    # Create shade polygon (Rectangle)
                    full_poly = create_rectangle(cen_pt,
                                                 4*MX, sh_width,
                                                 rot_ang)
                    poly_coords.append(list(full_poly.exterior.coords))
            wire_pole_data = [rot_ang_vec, sh_width_vec, MX, MY, poly_coords]
            save_pickle(pickle_path, wire_pole_data)
        # Extract standard module data
        std_rotangs = wire_pole_data[0]
        std_shwid = wire_pole_data[1]
        std_MX = wire_pole_data[2]
        std_MY = wire_pole_data[3]
        std_poly_coords = wire_pole_data[4]
        # Generate data for inputted Module
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        sh_width_vec = std_shwid
        rot_ang_vec = std_rotangs
    else:
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        std_MX = MX
        std_MY = MY
        poly_coords = []
        for idx_wid, sh_width in enumerate(sh_width_vec):
            sh_width = round(sh_width, 2)
            for rot_ang in rot_ang_vec:
                # Create shade polygon (Rectangle)
                full_poly = create_rectangle(cen_pt,
                                             4*MX, sh_width,
                                             rot_ang)
                poly_coords.append(list(full_poly.exterior.coords))
        std_poly_coords = poly_coords
    idx_sim = 0
    for idx_wid, sh_width in enumerate(sh_width_vec):
        sh_width = round(sh_width, 2)
        for rot_ang in rot_ang_vec:
            # Create shade polygon (Rectangle)
            shd_coords = std_poly_coords[idx_sim]
            for idx_c, tup_c in enumerate(shd_coords):
                shd_coords[idx_c] = (MX*tup_c[0]/std_MX, MY*tup_c[1]/std_MY)
            full_poly = Polygon(shd_coords)
            # Check overlap against module
            mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                              0].intersection(full_poly)
            mod_overlap_area = mod_overlap.area
            mod_overlap_area_perc = 100*mod_overlap.area / \
                maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
            # Calculate overlap area for each cell
            max_key = max(
                list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
            cell_area_overlap = cell_overlap_area(
                maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
                mod_overlap)
            # Calculate shading matrix
            op_shade_array = calc_cell_shading(
                cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
            # Add to shade dataframe
            shd_var = 'Width ' + str(idx_wid+1) + \
                ' Angle [deg] ' + str(round(rot_ang, 2))
            col_list = list(df_shd_sce.columns.values)
            df_new_row = pd.DataFrame(data=[['Standard',
                                             stype,
                                             shd_var,
                                             op_shade_array,
                                             full_poly,
                                             mod_overlap_area,
                                             mod_overlap_area_perc
                                             ]], columns=col_list)
            df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
            idx_sim += 1
    return df_shd_sce, sh_width_vec


def shade_pipe(maxsys_dict, df_shd_sce, rot_ang_vec=np.arange(15, 46, 15),
               sh_width_vec=np.arange(25, 115, 35),
               sh_len_vec=np.arange(25, 25, 1), translucence=1,
               dir_diff_ratio=1, use_std=True, use_len=False,
               pickle_fn='L1by4_Pipe_Data.pickle', stype='L1by4 Pipe',
               cen_pt='L1by4'):
    """
    Generate pipe shading scenario.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    rot_ang_vec : numpy.array, optional
        Rotation angle of the shading rectangle.
        The default is np.arange(15, 46, 15).
    sh_width_vec : numpy.array, optional
        Width of the shading rectangle. The default is np.arange(25, 115, 35).
    sh_len_vec : numpy.array, optional
        Length of the shading rectangle. The default is np.arange(25, 25, 1).
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.
    use_std : bool, optional
        If True, use a standard scenario and scale based on Module dimensions.
        The default is True.
    use_len : book, optional
        Use actual lengths of the rectangle. The default is False.
    pickle_fn : str, optional
        Name of the standard scenario pickle file.
        The default is 'L1by4_Pipe_Data.pickle'.
    stype : str, optional
        Shading type. The default is 'L1by4 Pipe'.
    cen_pt : str, optional
        Which location of the module to shade from. Options are L1by4, L0, L00,
        L1by2, L3by4, M0, M1by4, M1by2, M3by4,
        R0, R1by4, R1by2, R3by4. The default is 'L1by4'.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    sh_width_vec : numpy.array, optional
        Width of the shading rectangle.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    if cen_pt == 'L1by4':
        cen_pt = [
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'L0':
        cen_pt = [
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'L00':
        cen_pt = [
            0 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'L1by2':
        cen_pt = [
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.5*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'L3by4':
        cen_pt = [
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'M0':
        cen_pt = [
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
        # print(cen_pt)
    elif cen_pt == 'M1by4':
        cen_pt = [
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
        # print(cen_pt)
    elif cen_pt == 'M1by2':
        cen_pt = [
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.5*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'M3by4':
        cen_pt = [
            0.5 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'R0':
        cen_pt = [
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'R1by4':
        cen_pt = [
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.25 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'R1by2':
        cen_pt = [
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.5*maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif cen_pt == 'R3by4':
        cen_pt = [
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0],
            0.75 *
            maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        ]
    elif isinstance(cen_pt, str):
        raise ValueError(
            'Incorrect cen_pt. Options: L1by4, L1by2, L3by4, M1by4, M1by2, M3by4, R1by4, R1by2, R3by4, or list of corrdinates, eg. cen_pt = [x, y].')
    if use_std:
        pickle_path = pickle_fn
        # Check if pickle file with standard random data is available
        try:
            wire_pole_data = load_pickle(pickle_path)
        except FileNotFoundError:
            MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                    2, 0]
            MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0,
                                                                    2, 1]
            poly_coords = []
            for idx_wid, sh_width in enumerate(sh_width_vec):
                sh_width = round(sh_width, 2)
                if use_len:
                    sh_len = 2*round(sh_len_vec[idx_wid], 2)
                else:
                    sh_len = MY
                for rot_ang in rot_ang_vec:
                    # Create shade polygon (Rectangle)
                    if rot_ang == 90:
                        full_poly = create_rectangle(cen_pt,
                                                     sh_len, sh_width,
                                                     rot_ang)
                    else:
                        full_poly = create_rectangle(cen_pt,
                                                     sh_len, sh_width,
                                                     rot_ang)
                    poly_coords.append(list(full_poly.exterior.coords))
            wire_pole_data = [rot_ang_vec, sh_width_vec, MX, MY, poly_coords]
            save_pickle(pickle_path, wire_pole_data)
        # Extract standard module data
        std_rotangs = wire_pole_data[0]
        std_shwid = wire_pole_data[1]
        std_MX = wire_pole_data[2]
        std_MY = wire_pole_data[3]
        std_poly_coords = wire_pole_data[4]
        # Generate data for inputted Module
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        sh_width_vec = std_shwid
        rot_ang_vec = std_rotangs
    else:
        MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
        MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
        std_MX = MX
        std_MY = MY
        poly_coords = []
        for idx_wid, sh_width in enumerate(sh_width_vec):
            sh_width = round(sh_width, 2)
            if use_len:
                sh_len = 2*round(sh_len_vec[idx_wid], 2)
            else:
                sh_len = MY
            for rot_ang in rot_ang_vec:
                # Create shade polygon (Rectangle)
                if rot_ang == 90:
                    full_poly = create_rectangle(cen_pt,
                                                 sh_len, sh_width,
                                                 rot_ang)
                else:
                    full_poly = create_rectangle(cen_pt,
                                                 sh_len, sh_width,
                                                 rot_ang)
                poly_coords.append(list(full_poly.exterior.coords))
        std_poly_coords = poly_coords
    idx_sim = 0
    for idx_wid, sh_width in enumerate(sh_width_vec):
        sh_width = round(sh_width, 2)
        for rot_ang in rot_ang_vec:
            # Create shade polygon (Rectangle)
            shd_coords = std_poly_coords[idx_sim]
            for idx_c, tup_c in enumerate(shd_coords):
                shd_coords[idx_c] = (MX*tup_c[0]/std_MX, MY*tup_c[1]/std_MY)
            full_poly = Polygon(shd_coords)
            # Check overlap against module
            mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                              0].intersection(full_poly)
            mod_overlap_area = mod_overlap.area
            mod_overlap_area_perc = 100*mod_overlap.area / \
                maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
            # Calculate overlap area for each cell
            max_key = max(
                list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
            cell_area_overlap = cell_overlap_area(
                maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
                mod_overlap)
            # Calculate shading matrix
            op_shade_array = calc_cell_shading(
                cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
            # Add to shade dataframe
            shd_var = 'Width ' + str(idx_wid+1) + \
                ' Angle [deg] ' + str(rot_ang)
            col_list = list(df_shd_sce.columns.values)
            df_new_row = pd.DataFrame(data=[['Standard',
                                             stype,
                                             shd_var,
                                             op_shade_array,
                                             full_poly,
                                             mod_overlap_area,
                                             mod_overlap_area_perc
                                             ]], columns=col_list)
            df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
            idx_sim += 1
    return df_shd_sce, sh_width_vec


def shade_mixed1(maxsys_dict, df_shd_sce, translucence=1, dir_diff_ratio=1):
    """
    Generate the first version of a user generated mixed shade case.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    # Module dimensions
    MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
    MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
    # Std. shape dimensions.
    cen_pt_vec = [[0.0625*MX, 0.23*MY],
                  [0.1875*MX, 0.78*MY],
                  [0.4375*MX, 0.2464*MY],
                  [0.5625*MX, 0.32*MY],
                  [0.875*MX, 0],
                  [0.9375*MX, 0.8857*MY],
                  [0.25*MX, 0.5759*MY],
                  [0.4375*MX, 0.5107*MY]
                  ]
    shp_len_vec = [0.125*MX,
                   0.125*MX,
                   0.125*MX,
                   0.125*MX,
                   0.25*MX,
                   0.125*MX,
                   0.2*MX,
                   0.125*MX,
                   ]
    shp_wid_vec = [0.06*MY,
                   0.0643*MY,
                   0.064*MY,
                   0.05*MY,
                   0.0714*MY,
                   0.057*MY,
                   0.1517*MY,
                   0.021*MY,
                   ]
    # Create shade array
    full_poly = []
    for idx_cp, cp in enumerate(cen_pt_vec):
        sl = shp_len_vec[idx_cp]
        sw = shp_wid_vec[idx_cp]
        # Create shade polygon (Rectangle)
        shd_poly = create_rectangle(cp, sl, sw, 0)
        full_poly.append(shd_poly)
    full_poly = unary_union(full_poly)
    # Check overlap against module
    mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                      0].intersection(full_poly)
    mod_overlap_area = mod_overlap.area
    mod_overlap_area_perc = 100*mod_overlap.area / \
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
    # Calculate overlap area for each cell
    max_key = max(list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
    cell_area_overlap = cell_overlap_area(
        maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0], mod_overlap)
    # Calculate shading matrix
    op_shade_array = calc_cell_shading(
        cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
    # Add to shade dataframe
    col_list = list(df_shd_sce.columns.values)
    df_new_row = pd.DataFrame(data=[['Standard',
                                     'Mixed',
                                     'Mixed 1 (Rectangles)',
                                     op_shade_array,
                                     full_poly,
                                     mod_overlap_area,
                                     mod_overlap_area_perc
                                     ]], columns=col_list)
    df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def shade_mixed2(maxsys_dict, df_shd_sce, translucence=1, dir_diff_ratio=1):
    """
    Generate the second version of a user generated mixed shade case.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    # Module dimensions
    MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
    MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
    # Standard shape dimensions.
    cen_pt_vec = [[0, 0],
                  [0.125*MX, 0.7*MY],
                  [0.6*MX, 0.47*MY],
                  [0.6625*MX, 0.8786*MY],
                  [0.89*MX, 0.25*MY],
                  ]
    shp_x_vec = [0.5625*MX,
                 0.025*MX,
                 0.0625*MX,
                 0.06*MX,
                 0.125*MX,
                 ]
    shp_y_vec = [0.3071*MY,
                 0.025*MY,
                 0.07145*MY,
                 0.06*MY,
                 0.1*MY,
                 ]
    rot_vec = [0, 0, 45, 0, -45]
    shp_vec = ['triang', 'ell', 'ell', 'ell', 'ell']
    full_poly = []
    for idx_cp, cp in enumerate(cen_pt_vec):
        sx = shp_x_vec[idx_cp]
        sy = shp_y_vec[idx_cp]
        shp = shp_vec[idx_cp]
        rot_ang = rot_vec[idx_cp]
        # Create shade polygon (Triangle or Ellipse)
        if shp == 'triang':
            shd_poly = create_rtang_triangle(cp, sx, sy, rot_ang)
        elif shp == 'ell':
            shd_poly = create_ellipse(cp, sx, sy, rot_ang)
        else:
            raise ValueError('Wrong shape inputted! Options are triang or ell')
        full_poly.append(shd_poly)
    full_poly = unary_union(full_poly)
    # Check overlap against module
    mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                      0].intersection(full_poly)
    mod_overlap_area = mod_overlap.area
    mod_overlap_area_perc = 100*mod_overlap.area / \
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
    # Calculate overlap area for each cell
    max_key = max(list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
    cell_area_overlap = cell_overlap_area(
        maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0], mod_overlap)
    # Calculate shading matrix
    op_shade_array = calc_cell_shading(
        cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
    # Add to shade dataframe
    col_list = list(df_shd_sce.columns.values)
    df_new_row = pd.DataFrame(data=[['Standard',
                                     'Mixed',
                                     'Mixed 2 (Ellipse Triangle)',
                                     op_shade_array,
                                     full_poly,
                                     mod_overlap_area,
                                     mod_overlap_area_perc
                                     ]], columns=col_list)
    df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def shade_tree1(maxsys_dict, df_shd_sce, translucence=1, dir_diff_ratio=1):
    """
    Generate user generated tree branch shading.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    # Module dimensions
    MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
    MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
    # Standard shape dimensions.
    # Points
    poly_points = np.asarray([(MX, 0.02*MY), (0.87*MX, 0.02*MY),
                              (0.83*MX, 0.07*MY), (0.83*MX, 0.2*MY),
                              (0.45*MX, 0.2*MY), (0.45*MX, 0.24*MY),
                              (0.5*MX, 0.24*MY), (0.5*MX, 0.3*MY),
                              (0.83*MX, 0.3*MY), (0.83*MX, 0.35*MY),
                              (0.5*MX, 0.35*MY), (0.5*MX, 0.5*MY),
                              (0.75*MX, 0.5*MY), (0.75*MX, 0.45*MY),
                              (0.87*MX, 0.45*MY),
                              (0.87*MX, 0.3*MY), (0.93*MX, 0.3*MY),
                              (0.93*MX, 0.2*MY), (MX, 0.2*MY), (MX, 0.02*MY)])
    # Polygon
    tree_poly = Polygon(poly_points)
    # Check overlap against module
    mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                      0].intersection(tree_poly)
    mod_overlap_area = mod_overlap.area
    mod_overlap_area_perc = 100*mod_overlap.area / \
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
    # Calculate overlap area for each cell
    max_key = max(list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
    cell_area_overlap = cell_overlap_area(
        maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0], mod_overlap)
    # Calculate shading matrix
    op_shade_array = calc_cell_shading(
        cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
    # Add to shade dataframe
    col_list = list(df_shd_sce.columns.values)
    df_new_row = pd.DataFrame(data=[['Standard',
                                     'Tree',
                                     'BR Tree 1',
                                     op_shade_array,
                                     tree_poly,
                                     mod_overlap_area,
                                     mod_overlap_area_perc
                                     ]], columns=col_list)
    df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce


def shade_tree2(maxsys_dict, df_shd_sce, rot_ang, cen_pt,
                translucence=1, dir_diff_ratio=1):
    """
    Generate fern leaf shading. From Amazon plastic fern.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    rot_ang : float
        Rotation angle of the fern.
    cen_pt : str
        Point at which to align bottom most point of fern. Option is BR.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    # Load Shapely pickle file
    data_path = os.path.join(Path(__file__).parent, 'data')
    pickle_path = os.path.join(data_path, r'Updated_Plastic_Fern.pickle')
    fern_polygon = load_pickle(pickle_path)
    for idx_sc, r_ang in enumerate(rot_ang):
        cenpt = cen_pt[0]
        if cenpt == 'BR':
            cp = [cell_coords[-1, -1, 3, 0], cell_coords[-1, -1, 3, 1]]
        else:
            raise ValueError('Incorrect center point. Only option: BR.')
        # Rotate the fern
        rot_poly = rotate(fern_polygon, r_ang, origin=(0, 0))
        # Translate to center point
        tree_poly = translate(rot_poly, xoff=cp[0], yoff=cp[1])
        # Check overlap against module
        mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                          0].intersection(tree_poly)
        mod_overlap_area = mod_overlap.area
        mod_overlap_area_perc = 100*mod_overlap.area / \
            maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
        # Calculate overlap area for each cell
        max_key = max(
            list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
        cell_area_overlap = cell_overlap_area(
            maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
            mod_overlap)
        # Calculate shading matrix
        op_shade_array = calc_cell_shading(
            cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
        # Add to shade dataframe
        col_list = list(df_shd_sce.columns.values)
        df_new_row = pd.DataFrame(data=[['Standard',
                                         'Plastic Fern',
                                         cenpt + ' Fern',
                                         op_shade_array,
                                         tree_poly,
                                         mod_overlap_area,
                                         mod_overlap_area_perc
                                         ]], columns=col_list)
        df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
    return df_shd_sce


def shade_tree3(maxsys_dict, df_shd_sce, rot_ang, cen_pt,
                translucence=1, dir_diff_ratio=1):
    """
    Generate fern leaf shape that is easier to manufacture with max precision.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    rot_ang : float
        Rotation angle of the fern.
    cen_pt : str
        Point at which to align bottom most point of fern. Option is BR.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    # Load Shapely pickle file
    data_path = os.path.join(Path(__file__).parent, 'data')
    pickle_path = os.path.join(data_path, r'Plastic_Fern_2p0.pickle')
    fern_polygon = load_pickle(pickle_path)
    for idx_sc, r_ang in enumerate(rot_ang):
        cenpt = cen_pt[0]
        if cenpt == 'BR':
            cp = [cell_coords[-1, -1, 3, 0], cell_coords[-1, -1, 3, 1]]
        else:
            raise ValueError('Incorrect center point. Only option: BR.')
        # Rotate the fern
        rot_poly = rotate(fern_polygon, r_ang, origin=(0, 0))
        # Translate to center point
        tree_poly = translate(rot_poly, xoff=cp[0], yoff=cp[1])
        # Check overlap against module
        mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                          0].intersection(tree_poly)
        mod_overlap_area = mod_overlap.area
        mod_overlap_area_perc = 100*mod_overlap.area / \
            maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
        # Calculate overlap area for each cell
        max_key = max(
            list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
        cell_area_overlap = cell_overlap_area(
            maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0],
            mod_overlap)
        # Calculate shading matrix
        op_shade_array = calc_cell_shading(
            cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
        # Add to shade dataframe
        col_list = list(df_shd_sce.columns.values)
        df_new_row = pd.DataFrame(data=[['Standard',
                                         'Plastic Fern',
                                         cenpt + ' Fern',
                                         op_shade_array,
                                         tree_poly,
                                         mod_overlap_area,
                                         mod_overlap_area_perc
                                         ]], columns=col_list)
        df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)
    return df_shd_sce


def shade_user_define_objects(maxsys_dict, df_shd_sce, cen_pt_vec, shp_x_vec,
                              shp_y_vec, rot_vec, shp_vec,
                              translucence=1, dir_diff_ratio=1,
                              scen_name='User'):
    """
    Generate a user defined shading scenario.

    Parameters
    ----------
    maxsys_dict : dict
        Dictionary containing the module model.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    cen_pt_vec : list
        List of center points of each shape.
    shp_x_vec : list
        List of x lengths for each shape.
    shp_y_vec : list
        List of y lengths for each shape.
    rot_vec : list
        List of rotation angles for each shape.
    shp_vec : list
        List of shapes. Options are triang, ell, rect.
    translucence : float, optional
        Opacity of the shading. The default is 1.
    dir_diff_ratio : float, optional
        Direct to diffuse irradiance ratio. The default is 1.
    scen_name : str, optional
        Shading name. The default is 'User'.

    Returns
    -------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.

    """
    # Cell dimensions
    cell_coords = maxsys_dict['Physical_Info']['Cell_Coordinates']
    cell_len = cell_coords[0, 0, 3, 0] - cell_coords[0, 0, 0, 0]
    cell_width = cell_coords[0, 0, 1, 1] - cell_coords[0, 0, 0, 1]
    CELLAREA = cell_len * cell_width
    full_poly = []
    # Module dimensions
    MX = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 0]
    MY = maxsys_dict['Physical_Info']['Module_Coordinates'][-1, 0, 2, 1]
    for idx_cp, sx in enumerate(shp_x_vec):
        cp = cen_pt_vec[idx_cp]
        if isinstance(cp, str):
            if 'cell' in cp:
                cell_num = [int(s) for s in re.findall(r'\d+', cp)]
                cell_idx = np.where(
                    maxsys_dict['Physical_Info']['Index_Map'] == cell_num)
                try:
                    if cp == 'cell_mid':
                        cp = [cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 0] + cell_len*0.5,
                              cell_coords[cell_idx[0][0],
                                          cell_idx[1][0],
                                          0, 1] + cell_width*0.5]
                    elif cp == 'cell_top':
                        cp = [cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 0] + cell_len*0.5,
                              cell_coords[cell_idx[0][0],
                                          cell_idx[1][0],
                                          0, 1] + (cell_width - 0.5*shp_y_vec[idx_cp])]
                    elif cp == 'cell_bottom':
                        cp = [cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 0] + cell_len*0.5,
                              cell_coords[cell_idx[0][0],
                                          cell_idx[1][0],
                                          0, 1] + (0.5*shp_y_vec[idx_cp])]
                    elif cp == 'cell_left':
                        cp = [cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 0] + (0.5*sx),
                              cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 1] + cell_width*0.5]
                    elif cp == 'cell_right':
                        cp = [cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 0] + (cell_len - 0.5*sx),
                              cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 1] + cell_width*0.5]
                    else:
                        cp = [cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 0] + cell_len*0.5,
                              cell_coords[cell_idx[0][0],
                                          cell_idx[1][0], 0, 1] + cell_width*0.5]
                except IndexError:
                    cell_row = int((cell_coords.shape[0] - 1)*0.5)
                    cell_col = int((cell_coords.shape[1] - 1)*0.5)
                    # cp = [cell_coords[1, 2, 0, 0] + cell_len*0.5,
                    #       cell_coords[1, 2, 0, 1] + cell_width*0.5]
                    # cp = [cell_coords[cell_row,
                    # cell_col, 0, 0] + cell_len*0.5,
                    #       cell_coords[cell_row,
                    # cell_col, 0, 1] + cell_width*0.5]
                    if cp == 'cell_mid':
                        cp = [cell_coords[cell_row,
                                          cell_col, 0, 0] + cell_len*0.5,
                              cell_coords[cell_row,
                                          cell_col, 0, 1] + cell_width*0.5]
                    elif cp == 'cell_top':
                        cp = [cell_coords[cell_row,
                                          cell_col, 0, 0] + cell_len*0.5,
                              cell_coords[cell_row,
                                          cell_col, 0, 1] + (cell_width - 0.5*shp_y_vec[idx_cp])]
                    elif cp == 'cell_bottom':
                        cp = [cell_coords[cell_row,
                                          cell_col, 0, 0] + cell_len*0.5,
                              cell_coords[cell_row,
                                          cell_col, 0, 1] + (0.5*shp_y_vec[idx_cp])]
                    elif cp == 'cell_left':
                        cp = [cell_coords[cell_row,
                                          cell_col, 0, 0] + (0.5*sx),
                              cell_coords[cell_row,
                                          cell_col, 0, 1] + cell_width*0.5]
                    elif cp == 'cell_right':
                        cp = [cell_coords[cell_row,
                                          cell_col, 0, 0] + (cell_len - 0.5*sx),
                              cell_coords[cell_row,
                                          cell_col, 0, 1] + cell_width*0.5]
                    else:
                        cp = [cell_coords[cell_row,
                                          cell_col, 0, 0] + cell_len*0.5,
                              cell_coords[cell_row,
                                          cell_col, 0, 1] + cell_width*0.5]
                    # cp = [cell_coords[1, 2, 0, 0] + cell_len*0.5,
                    #       cell_coords[1, 2, 0, 1] + cell_width*0.5]
            elif cp == 'BR':
                cell_x = cell_coords[-1, -1, 3, 0]
                cell_y = cell_coords[-1, -1, 3, 1]
                cp = [cell_x - 0.5*sx, cell_y + 0.5*shp_y_vec[idx_cp]]
            elif cp == 'TL':
                cell_x = cell_coords[0, 0, 1, 0]
                cell_y = cell_coords[0, 0, 1, 1]
                cp = [cell_x + 0.5*sx, cell_y - 0.5*shp_y_vec[idx_cp]]
            elif cp == 'BM':
                cp = [0.5*MX, 0.5*shp_y_vec[idx_cp]]
            elif cp == 'TM':
                cp = [0.5*MX, MY - 0.5*shp_y_vec[idx_cp]]
            elif cp == 'MM':
                cp = [0.5*MX, 0.5*MY]
        sy = shp_y_vec[idx_cp]
        shp = shp_vec[idx_cp]
        rot_ang = rot_vec[idx_cp]
        # Create shade polygon (Triangle or Ellipse)
        if shp == 'triang':
            shd_poly = create_rtang_triangle(cp, sx, sy, rot_ang)
        elif shp == 'ell':
            shd_poly = create_ellipse(cp, sx, sy, rot_ang)
        elif shp == 'rect':
            shd_poly = create_rectangle(cp, sx, sy, rot_ang)
        else:
            raise ValueError(
                'Wrong shape inputted! Options are triang, ell, rect.')
        full_poly.append(shd_poly)
    full_poly = unary_union(full_poly)
    # Check overlap against module
    mod_overlap = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1,
                                                                      0].intersection(full_poly)
    mod_overlap_area = mod_overlap.area
    mod_overlap_area_perc = 100*mod_overlap.area / \
        maxsys_dict['Physical_Info']['Module_Polygon'].iloc[-1, 0].area
    # Calculate overlap area for each cell
    max_key = max(list(maxsys_dict['Physical_Info']['Cell_Polygons'].keys()))
    cell_area_overlap = cell_overlap_area(
        maxsys_dict['Physical_Info']['Cell_Polygons'][max_key][0], mod_overlap)
    # Calculate shading matrix
    cell_len_vec = cell_coords[:, :, 3, 0] - cell_coords[:, :, 0, 0]
    cell_wid_vec = cell_coords[:, :, 1, 1] - cell_coords[:, :, 0, 1]
    CELLAREA = cell_len_vec*cell_wid_vec
    op_shade_array = calc_cell_shading(
        cell_area_overlap, CELLAREA, translucence, dir_diff_ratio)
    # Add to shade dataframe
    col_list = list(df_shd_sce.columns.values)
    df_new_row = pd.DataFrame(data=[['User Defined',
                                     'Mixed',
                                     scen_name,
                                     op_shade_array,
                                     full_poly,
                                     mod_overlap_area,
                                     mod_overlap_area_perc
                                     ]], columns=col_list)
    df_shd_sce = pd.concat([df_shd_sce, df_new_row], ignore_index=True)

    return df_shd_sce
