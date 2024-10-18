# -*- coding: utf-8 -*-
"""Plotting functions."""

import os
import dataframe_image as dfi

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from .utils import create_letter_list

plt.style.use('ggplot')


def plot_mod_idx(mods_sys_dict):
    """
    Generate the plot for the module index.

    The plot is created in a "module_index" folder in the current working directory.

    Parameters
    ----------
    mods_sys_dict : dict
        Dictionary containing the physical and electrical models of modules in the simulation.

    Returns
    -------
    None.

    """
    # Get current working directory (old)
    cw = os.getcwd()
    # Create new folder
    newpath = os.path.join(cw, r'module_index')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # Change directory to new
    os.chdir(newpath)
    mod_sys_keys = list(mods_sys_dict.keys())
    # Run for loop on modules
    for mod_name in mod_sys_keys:
        cell_mod_keys = list(mods_sys_dict[mod_name].keys())
        for cell_name in cell_mod_keys:
            orient_keys = list(mods_sys_dict[mod_name][cell_name].keys())
            for orient in orient_keys:
                ec_keys = list(mods_sys_dict[mod_name][cell_name][orient].keys())
                for ec_type in ec_keys:
                    # Create title for plot
                    plot_label = mods_sys_dict[mod_name][cell_name][orient][ec_type]['Sim_info']['plot_label']
                    title_str = plot_label + '_cell_index'
                    s = mods_sys_dict[mod_name][cell_name][orient][ec_type]['Physical_Info']['Formatted_Idx_Map']
                    dfi.export(s, title_str + '.png')
    # Change current working directly to old
    os.chdir(cw)


def print_idx_map(idx_map, idx_crosstie=None):
    """
    Print the index map with some fancy formatting.

    This is useful for visualization in Jupyter Notebooks.

    Parameters
    ----------
    idx_map : numpy.ndarray
        An array of size num_cells_y X num_cells_x with indices for
        each cell in the module.

    Returns
    -------
    Data frame
        Stylized Data Frame.

    """
    df = pd.DataFrame(idx_map,
                      index=range(1, idx_map.shape[0]+1),
                      columns=create_letter_list(idx_map.shape[1]))
    if idx_crosstie is None:
        idx_crosstie = np.zeros(idx_map.shape, dtype=bool)
    df_ct = pd.DataFrame(idx_crosstie,
                         index=range(1, idx_map.shape[0]+1),
                         columns=create_letter_list(idx_map.shape[1]))
    s = df.style
    s.set_table_styles([  # create internal CSS classes
        {'selector': '.True', 'props': 'background-color: #ffe6e6;'},
        {'selector': '.False', 'props': 'background-color: #e6ffe6;'},
    ], overwrite=False)
    s.set_td_classes(df_ct)
    s.set_table_styles([  # create internal CSS classes
        {'selector': '.True', 'props': 'border: 2px dashed red;'},
        {'selector': '.False', 'props': 'border: 2px dashed green;'},
    ], overwrite=False)
    s.set_td_classes(df_ct)
    return s


def plot_shade_module(maxsys_dict, df_shd_sce, plot_file='Module_Shade_Scenarios.pdf', plot_show=False):
    """
    Generate PDF file containing plots of all scenarios for each module in the simulation.

    Parameters
    ----------
    mods_sys_dict : dict
        Dictionary containing the physical and electrical models of modules in the simulation.
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    plot_file : str, optional
        File path of plotting file. The default is 'Module_Shade_Scenarios.pdf'.
    plot_show : bool, optional
        Display plots while running. The default is False.

    Returns
    -------
    None.

    """
    with PdfPages(plot_file) as pdf:
        # Run for loop on rows of df
        for idx_row in range(df_shd_sce.shape[0]):
            # Extract info
            sc_def = df_shd_sce.iloc[idx_row, 0]
            sc_typ = df_shd_sce.iloc[idx_row, 1]
            sc_var = df_shd_sce.iloc[idx_row, 2]
            full_poly = df_shd_sce.iloc[idx_row, 4]
            mod_sh_area = round(df_shd_sce.iloc[idx_row, 5], 2)
            mod_sh_area_per = round(df_shd_sce.iloc[idx_row, 6], 2)
            # Create title for plot
            title_str = sc_def + ' | ' + sc_typ + ' | ' + sc_var
            # Plot
            fig = plt.figure()
            # plot module
            mx, my = maxsys_dict['Physical_Info']['Module_Polygon'].iloc[0, 0].exterior.xy
            maxx = max(mx)
            maxy = max(my)
            maxp = max(maxx, maxy)
            plt.plot(mx, my, 'g'+'-')
            # plot cells
            for idx_row in range(maxsys_dict['Physical_Info']['Cell_Polygons'][0][0].shape[0]):
                for idx_col in range(maxsys_dict['Physical_Info']['Cell_Polygons'][0][0].shape[1]):
                    cell_poly = maxsys_dict['Physical_Info']['Cell_Polygons'][0][0].iloc[idx_row, idx_col]
                    cx, cy = cell_poly.exterior.xy
                    plt.plot(cx, cy, 'b'+'-')
            # plot shade
            if full_poly.type == 'Polygon':
                sx, sy = full_poly.exterior.xy
                plt.fill(sx, sy, 'r')
            elif full_poly.type == 'MultiPolygon':
                full_poly = list(full_poly)
                for shd_poly in full_poly:
                    sx, sy = shd_poly.exterior.xy
                    plt.fill(sx, sy, 'r')
            plt.xlim([-10, maxp + 50])
            plt.ylim([-10, maxp + 50])
            plt.title(title_str, fontsize=10)
            plt.ylabel('mm')
            plt.xlabel('mm')
            min_ax = min(maxx, maxy)
            max_ax = max(maxx, maxy)
            plt.text(min_ax+100, max_ax, 'Module Shaded Area Percent [%] : ' + str(mod_sh_area_per),
                     horizontalalignment='left',
                     verticalalignment='center',
                     fontsize=6.5)
            plt.text(min_ax+100, max_ax-100, 'Module Shaded Area [mm^2] : ' + str(mod_sh_area),
                     horizontalalignment='left',
                     verticalalignment='center',
                     fontsize=6.5)
            pdf.savefig(figure=fig)
            if not plot_show:
                plt.close()


def plot_shade_array(df_shd_sce, plot_label, is_Landscape):
    """
    Generate the shading intersection arrays for each cell in the module.

    Parameters
    ----------
    df_shd_sce : pandas.DataFrame
        Dataframe containing the shading scenarios information.
    plot_label : str
        Plot label of the module.
    is_Landscape : bool
        Is module in Landscape orientation?

    Returns
    -------
    None.

    """
    # Get current working directory (old)
    cw = os.getcwd()
    # Create new folder
    newpath = os.path.join(cw, r'shade_arrays')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # Change directory to new
    os.chdir(newpath)
    # Run for loop on rows of df
    for idx_row in range(df_shd_sce.shape[0]):
        # Extract DF Info
        idx_map = df_shd_sce.iloc[idx_row, 3]
        if is_Landscape:
            idx_map = np.rot90(idx_map, axes=(1, 0))
        idx_map = np.round(idx_map, 2)
        sc_def = df_shd_sce.iloc[idx_row, 0]
        sc_typ = df_shd_sce.iloc[idx_row, 1]
        sc_var = df_shd_sce.iloc[idx_row, 2]
        # Create title for plot
        title_str = plot_label + '_' + sc_def + '_' + sc_typ + '_' + sc_var
        # Create Module DF
        df = pd.DataFrame(idx_map,
                          index=range(1, idx_map.shape[0]+1),
                          columns=create_letter_list(idx_map.shape[1]))
        # Add Gradient styling
        s = df.style.background_gradient(axis=None, vmin=0.0, vmax=1.0)
        # Convert to PNG and store in "shade_arrays" folder
        dfi.export(s, title_str + '.png')
    # Change current working directly to old
    os.chdir(cw)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Create a random colormap to be used together with matplotlib.

    Useful for segmentation tasks.
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    # from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(
                HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        # random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        # random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    # if verbose:
    #     from matplotlib import colors, colorbar
    #     from matplotlib import pyplot as plt
    #     fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

    #     bounds = np.linspace(0, nlabels, nlabels + 1)
    #     norm = colors.BoundaryNorm(bounds, nlabels)

    #     cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
    #                                boundaries=bounds, format='%1i', orientation=u'horizontal')

    return randRGBcolors


def gen_sing_mod_shade_type_plots(dfCases, plot_show=False):
    """
    Generate bar charts for each module with shade sccenarios grouped together.

    Parameters
    ----------
    dfCases : pandas.DataFrame
        Dataframe containing summarized results.
    plot_show : bool, optional
        Display plot while running. The default is False.

    Returns
    -------
    None.

    """
    # Get current working directory (old)
    cw = os.getcwd()
    # Create new folder
    newpath = cw + '\\' + r'results_plots'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Get Mod names
    mod_names = dfCases['Module'].unique()
    # Get unique num shade mods
    num_mod_shd = dfCases['Num Mod shade'].unique()
    col_list = list(dfCases.columns.values)

    # For each module type
    for mod_name in mod_names:
        # For each number of shaded modules
        for num_mod in num_mod_shd:
            # Reduce DF
            sub_df = dfCases[(dfCases['Module'] == mod_name) &
                             (dfCases['Num Mod shade'] == num_mod)]
            cell_names = sub_df['Cell Name'].unique()
            for cell_name in cell_names:
                sub_df1 = sub_df[(sub_df['Cell Name'] == cell_name)].copy()
                orientations = sub_df['Orientation'].unique()
                for orientation in orientations:
                    sub_df11 = sub_df1[(
                        sub_df1['Orientation'] == orientation)].copy()
                    acdcs = sub_df11['DC/AC'].unique()
                    for acdc in acdcs:
                        sub_df2 = sub_df11[(sub_df11['DC/AC'] == acdc)].copy()
                        # Count number of shade types
                        shade_type_counts = sub_df2['Shade Type'].value_counts(
                        )
                        # Separate low counts from high counts
                        shade_type_great = shade_type_counts[shade_type_counts > shade_type_counts.min(
                        )].index.tolist()
                        shade_type_less = shade_type_counts[shade_type_counts <= shade_type_counts.min(
                        )].index.tolist()
                        shade_type_less.remove('No Shade')
                        # Reduce DF for Low counts cases
                        shade_type_less.insert(0, 'No Shade')
                        sub_sub_df = sub_df2[sub_df2['Shade Type'].isin(
                            shade_type_less)]
                        sub_sub_df_all = sub_sub_df.copy()
                        sub_sub_df['Shade Type Var'] = sub_sub_df['Shade Type'] + \
                            ' | ' + sub_sub_df['Shade Variation']
                        # Generate IV & PV Curves for low counts cases
                        plot_label_idx = sub_sub_df.index.tolist()[0]
                        plot_label = sub_sub_df['Plot Label'][plot_label_idx]
                        plot_file = newpath + '\\' + plot_label + '_' + \
                            '_NumModsShade_' + str(num_mod) + '.pdf'
                        with PdfPages(plot_file) as pdf:
                            new_cmap = rand_cmap(
                                sub_sub_df.shape[0], type='bright', first_color_black=True, last_color_black=False,
                                verbose=True)
                            # Plot IV & PV
                            shd_typg = 'Mixed'
                            pdf = plot_iv_pv_curves(pdf, sub_sub_df, shd_typg,
                                                    plot_label,
                                                    num_mod, plot_show, new_cmap)
                            # Plot Power Change & Pmp
                            pdf = plot_PC_Pmp_single(pdf, sub_sub_df, shd_typg,
                                                     plot_label,
                                                     num_mod, plot_show, newpath)
                            # Generate IV & PV Curves for high counts cases
                            for shd_typg in shade_type_great:
                                sub_sub_df = sub_df2[(sub_df2['Shade Type'] == shd_typg) | (
                                    sub_df2['Shade Type'] == 'No Shade')]
                                sub_sub_df['Shade Type Var'] = sub_sub_df['Shade Type'] + \
                                    ' | ' + sub_sub_df['Shade Variation']
                                new_cmap = rand_cmap(
                                    sub_sub_df.shape[0], type='bright', first_color_black=True, last_color_black=False,
                                    verbose=True)
                                # Plot IV & PV
                                pdf = plot_iv_pv_curves(pdf, sub_sub_df, shd_typg,
                                                        plot_label,
                                                        num_mod, plot_show, new_cmap)
                                # Plot Power Change & Pmp
                                pdf = plot_PC_Pmp_single(pdf, sub_sub_df, shd_typg,
                                                         plot_label,
                                                         num_mod, plot_show, newpath)
                                sub_sub_df = sub_df2[(
                                    sub_df2['Shade Type'] == shd_typg)]
                                sss_df_idx = sub_sub_df.index.tolist()
                                sss_avg = sub_sub_df[['Mod. Shade %', 'Pmp [W]', 'Vmp [V]', 'Imp [A]',
                                                      'Voc [V]', 'Isc [A]', 'FF', 'Power change [%]']].median(axis=0)
                                sss_min = sub_sub_df[['Mod. Shade %', 'Pmp [W]', 'Vmp [V]', 'Imp [A]',
                                                      'Voc [V]', 'Isc [A]', 'FF', 'Power change [%]']].min(axis=0)
                                sss_max = sub_sub_df[['Mod. Shade %', 'Pmp [W]', 'Vmp [V]', 'Imp [A]',
                                                      'Voc [V]', 'Isc [A]', 'FF', 'Power change [%]']].max(axis=0)
                                df_new_row = pd.DataFrame(data=[[mod_name, cell_name, orientation, acdc, plot_label,
                                                                 num_mod, sub_sub_df['Shade Definition'][sss_df_idx[0]],
                                                                 sub_sub_df['Shade Type'][sss_df_idx[0]
                                                                                          ], 'Agg. Avg.',
                                                                 (sss_min['Mod. Shade %'] +
                                                                  sss_max['Mod. Shade %'] + sss_avg['Mod. Shade %'])/3,
                                                                 (sss_min['Pmp [W]'] +
                                                                  sss_max['Pmp [W]'] + sss_avg['Pmp [W]'])/3,
                                                                 (sss_min['Vmp [V]'] +
                                                                  sss_max['Vmp [V]'] + sss_avg['Vmp [V]'])/3,
                                                                 (sss_min['Imp [A]'] +
                                                                  sss_max['Imp [A]'] + sss_avg['Imp [A]'])/3,
                                                                 (sss_min['Voc [V]'] +
                                                                  sss_max['Voc [V]'] + sss_avg['Voc [V]'])/3,
                                                                 (sss_min['Isc [A]'] +
                                                                  sss_max['Isc [A]'] + sss_avg['Isc [A]'])/3,
                                                                 (sss_min['FF'] +
                                                                  sss_max['FF'] + sss_avg['FF'])/3,
                                                                 (sss_min['Power change [%]'] +
                                                                  sss_max['Power change [%]'] + sss_avg['Power change [%]'])/3,
                                                                 np.nan, np.nan, np.nan, np.nan
                                                                 ]
                                                                ], columns=col_list)
                                sub_sub_df_all = pd.concat(
                                    [sub_sub_df_all, df_new_row], ignore_index=True)
                            sub_sub_df_all['Shade Type Var'] = sub_sub_df_all['Shade Type'] + \
                                ' | ' + sub_sub_df_all['Shade Variation']
                            # Plot Power Change & Pmp
                            shd_typg = 'Mixed'
                            pdf = plot_PC_Pmp_single(pdf, sub_sub_df_all, shd_typg,
                                                     plot_label,
                                                     num_mod, plot_show, newpath)
                            # plot_PC_Box_single(pdf, sub_df2, shd_typg, plot_label,
                            # num_mod, plot_show)


def gen_all_mods_num_shd_plots(dfCases, plot_show=False):
    """
    Generate bar plots for all modules in the same plot, grouping shade scenarios together.

    Parameters
    ----------
    dfCases : pandas.DataFrame
        Dataframe containing summarized results.
    plot_show : bool, optional
        Display plot while running. The default is False.

    Returns
    -------
    None.

    """
    # Get current working directory (old)
    cw = os.getcwd()
    # Create new folder
    newpath = cw + '\\' + r'results_plots'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Get Mod names
    mod_names = dfCases['Module'].unique()
    cell_names = dfCases['Cell Name'].unique()
    orientations = dfCases['Orientation'].unique()
    # Get unique num shade mods
    num_mod_shd = dfCases['Num Mod shade'].unique()
    col_list = list(dfCases.columns.values)

    # For each number of shaded modules
    for num_mod in num_mod_shd:
        # Reduce DF
        sub_df = dfCases[dfCases['Num Mod shade'] == num_mod]
        # Count number of shade types
        shade_type_counts = sub_df['Shade Type'].value_counts()
        # Separate low counts from high counts
        shade_type_great = shade_type_counts[shade_type_counts > shade_type_counts.min(
        )].index.tolist()
        shade_type_less = shade_type_counts[shade_type_counts <= shade_type_counts.min(
        )].index.tolist()
        shade_type_less.remove('No Shade')
        # Reduce DF for Low counts cases
        shade_type_less.insert(0, 'No Shade')
        sub_sub_df = sub_df[sub_df['Shade Type'].isin(shade_type_less)]
        sub_sub_df['Shade Type Var'] = sub_sub_df['Shade Type'] + \
            ' | ' + sub_sub_df['Shade Variation']

        sub_sub_df_all = sub_sub_df.copy()
        plot_file = newpath + '\\' + \
            'AllMods_NumModsShade_' + str(num_mod) + '.pdf'
        leg_file = newpath + '\\' + 'AllMods_NumModsShade_' + \
            str(num_mod) + '_legend.png'
        with PdfPages(plot_file) as pdf:
            # Plot Power Change & Pmp
            shd_typg = 'Mixed'
            pdf = plot_PC_Pmp_all(pdf, sub_sub_df, shd_typg,
                                  num_mod, plot_show, leg_file=leg_file)
            # HI count cases
            for shd_typg in shade_type_great:
                sub_sub_df = sub_df[(sub_df['Shade Type'] == shd_typg) | (
                    sub_df['Shade Type'] == 'No Shade')]
                sub_sub_df['Shade Type Var'] = sub_sub_df['Shade Type'] + \
                    ' | ' + sub_sub_df['Shade Variation']
                # Plot Power Change & Pmp
                pdf = plot_PC_Pmp_all(
                    pdf, sub_sub_df, shd_typg, num_mod, plot_show, leg_file=leg_file)
                # Calculate Averages of Hi count cases and plot wil low count cases
                for mod_name in mod_names:
                    sss_df = sub_sub_df[sub_sub_df['Module'] == mod_name]
                    cell_names = sss_df['Cell Name'].unique()
                    for cell_name in cell_names:
                        sss_df1 = sss_df[(sss_df['Cell Name']
                                          == cell_name)].copy()
                        orientations = sss_df1['Orientation'].unique()
                        for orientation in orientations:
                            sss_df11 = sss_df1[(
                                sss_df1['Orientation'] == orientation)].copy()
                            acdcs = sss_df11['DC/AC'].unique()
                            for acdc in acdcs:
                                sss_df2 = sss_df11[(
                                    sss_df11['DC/AC'] == acdc)].copy()
                                sss_df2 = sss_df2[sss_df2['Shade Type'].str.contains(
                                    'No Shade') == False]
                                sss_df_idx = sss_df2.index.tolist()
                                plot_label = sub_sub_df['Plot Label'][sss_df_idx[0]]
                                sss_avg = sss_df2[['Mod. Shade %', 'Pmp [W]', 'Vmp [V]', 'Imp [A]',
                                                   'Voc [V]', 'Isc [A]', 'FF', 'Power change [%]']].median(axis=0)
                                sss_min = sss_df2[['Mod. Shade %', 'Pmp [W]', 'Vmp [V]', 'Imp [A]',
                                                   'Voc [V]', 'Isc [A]', 'FF', 'Power change [%]']].min(axis=0)
                                sss_max = sss_df2[['Mod. Shade %', 'Pmp [W]', 'Vmp [V]', 'Imp [A]',
                                                   'Voc [V]', 'Isc [A]', 'FF', 'Power change [%]']].max(axis=0)
                                df_new_row = pd.DataFrame(data=[[mod_name, cell_name, orientation, acdc, plot_label,
                                                                 num_mod, sub_sub_df['Shade Definition'][sss_df_idx[0]],
                                                                 sub_sub_df['Shade Type'][sss_df_idx[0]
                                                                                          ], 'Agg. Avg.',
                                                                 (sss_min['Mod. Shade %'] +
                                                                  sss_max['Mod. Shade %'] + sss_avg['Mod. Shade %'])/3,
                                                                 (sss_min['Pmp [W]'] +
                                                                  sss_max['Pmp [W]'] + sss_avg['Pmp [W]'])/3,
                                                                 (sss_min['Vmp [V]'] +
                                                                  sss_max['Vmp [V]'] + sss_avg['Vmp [V]'])/3,
                                                                 (sss_min['Imp [A]'] +
                                                                  sss_max['Imp [A]'] + sss_avg['Imp [A]'])/3,
                                                                 (sss_min['Voc [V]'] +
                                                                  sss_max['Voc [V]'] + sss_avg['Voc [V]'])/3,
                                                                 (sss_min['Isc [A]'] +
                                                                  sss_max['Isc [A]'] + sss_avg['Isc [A]'])/3,
                                                                 (sss_min['FF'] +
                                                                  sss_max['FF'] + sss_avg['FF'])/3,
                                                                 (sss_min['Power change [%]'] +
                                                                  sss_max['Power change [%]'] + sss_avg['Power change [%]'])/3,
                                                                 np.nan, np.nan, np.nan, np.nan
                                                                 ]
                                                                ], columns=col_list)
                                sub_sub_df_all = pd.concat(
                                    [sub_sub_df_all, df_new_row], ignore_index=True)
                sub_sub_df_all['Shade Type Var'] = sub_sub_df_all['Shade Type'] + \
                    ' | ' + sub_sub_df_all['Shade Variation']
                # Plot Power Change & Pmp
                shd_typg = 'Mixed'
                pdf = plot_PC_Pmp_all(
                    pdf, sub_sub_df_all, shd_typg, num_mod, plot_show, leg_file=leg_file)
                # pdf = plot_PC_box_all(pdf, sub_df, shd_typg, num_mod, plot_show, leg_file = leg_file)


def plot_iv_pv_curves(pdf, sub_sub_df, shd_typg, plot_label,
                      num_mod, plot_show, new_cmap):
    """
    Plot IV and PV curves for grouped shade scenarios.

    Parameters
    ----------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.
    sub_sub_df : pandas.DataFrame
        Dataframe containing all scenarios to be plotted.
    shd_typg : str
        Grouped shade scenario name.
    plot_label : str
        Plot label for module.
    num_mod : int
        Number of modules that are shaded in the string.
    plot_show : bool, optional
        Display plot while running. The default is False.
    new_cmap : Matplotlib cmap
        Color map to use in plot.

    Returns
    -------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.

    """
    # Plot IV
    fig = plt.figure()
    p = 0
    xtick_lst = []
    for idx_row in sub_sub_df.index.tolist():
        # Extract I,V,P
        Isys = sub_sub_df['Isys [A]'][idx_row]
        Vsys = sub_sub_df['Vsys [V]'][idx_row]
        Psys = sub_sub_df['Psys [W]'][idx_row]
        Vmp = sub_sub_df['Vmp [V]'][idx_row]
        Imp = sub_sub_df['Imp [A]'][idx_row]
        Pmp = sub_sub_df['Pmp [W]'][idx_row]
        Voc = sub_sub_df['Voc [V]'][idx_row]
        Isc = sub_sub_df['Isc [A]'][idx_row]
        # Legend text
        leg_txt = sub_sub_df['Shade Type'][idx_row] + \
            ' | ' + sub_sub_df['Shade Variation'][idx_row]
        xtick_lst.append(leg_txt)
        # Title text
        titl_txt = shd_typg + ' IV Curve ' + \
            plot_label + ' Num Mods Shaded ' + str(num_mod)
        # Plot
        plt.plot(Vsys, Isys, c=new_cmap[p], label=leg_txt)
        plt.plot(Vmp, Imp, c=new_cmap[p], linestyle='', marker='o')
        plt.plot(Voc, 0, c=new_cmap[p], linestyle='', marker='o')
        plt.plot(0, Isc, c=new_cmap[p], linestyle='', marker='o')
        p += 1
    plt.grid(True)
    plt.title(titl_txt, fontsize=8)
    plt.ylabel('current [A]')
    plt.xlabel('voltage [V]')
    plt.xlim([0, sub_sub_df['Voc [V]'].max()*1.1])
    plt.ylim([0, sub_sub_df['Isc [A]'].max()*1.1])
    plt.legend(fontsize=8)
    pdf.savefig(figure=fig)
    if not plot_show:
        plt.close()
    # Plot PV
    fig = plt.figure()
    plt.style.use('ggplot')
    p = 0
    for idx_row in sub_sub_df.index.tolist():
        # Extract I,V,P
        Isys = sub_sub_df['Isys [A]'][idx_row]
        Vsys = sub_sub_df['Vsys [V]'][idx_row]
        Psys = sub_sub_df['Psys [W]'][idx_row]
        Vmp = sub_sub_df['Vmp [V]'][idx_row]
        Imp = sub_sub_df['Imp [A]'][idx_row]
        Pmp = sub_sub_df['Pmp [W]'][idx_row]
        Voc = sub_sub_df['Voc [V]'][idx_row]
        Isc = sub_sub_df['Isc [A]'][idx_row]
        # Legend text
        leg_txt = sub_sub_df['Shade Type'][idx_row] + \
            ' | ' + sub_sub_df['Shade Variation'][idx_row]
        # Title text
        titl_txt = shd_typg + ' PV Curve ' + \
            plot_label + ' Num Mods Shaded ' + str(num_mod)
        # Plot
        plt.plot(Vsys, Psys, c=new_cmap[p], label=leg_txt)
        plt.plot(Vmp, Pmp, c=new_cmap[p], linestyle='', marker='o')
        p += 1
    plt.grid(True)
    plt.title(titl_txt, fontsize=8)
    plt.ylabel('power [W]')
    plt.xlabel('voltage [V]')
    plt.xlim([0, sub_sub_df['Voc [V]'].max()*1.1])
    plt.ylim([0, sub_sub_df['Pmp [W]'].max()*1.1])
    plt.legend(fontsize=4)
    pdf.savefig(figure=fig)
    if not plot_show:
        plt.close()
    return pdf


def plot_PC_Pmp_single(pdf, sub_sub_df, shd_typg, plot_label,
                       num_mod, plot_show):
    """
    Plot the Pmp bar chart for grouped shade scenarios.

    Parameters
    ----------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.
    sub_sub_df : pandas.DataFrame
        Dataframe containing all scenarios to be plotted.
    shd_typg : str
        Grouped shade scenario name.
    plot_label : str
        Plot label for module.
    num_mod : int
        Number of modules that are shaded in the string.
    plot_show : bool, optional
        Display plot while running. The default is False.

    Returns
    -------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.

    """
    # Plot Power Change
    titl_txt = shd_typg + ' Power Change ' + \
        plot_label + ' Num Mods Shaded ' + str(num_mod)
    sub_sub_df.plot.bar(x='Shade Type Var', y='Power change [%]')
    plt.title(titl_txt, fontsize=8)
    plt.xticks(fontsize=6)
    plt.subplots_adjust(bottom=0.5)
    pdf.savefig()
    if not plot_show:
        plt.close()
    # dfi.export(sub_sub_df[['Plot Label', 'Shade Type', 'Shade Variation', 'Power change [%]']],
    #         newpath + '\\' + shd_typg + ' Power Change ' + plot_label + ' Num Mods Shaded ' +  str(num_mod) + '.png')
    # Plot Pmp
    titl_txt = shd_typg + ' Pmp ' + plot_label + \
        ' Num Mods Shaded ' + str(num_mod)
    sub_sub_df.plot.bar(x='Shade Type Var', y='Pmp [W]')
    plt.title(titl_txt, fontsize=8)
    plt.xticks(fontsize=5)
    plt.subplots_adjust(bottom=0.5)
    pdf.savefig()
    if not plot_show:
        plt.close()
    # dfi.export(sub_sub_df[['Plot Label', 'Shade Type', 'Shade Variation', 'Pmp [W]']],
    #            newpath + '\\' + shd_typg + ' Pmp ' + plot_label + ' Num Mods Shaded ' +  str(num_mod) + '.png')
    return pdf


def plot_PC_Box_single(pdf, sub_sub_df, shd_typg, plot_label,
                       num_mod, plot_show):
    """
    Plot boxplot of Pmp for grouped shade scenarios.

    Parameters
    ----------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.
    sub_sub_df : pandas.DataFrame
        Dataframe containing all scenarios to be plotted.
    shd_typg : str
        Grouped shade scenario name.
    plot_label : str
        Plot label for module.
    num_mod : int
        Number of modules that are shaded in the string.
    plot_show : bool, optional
        Display plot while running. The default is False.

    Returns
    -------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.

    """
    # Plot Power Change
    copy_sub_sub_df = sub_sub_df.copy()

    titl_txt = shd_typg + ' Num Mods Shaded ' + str(num_mod)
    sns.boxplot(data=copy_sub_sub_df, x="Shade Type", y="Power change [%]")
    plt.xticks(fontsize=6, rotation=90)
    # plt.ylim([-105, 0])
    plt.title(titl_txt, fontsize=8)
    plt.legend(fontsize=5)

    plt.subplots_adjust(bottom=0.5)
    pdf.savefig()
    if not plot_show:
        plt.close()
    plt.clf()
    return pdf


def plot_PC_Pmp_all(pdf, sub_sub_df, shd_typg, num_mod, plot_show, show_legend=False, leg_file='legend.png'):
    """
    Generate the Pmp bar plots for grouped shade scenarios for all modules in one plot.

    Parameters
    ----------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.
    sub_sub_df : pandas.DataFrame
        Dataframe containing all scenarios to be plotted.
    shd_typg : str
        Grouped shade scenario name.
    num_mod : int
        Number of modules that are shaded in the string.
    plot_show : bool, optional
        Display plot while running. The default is False.
    show_legend : bool, optional
        Display legend. The default is False.
    leg_file : str, optional
        Legend screenshot file path. The default is 'legend.png'.

    Returns
    -------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.

    """
    copy_sub_sub_df = sub_sub_df.copy()
    copy_sub_sub_df['Full Module'] = copy_sub_sub_df['Module'] + ' | ' + \
        copy_sub_sub_df['Cell Name'] + ' | ' + copy_sub_sub_df['Orientation']

    titl_txt = shd_typg + ' Num Mods Shaded ' + str(num_mod)
    sns.barplot(data=copy_sub_sub_df, x="Shade Type Var",
                y="Power change [%]", hue="Plot Label", palette="Set2")
    plt.xticks(fontsize=6, rotation=90)
    #plt.ylim([-105, 0])
    plt.title(titl_txt, fontsize=8)
    if show_legend:
        plt.legend(fontsize=5)
    else:
        legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if not os.path.isfile(leg_file):
            export_legend(legend, filename=leg_file)
        legend.remove()
    plt.subplots_adjust(bottom=0.5)
    pdf.savefig()
    if not plot_show:
        plt.close()
    plt.clf()
    # Plot Pmp
    sns.barplot(data=copy_sub_sub_df, x="Shade Type Var",
                y="Pmp [W]", hue="Plot Label", palette="Set2")
    plt.xticks(fontsize=5, rotation=90)
    plt.title(titl_txt, fontsize=8)
    if show_legend:
        plt.legend(fontsize=3)
    else:
        legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if not os.path.isfile(leg_file):
            export_legend(legend, filename=leg_file)
        legend.remove()
    plt.subplots_adjust(bottom=0.5)
    pdf.savefig()
    if not plot_show:
        plt.close()
    plt.clf()
    return pdf


def plot_PC_box_all(pdf, sub_df, shd_typg, num_mod, plot_show, show_legend=False, leg_file='legend.png'):
    """
    Generate the Pmp box plots for grouped shade scenarios for all modules in one plot.

    Parameters
    ----------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.
    sub_sub_df : pandas.DataFrame
        Dataframe containing all scenarios to be plotted.
    shd_typg : str
        Grouped shade scenario name.
    num_mod : int
        Number of modules that are shaded in the string.
    plot_show : bool, optional
        Display plot while running. The default is False.
    show_legend : bool, optional
        Display legend. The default is False.
    leg_file : str, optional
        Legend screenshot file path. The default is 'legend.png'.

    Returns
    -------
    pdf : multi-pdf object
        Matplotlib object containing all figures to be saved to pdf.

    Returns
    -------
    pdf : TYPE
        DESCRIPTION.

    """
    copy_sub_sub_df = sub_df.copy()

    titl_txt = shd_typg + ' Num Mods Shaded ' + str(num_mod)
    sns.boxplot(data=copy_sub_sub_df, x="Shade Type",
                y="Power change [%]", hue="Plot Label", palette="Set2")
    plt.xticks(fontsize=6, rotation=90)
    # plt.ylim([-105, 0])
    plt.title(titl_txt, fontsize=8)
    if show_legend:
        plt.legend(fontsize=5)
    else:
        legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if not os.path.isfile(leg_file):
            export_legend(legend, filename=leg_file)
        legend.remove()
    plt.subplots_adjust(bottom=0.5)
    pdf.savefig()
    if not plot_show:
        plt.close()
    plt.clf()
    return pdf


def export_legend(legend, expand=[-5, -5, 5, 5], filename="legend.png"):
    """
    Export the legend of a plot into an image file.

    Parameters
    ----------
    legend : matplotlib object
        Legend object.
    expand : list, optional
        Size scaling for the legend expansion. The default is [-5, -5, 5, 5].
    filename : str, optional
        Legend screenshot file path. The default is "legend.png".

    Returns
    -------
    None.

    """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
