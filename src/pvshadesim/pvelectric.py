# -*- coding: utf-8 -*-
"""Simulate the electrical model of module/ system for all shade scenarios."""

import time
import os
from pathlib import Path
import copy
import warnings

import pandas as pd
import numpy as np

from v_pvmismatch import vpvsystem, vpvcell, vpvmodule, vpvstring, cell_curr
from v_pvmismatch.utils import round_to_dec

from .utils import save_pickle, delete_files_with_substrings

warnings.filterwarnings("ignore")
db_path = os.path.join(Path(__file__).parent, 'db')
# Create IV database path
IV_fold = 'IV_DB'
IV_DB_loc = os.path.join(db_path, IV_fold)
if os.path.isdir(IV_DB_loc) is not True:
    os.makedirs(IV_DB_loc)


def gen_pvmmvec_shade_results(mods_sys_dict,
                              pickle_fn='Gen_PVMM_Vectorized_Shade_Results.pickle',
                              irrad_suns=1, Tcell=298.15, NPTS=1500,
                              NPTS_cell=100, use_cell_NPT=False,
                              save_detailed=False, TUV_class=False,
                              for_gui=False,
                              excel_fn="PVMM_Vectorized_Shade_Simulation_Results.xlsx",
                              d_p_fn='Detailed_Data.pickle',
                              run_cellcurr=True,
                              c_p_fn='Cell_current.pickle', Ee_round=4,
                              run_isc=False):
    """
    Run vectorized PVMismatch for all modules, and shade scenarios in sim.

    Parameters
    ----------
    mods_sys_dict : dict
        Dict containing physical and electrical models of modules in sim.
    pickle_fn : str, optional
        Pickle file containing all detailed results.
        The default is 'Gen_PVMM_Vectorized_Shade_Results.pickle'.
    irrad_suns : float, optional
        Nominal irradiance in suns. The default is 1.
    Tcell : float, optional
        Nominal cell temperature in kelvin. The default is 298.15.
    NPTS : int, optional
        Number of points in IV curve. The default is 1500.
    NPTS_cell : int, optional
        Number of points in cell IV curve. The default is 100.
    use_cell_NPT : bool, optional
        Use separate NPTS_cell parameter. The default is False.
    save_detailed : bool, optional
        Save detailed results. The default is False.
    TUV_class : bool, optional
        Run TUV shading tests. The default is False.
    for_gui : bool, optional
        Generate module pickle files for Maxeon shading GUI.
        The default is False.
    excel_fn : str, optional
        Path of Results output file.
        The default is "PVMM_Vectorized_Shade_Simulation_Results.xlsx".
    d_p_fn : str, optional
        Detailed pickle file name. The default is 'Detailed_Data.pickle'.
    run_cellcurr : bool, optional
        Run cell current estimation model.
        The default is True.
    c_p_fn : str, optional
        Cell current estimation pickle file name.
        The default is 'Cell_current.pickle'.
    Ee_round : int, optional
        Rounding factor for Irradiance.
        The default is 2.

    Returns
    -------
    dfCases : pandas.DataFrame
        Dataframe containing summarized results.

    """
    data_path = os.path.join(os.getcwd(), 'VPVMM_data')
    if os.path.isdir(data_path) is not True:
        os.makedirs(data_path)
    t0 = time.time()
    if pickle_fn is not None:
        # Create Empty dataframe to store results
        res_names = ['Module', 'Cell Name', 'Orientation', 'DC/AC',
                     'Plot Label', 'Num Mod shade', 'Shade Definition',
                     'Shade Type', 'Shade Variation', 'Mod. Shade %',
                     'Pmp [W]', 'Vmp [V]', 'Imp [A]', 'Voc [V]',
                     'Isc [A]', 'FF', 'Power change [%]',
                     'Num_BPdiode_active',
                     'ncells_Rev_mpp', 'ncells_Rev_isc',
                     'AL', 'TUV Class', 'Isys [A]',
                     'Vsys [V]', 'Psys [W]', 'sys_class']
        dfCases = pd.DataFrame(columns=res_names)
        if TUV_class:
            tuv_cols = ['Module', 'Cell Name', 'Plot Label',
                        'Shade Type', 'AL', 'TUV Score']
            df_TUV = pd.DataFrame(columns=tuv_cols)
        mod_sys_keys = list(mods_sys_dict.keys())
        if save_detailed:
            detailed_dict = {}
        for mod_name in mod_sys_keys:
            cell_mod_keys = list(mods_sys_dict[mod_name].keys())
            for cell_name in cell_mod_keys:
                orient_keys = list(mods_sys_dict[mod_name][cell_name].keys())
                for orient in orient_keys:
                    ec_keys = list(
                        mods_sys_dict[mod_name][cell_name][orient].keys())
                    for ec_type in ec_keys:
                        t1 = time.time()
                        maxsys_dict = mods_sys_dict[mod_name][cell_name][orient][ec_type]
                        df_shd_sce = mods_sys_dict[mod_name][cell_name][orient][ec_type]['Shade Scenarios']
                        idx_map = maxsys_dict['Physical_Info']['Index_Map']
                        # Get base PVMM module
                        maxmod = maxsys_dict['Electrical_Circuit']['PV_Module']
                        outer_circuit = maxsys_dict['Electrical_Circuit']['outer_circuit']
                        # Extract Sim info
                        str_len = int(maxsys_dict['Sim_info']['str_len'])
                        num_str = int(maxsys_dict['Sim_info']['num_str'])
                        num_mods_shade = maxsys_dict['Sim_info']['num_mods_shade']
                        is_AC_Mod = maxsys_dict['Sim_info']['is_AC_Mod']
                        is_sub_Mod = maxsys_dict['Sim_info']['is_sub_Mod']
                        plot_label = maxsys_dict['Sim_info']['plot_label']
                        # Build Ee array for system
                        SA_list = maxsys_dict['Shade Scenarios']['Shade Array'].to_list(
                        )
                        Ee_shdarr_np = np.stack(SA_list, axis=0)
                        # Ee_shdarr_np[Ee_shdarr_np < 0.01] = 0.01
                        Ee_shdarr_np = np.round(irrad_suns*(1-Ee_shdarr_np),
                                                Ee_round)
                        Ee_arr_np = irrad_suns * \
                            np.ones(
                                (Ee_shdarr_np.shape[0], num_str, str_len,
                                 idx_map.shape[0], idx_map.shape[1]))
                        for nmod_sh in num_mods_shade:
                            if nmod_sh > str_len:
                                nmod_sh = str_len
                            for idx_m in range(nmod_sh):
                                for i_str in range(num_str):
                                    Ee_arr_np[:, i_str, idx_m,
                                              :, :] = Ee_shdarr_np
                            # Create sub DF
                            dfSubCases = pd.DataFrame(
                                columns=res_names,
                                index=list(range(df_shd_sce.shape[0])))
                            dfSubCases['Module'] = mod_name
                            dfSubCases['Cell Name'] = cell_name
                            dfSubCases['Orientation'] = orient
                            dfSubCases['DC/AC'] = ec_type
                            dfSubCases['Plot Label'] = plot_label
                            dfSubCases['Num Mod shade'] = nmod_sh
                            dfSubCases['Shade Definition'] = df_shd_sce['Scenario Definition'].to_list(
                            )
                            dfSubCases['Shade Type'] = df_shd_sce['Scenario Type'].to_list(
                            )
                            dfSubCases['Shade Variation'] = df_shd_sce['Scenario Variation'].to_list(
                            )
                            dfSubCases['Mod. Shade %'] = df_shd_sce['Module Shaded Area Percentage'].to_list(
                            )
                            # Vectorized PVMM

                            # Cell pos and Vbypass
                            cell_pos = maxsys_dict['Electrical_Circuit']['Cell_Postion']
                            maxmod = maxsys_dict['Electrical_Circuit']['PV_Module']
                            cell_type = maxsys_dict['Physical_Info']['Cell_type']

                            # Generate Ee & Tcell array for simulation
                            Ee_vec, Tcell_vec = vpvsystem.gen_sys_Ee_Tcell_array(
                                Ee_shdarr_np.shape[0], num_str, str_len,
                                idx_map.shape[0], idx_map.shape[1],
                                Ee_arr_np, Tcell)
                            # Get unique (Ee, Tcell) at cell level
                            Ee_cell, Tcell_cell, u_cell_type, _ = vpvsystem.get_unique_Ee_Tcell(
                                Ee_vec, Tcell_vec, search_type='cell', cell_type=cell_type
                            )
                            # Get unique (Ee, Tcell) at module level
                            Ee_mod, Tcell_mod, _, cts_mod = vpvsystem.get_unique_Ee_Tcell(
                                Ee_vec, Tcell_vec, search_type='module'
                            )
                            # Get unique Ee at string level
                            Ee_str, Tcell_str, _, _ = vpvsystem.get_unique_Ee_Tcell(
                                Ee_vec, Tcell_vec, search_type='string')
                            # CELL #
                            # Extract cell prms
                            u_ctype = np.unique(cell_type)
                            pvcs = []
                            for uct in u_ctype:
                                idx_ct = np.where(cell_type == uct)
                                i_map = idx_map[idx_ct[0][0], idx_ct[1][0]]
                                pvc = copy.deepcopy(
                                    maxsys_dict['Electrical_Circuit']['PV_Module'].pvcells[i_map])
                                pvcs.append(pvc)
                            # Run 2 diode model on unique Ee
                            cfname_pre = '_'.join([plot_label, 'cell_data'])
                            vpvcell.two_diode_model(
                                pvcs, Ee_cell, u_cell_type, Tcell_cell,
                                NPTS=NPTS, NPTS_cell=NPTS_cell,
                                use_cell_NPT=use_cell_NPT,
                                fname_pre=cfname_pre, res_path=data_path)
                            # MODULE #
                            if is_sub_Mod:
                                mfname_pre = '_'.join([plot_label,
                                                       'submod_data'])
                                NPT_dict = vpvmodule.calcsubMods(
                                    cell_pos, maxmod, idx_map,
                                    Ee_mod, Tcell_mod, Ee_cell, u_cell_type,
                                    Tcell_cell, cell_type,
                                    cfname_pre=cfname_pre, res_path=data_path,
                                    mfname_pre=mfname_pre)
                            else:
                                mfname_pre = '_'.join([plot_label,
                                                       'mod_data'])
                                NPT_dict = vpvmodule.calcMods(
                                    cell_pos, maxmod, idx_map,
                                    Ee_mod, Tcell_mod, Ee_cell, Tcell_cell,
                                    u_cell_type, cell_type, outer_circuit,
                                    run_bpact=True,
                                    run_cellcurr=run_cellcurr,
                                    cfname_pre=cfname_pre,
                                    res_path=data_path, mfname_pre=mfname_pre)
                            sfname_pre = '_'.join([plot_label, 'sys_data'])
                            if is_AC_Mod:
                                # AC SYSTEM #
                                sys_data = vpvsystem.calcACSystem(
                                    Ee_vec, Tcell_vec,
                                    Ee_mod, Tcell_mod, NPT_dict,
                                    run_cellcurr=run_cellcurr,
                                    mfname_pre=mfname_pre,
                                    res_path=data_path, sfname_pre=sfname_pre)
                                if run_cellcurr:
                                    ccmod = cell_curr.est_cell_current_AC(
                                        idx_map, res_path=data_path,
                                        sfname_pre=sfname_pre)
                            else:
                                if is_sub_Mod:
                                    # SUB MODULE MPPT ###
                                    sys_data = vpvsystem.calcsubModuleSystem(
                                        Ee_vec, Ee_mod,
                                        NPT_dict,
                                        run_bpact=False,
                                        run_annual=False,
                                        save_bpact_freq=False,
                                        round_decimals=Ee_round,
                                        mfname_pre=mfname_pre,
                                        res_path=data_path,
                                        sfname_pre=sfname_pre)
                                else:
                                    # DC #
                                    # STRING #
                                    stfname_pre = '_'.join([plot_label,
                                                            'str_data'])
                                    vpvstring.calcStrings(
                                        Ee_str, Tcell_str, Ee_mod, Tcell_mod,
                                        NPT_dict,
                                        run_cellcurr=run_cellcurr,
                                        mfname_pre=mfname_pre,
                                        res_path=data_path,
                                        stfname_pre=stfname_pre)
                                    # SYSTEM #
                                    sys_data = vpvsystem.calcSystem(
                                        Ee_vec, Tcell_vec,
                                        Ee_str, Tcell_str, NPT_dict,
                                        run_cellcurr=run_cellcurr,
                                        stfname_pre=stfname_pre,
                                        res_path=data_path,
                                        sfname_pre=sfname_pre)
                                    if run_cellcurr:
                                        ccmod = cell_curr.est_cell_current_DC(
                                            sys_data,
                                            res_path=data_path,
                                            sfname_pre=sfname_pre,
                                            stfname_pre=stfname_pre,
                                            mfname_pre=mfname_pre,
                                            cell_index=idx_map)

                            dfSubCases['Pmp [W]'] = sys_data['Pmp'].tolist()
                            dfSubCases['Vmp [V]'] = sys_data['Vmp'].tolist()
                            dfSubCases['Imp [A]'] = sys_data['Imp'].tolist()
                            dfSubCases['Voc [V]'] = sys_data['Voc'].tolist()
                            dfSubCases['Isc [A]'] = sys_data['Isc'].tolist()
                            dfSubCases['FF'] = sys_data['FF'].tolist()
                            try:
                                dfSubCases['Num_BPdiode_active'] = sys_data['num_active_bpd'].tolist(
                                )
                            except AttributeError:
                                dfSubCases['Num_BPdiode_active'] = 0
                            if run_cellcurr:
                                dfSubCases['ncells_Rev_mpp'] = np.sum(ccmod['cell_isRev_mp'],
                                                                      axis=(1, 2, 3, 4)).tolist()
                                if run_isc:
                                    dfSubCases['ncells_Rev_isc'] = np.sum(ccmod['cell_isRev_sc'],
                                                                          axis=(1, 2, 3, 4)).tolist()
                            pmp0 = sys_data['Pmp'][0]
                            dfSubCases['Power change [%]'] = 100 * \
                                (dfSubCases['Pmp [W]']/pmp0 - 1)
                            dfSubCases['Isys [A]'] = sys_data['Isys'].tolist()
                            dfSubCases['Vsys [V]'] = sys_data['Vsys'].tolist()
                            dfSubCases['Psys [W]'] = sys_data['Psys'].tolist()
                            dfSubCases['sys_class'][0] = sys_data
                            # Calculate TUV Additional Loss (AL)
                            dfSubCases['AL'] = -1*dfSubCases['Power change [%]'] - \
                                dfSubCases['Mod. Shade %']
                            # Calculate the TUV class if required
                            if TUV_class:
                                TUV_out, AL_out = calc_TUV_class(
                                    dfSubCases['AL'].to_list())
                                df_subTUV = pd.DataFrame(
                                    columns=tuv_cols,
                                    index=list(range(len(TUV_out))))
                                df_subTUV['Module'] = mod_name
                                df_subTUV['Cell Name'] = cell_name
                                df_subTUV['Plot Label'] = plot_label
                                df_subTUV['Shade Type'] = ['Unshaded',
                                                           'Long-side',
                                                           'Short-side',
                                                           'Single spot',
                                                           'Multiple spots',
                                                           'Diagonal']
                                df_subTUV['AL'] = AL_out
                                df_subTUV['TUV Score'] = TUV_out
                                df_TUV = pd.concat([df_TUV, df_subTUV])
                            dfCases = pd.concat([dfCases, dfSubCases])
                            if for_gui:
                                module_dict = {}
                                module_dict['Plot_label'] = plot_label
                                module_dict['Irr'] = {}
                                module_dict['Irr']['Sim'] = Ee_vec
                                module_dict['Irr']['String'] = Ee_str
                                module_dict['Irr']['Module'] = Ee_mod
                                module_dict['Irr']['Cell'] = Ee_cell
                                # Save IV Data
                                module_dict['IV'] = {}
                                module_dict['IV']['Sim'] = sys_data

                                # Save other information
                                module_dict['Other'] = {}
                                module_dict['Other']['Cell_Index'] = maxsys_dict['Physical_Info']['Index_Map']
                                module_dict['Other']['Cell_Pos'] = maxsys_dict['Electrical_Circuit']['Cell_Postion']
                                module_dict['Other']['Shade_DF'] = df_shd_sce
                                module_dict['Other']['Module_Polygons'] = maxsys_dict['Physical_Info']['Module_Polygon']
                                module_dict['Cell_Polygons'] = maxsys_dict['Physical_Info']['Cell_Polygons']
                            if save_detailed:
                                detailed_dict[plot_label] = {}
                                # Save Irradiances
                                detailed_dict[plot_label]['Irr'] = {}
                                detailed_dict[plot_label]['Irr']['Sim'] = Ee_vec
                                detailed_dict[plot_label]['Irr']['String'] = Ee_str
                                detailed_dict[plot_label]['Irr']['Module'] = Ee_mod
                                detailed_dict[plot_label]['Irr']['Cell'] = Ee_cell
                                # Save IV Data
                                detailed_dict[plot_label]['IV'] = {}
                                detailed_dict[plot_label]['IV']['Sim'] = sys_data
                                # Save other information
                                detailed_dict[plot_label]['Other'] = {}
                                detailed_dict[plot_label]['Other']['Cell_Index'] = maxsys_dict['Physical_Info']['Index_Map']
                                detailed_dict[plot_label]['Other']['Cell_Pos'] = maxsys_dict['Electrical_Circuit']['Cell_Postion']
                                detailed_dict[plot_label]['Other']['Shade_DF'] = df_shd_sce
                                detailed_dict[plot_label]['Other']['Module_Polygons'] = maxsys_dict['Physical_Info']['Module_Polygon']
                                detailed_dict[plot_label]['Other']['Cell_Polygons'] = maxsys_dict['Physical_Info']['Cell_Polygons']
                            if for_gui:
                                save_pickle(plot_label+'.pickle', module_dict)
                            if run_cellcurr:
                                cc_mod = {}
                                cc_mod[plot_label] = copy.deepcopy(ccmod)
                                save_pickle('_'.join([plot_label, c_p_fn]),
                                            cc_mod)
                        # Delete all IV data
                        delete_files_with_substrings(folder_path=data_path,
                                                     substrings=[cfname_pre,
                                                                 mfname_pre,
                                                                 stfname_pre,
                                                                 sfname_pre],
                                                     extension='.pickle')
                        print('Time elapsed to run ' + plot_label +
                              ': ' + str(time.time() - t1) + ' s')

        save_pickle(pickle_fn, dfCases)
        dfCases_xls = dfCases.drop(
            ['Isys [A]', 'Vsys [V]', 'Psys [W]', 'sys_class'], axis=1)

        dfCases_xls.to_excel(excel_fn,
                             sheet_name='Results', index=False)
        if TUV_class:
            with pd.ExcelWriter('TUV.xlsx', engine='openpyxl') as writer:
                df_class = calc_TUV_grade(df_TUV)
                df_class.to_excel(writer,
                                  sheet_name='Classifications', index=False)
                df_TUV.to_excel(writer,
                                sheet_name='Scores', index=False)
        if save_detailed:
            save_pickle(d_p_fn, detailed_dict)
    print('Time elapsed: ' + str(time.time() - t0) + ' s')
    return dfCases


def calc_TUV_grade(df_TUV):
    """
    Calculate the TUV grades based on the Total TUV scores.

    Parameters
    ----------
    AL : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df_class = df_TUV.groupby(['Plot Label'], as_index=False).sum()
    df_class = df_class[['Plot Label', 'TUV Score']]
    df_class['TUV Class'] = np.where(df_class['TUV Score'] >= 16, 'A+',
                                     np.where(df_class['TUV Score'] >= 11, 'A',
                                     np.where(df_class['TUV Score'] >= 6, 'B',
                                              'C')))
    return df_class


def calc_TUV_class(AL):
    """
    Generate the TUV grading for shade testing.

    Parameters
    ----------
    AL : list
        Additional Loss.

    Returns
    -------
    Class_TUV : list
        TUV class for each shade type.

    """
    Class_TUV = [0]*6
    # Split ALs
    AL_long = max(AL[1], AL[2])
    AL_short = max(AL[3], AL[4])
    AL_spot1 = AL[5]
    AL_spot2 = AL[6]
    AL_diagonal = AL[7]
    AL_out = [0, AL_long, AL_short, AL_spot1, AL_spot2, AL_diagonal]
    # Long edge
    if AL_long <= 20.:
        Class_TUV[1] = 4
    elif AL_long > 20. and AL_long <= 30.:
        Class_TUV[1] = 3
    elif AL_long > 30. and AL_long <= 50.:
        Class_TUV[1] = 2
    elif AL_long > 50.:
        Class_TUV[1] = 1
    # Short edge
    if AL_short <= 25.:
        Class_TUV[2] = 4
    elif AL_short > 25. and AL_short <= 50.:
        Class_TUV[2] = 3
    elif AL_short > 50. and AL_short <= 75.:
        Class_TUV[2] = 2
    elif AL_short > 75.:
        Class_TUV[2] = 1
    # Spot
    if AL_spot1 <= 10.:
        Class_TUV[3] = 4
    elif AL_spot1 > 10. and AL_spot1 <= 20.:
        Class_TUV[3] = 3
    elif AL_spot1 > 20. and AL_spot1 <= 30.:
        Class_TUV[3] = 2
    elif AL_spot1 > 30.:
        Class_TUV[3] = 1
    # Multi-Spot
    if AL_spot2 <= 20.:
        Class_TUV[4] = 4
    elif AL_spot2 > 20. and AL_spot2 <= 25.:
        Class_TUV[4] = 3
    elif AL_spot2 > 25. and AL_spot2 <= 35.:
        Class_TUV[4] = 2
    elif AL_spot2 > 35.:
        Class_TUV[4] = 1
    # Diagonal
    if AL_diagonal <= 30.:
        Class_TUV[5] = 4
    elif AL_diagonal > 30. and AL_diagonal <= 60.:
        Class_TUV[5] = 3
    elif AL_diagonal > 60. and AL_diagonal <= 80.:
        Class_TUV[5] = 2
    elif AL_diagonal > 80.:
        Class_TUV[5] = 1

    return Class_TUV, AL_out
