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

from .utils import save_pickle

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
                              c_p_fn='Cell_current.pickle', Ee_round=2,
                              IV_DB_loc=IV_DB_loc, IV_res=0.02,
                              IV_trk_ct=True):
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

                            if IV_DB_loc:
                                Ee_arr_np = round_to_dec(Ee_arr_np,
                                                         IV_res).round(
                                                             Ee_round)
                            # Generate Ee & Tcell array for simulation
                            Ee_vec, Tcell_vec = vpvsystem.gen_sys_Ee_Tcell_array(
                                Ee_shdarr_np.shape[0], num_str, str_len,
                                idx_map.shape[0], idx_map.shape[1],
                                Ee_arr_np, Tcell)
                            # Get unique Ee at cell level
                            Ee_cell, u_cell_type = vpvsystem.get_unique_Ee(
                                Ee_vec, search_type='cell',
                                cell_type=cell_type)
                            # Get unique Ee at module level
                            Ee_mod, _ = vpvsystem.get_unique_Ee(
                                Ee_vec, search_type='module')
                            # Get unique Ee at string level
                            Ee_str, _ = vpvsystem.get_unique_Ee(
                                Ee_vec, search_type='string')
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
                            # Define cell database names
                            if IV_DB_loc:
                                cell_DBs = [IV_res]
                            else:
                                cell_DBs = None
                            # Run 2 diode model on unique Ee
                            cell_data = vpvcell.two_diode_model(
                                pvcs, Ee_cell, u_cell_type,
                                Tcell*np.ones(Ee_cell.shape), NPTS=NPTS,
                                NPTS_cell=NPTS_cell,
                                use_cell_NPT=use_cell_NPT,
                                cell_DBs=cell_DBs, Ee_round=Ee_round)
                            NPT_dict = cell_data['NPT']
                            # MODULE #
                            # Pre-generated
                            # Define module database names
                            mod_DBs = gen_modDBs_struct(cell_name, mod_name,
                                                        NPTS, NPTS_cell,
                                                        IV_DB_loc, IV_res)
                            # Define substring database names
                            ss_DBs = gen_SSDBs_struct(cell_name, mod_name,
                                                      NPTS, NPTS_cell,
                                                      IV_DB_loc, IV_res)
                            mod_data = vpvmodule.calcMods(
                                cell_pos, maxmod, idx_map, Ee_mod, Ee_cell,
                                u_cell_type, cell_type,
                                cell_data, outer_circuit,
                                run_cellcurr=run_cellcurr, mod_DBs=mod_DBs,
                                ss_DBs=ss_DBs, Ee_round=Ee_round,
                                IV_trk_ct=IV_trk_ct)
                            if is_AC_Mod:
                                # AC SYSTEM #
                                sys_data = vpvsystem.calcACSystem(
                                    Ee_vec, Ee_mod, mod_data, NPT_dict,
                                    run_cellcurr=run_cellcurr)
                                if run_cellcurr:
                                    ccmod = cell_curr.est_cell_current_AC(
                                        sys_data, idx_map)
                            else:
                                # DC #
                                # STRING #
                                str_data = vpvstring.calcStrings(
                                    Ee_str, Ee_mod, mod_data, NPT_dict,
                                    run_cellcurr=run_cellcurr)
                                # SYSTEM #
                                sys_data = vpvsystem.calcSystem(
                                    Ee_vec, Ee_str, str_data, NPT_dict,
                                    run_cellcurr=run_cellcurr)
                                if run_cellcurr:
                                    ccmod = cell_curr.est_cell_current_DC(
                                        sys_data, str_data, mod_data, idx_map)

                            dfSubCases['Pmp [W]'] = sys_data['Pmp'].tolist()
                            dfSubCases['Vmp [V]'] = sys_data['Vmp'].tolist()
                            dfSubCases['Imp [A]'] = sys_data['Imp'].tolist()
                            dfSubCases['Voc [V]'] = sys_data['Voc'].tolist()
                            dfSubCases['Isc [A]'] = sys_data['Isc'].tolist()
                            dfSubCases['FF'] = sys_data['FF'].tolist()
                            dfSubCases['Num_BPdiode_active'] = sys_data['num_active_bpd'].tolist(
                            )
                            if run_cellcurr:
                                dfSubCases['ncells_Rev_mpp'] = np.sum(ccmod['cell_isRev_mp'],
                                                                      axis=(1, 2, 3, 4)).tolist()
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
                                dfSubCases['TUV Class'] = calc_TUV_class(
                                    dfSubCases['AL'].to_list())
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
                                if not is_AC_Mod:
                                    detailed_dict[plot_label]['IV']['String'] = str_data
                                detailed_dict[plot_label]['IV']['Module'] = mod_data
                                detailed_dict[plot_label]['IV']['Cell'] = cell_data
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
                        print('Time elapsed to run ' + plot_label +
                              ': ' + str(time.time() - t1) + ' s')

        save_pickle(pickle_fn, dfCases)
        dfCases_xls = dfCases.drop(
            ['Isys [A]', 'Vsys [V]', 'Psys [W]', 'sys_class'], axis=1)

        dfCases_xls.to_excel(excel_fn,
                             sheet_name='Results', index=False)
        if save_detailed:
            save_pickle(d_p_fn, detailed_dict)
    print('Time elapsed: ' + str(time.time() - t0) + ' s')
    return dfCases


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
    Class_TUV = ['NA']*5
    # Split ALs
    AL_long = AL[1]
    AL_short = AL[2]
    AL_spot = AL[3]
    AL_diagonal = AL[4]
    # Unshaded Class
    Class_TUV[0] = 'NA'
    # Long edge
    if AL_long <= 10.:
        Class_TUV[1] = 'A+'
    elif AL_long > 10. and AL_long <= 30.:
        Class_TUV[1] = 'A'
    elif AL_long > 30. and AL_long <= 50.:
        Class_TUV[1] = 'B'
    elif AL_long > 50.:
        Class_TUV[1] = 'C'
    # Short edge
    if AL_short <= 25.:
        Class_TUV[2] = 'A+'
    elif AL_short > 25. and AL_short <= 50.:
        Class_TUV[2] = 'A'
    elif AL_short > 50. and AL_short <= 85.:
        Class_TUV[2] = 'B'
    elif AL_short > 85.:
        Class_TUV[2] = 'C'
    # Spot
    if AL_spot <= 10.:
        Class_TUV[3] = 'A+'
    elif AL_spot > 10. and AL_spot <= 20.:
        Class_TUV[3] = 'A'
    elif AL_spot > 20. and AL_spot <= 40.:
        Class_TUV[3] = 'B'
    elif AL_spot > 40.:
        Class_TUV[3] = 'C'
    # Diagonal
    if AL_diagonal <= 30.:
        Class_TUV[4] = 'A+'
    elif AL_diagonal > 30. and AL_diagonal <= 60.:
        Class_TUV[4] = 'A'
    elif AL_diagonal > 60. and AL_diagonal <= 80.:
        Class_TUV[4] = 'B'
    elif AL_diagonal > 80.:
        Class_TUV[4] = 'C'

    return Class_TUV


def gen_modDBs_struct(cell_name, mod_name, NPTS, NPTS_cell,
                      IV_DB_loc, IV_res):
    """
    Generate module database struct.

    Parameters
    ----------
    cell_name : TYPE
        DESCRIPTION.
    mod_name : TYPE
        DESCRIPTION.
    NPTS : TYPE
        DESCRIPTION.
    NPTS_cell : TYPE
        DESCRIPTION.
    IV_DB_loc : TYPE
        DESCRIPTION.
    IV_res : TYPE
        DESCRIPTION.

    Returns
    -------
    mod_DBs : TYPE
        DESCRIPTION.

    """
    if IV_DB_loc:
        mod_irr_db_name = '_'.join(['pvmodule', 'Irr',
                                    mod_name, cell_name, 'NPTS',
                                    str(NPTS),
                                    'NPTS_cell',
                                    str(NPTS_cell)]
                                   ) + '.pickle'
        mod_isc_db_name = '_'.join(['pvmodule', 'Isc',
                                    mod_name, cell_name, 'NPTS',
                                    str(NPTS),
                                    'NPTS_cell',
                                    str(NPTS_cell)]
                                   ) + '.pickle'
        mod_I_db_name = '_'.join(['pvmodule', 'I',
                                  mod_name, cell_name, 'NPTS',
                                  str(NPTS),
                                  'NPTS_cell',
                                  str(NPTS_cell)]
                                 ) + '.pickle'
        mod_V_db_name = '_'.join(['pvmodule', 'V',
                                  mod_name, cell_name, 'NPTS',
                                  str(NPTS),
                                  'NPTS_cell',
                                  str(NPTS_cell)]
                                 ) + '.pickle'
        mod_Isubstr_db_name = '_'.join(['pvmodule',
                                        'Isubstr',
                                        mod_name, cell_name, 'NPTS',
                                        str(NPTS),
                                        'NPTS_cell',
                                        str(NPTS_cell)]
                                       ) + '.pickle'
        mod_Vsubstr_db_name = '_'.join(['pvmodule',
                                        'Vsubstr',
                                        mod_name, cell_name, 'NPTS',
                                        str(NPTS),
                                        'NPTS_cell',
                                        str(NPTS_cell)]
                                       ) + '.pickle'
        mod_Isubstr_pbp_db_name = '_'.join(['pvmodule',
                                            'Isubstr_pbp',
                                            mod_name, cell_name,
                                            'NPTS',
                                            str(NPTS),
                                            'NPTS_cell',
                                            str(NPTS_cell)
                                            ]) + '.pickle'
        mod_Vsubstr_pbp_db_name = '_'.join(['pvmodule',
                                            'Vsubstr_pbp',
                                            mod_name, cell_name,
                                            'NPTS',
                                            str(NPTS),
                                            'NPTS_cell',
                                            str(NPTS_cell)
                                            ]) + '.pickle'
        mod_bypassed_db_name = '_'.join(['pvmodule',
                                         'bypassed',
                                         mod_name, cell_name,
                                         'NPTS',
                                         str(NPTS),
                                         'NPTS_cell',
                                         str(NPTS_cell)]
                                        ) + '.pickle'
        mod_counts_db_name = '_'.join(['pvmodule',
                                       'counts',
                                       mod_name, cell_name,
                                       'NPTS',
                                       str(NPTS),
                                       'NPTS_cell',
                                       str(NPTS_cell)]
                                      ) + '.pickle'
        mod_DBs = [os.path.join(IV_DB_loc,
                                mod_irr_db_name),
                   os.path.join(IV_DB_loc,
                                mod_isc_db_name),
                   os.path.join(IV_DB_loc,
                                mod_I_db_name),
                   os.path.join(IV_DB_loc,
                                mod_V_db_name),
                   os.path.join(IV_DB_loc,
                                mod_Isubstr_db_name),
                   os.path.join(IV_DB_loc,
                                mod_Vsubstr_db_name),
                   os.path.join(IV_DB_loc,
                                mod_Isubstr_pbp_db_name),
                   os.path.join(IV_DB_loc,
                                mod_Vsubstr_pbp_db_name),
                   os.path.join(IV_DB_loc,
                                mod_bypassed_db_name),
                   os.path.join(IV_DB_loc,
                                mod_counts_db_name),
                   IV_res]
    else:
        mod_DBs = None
    return mod_DBs


def gen_SSDBs_struct(cell_name, mod_name, NPTS, NPTS_cell,
                     IV_DB_loc, IV_res):
    """
    Generate module substring database struct.

    Parameters
    ----------
    cell_name : TYPE
        DESCRIPTION.
    mod_name : TYPE
        DESCRIPTION.
    NPTS : TYPE
        DESCRIPTION.
    NPTS_cell : TYPE
        DESCRIPTION.
    IV_DB_loc : TYPE
        DESCRIPTION.
    IV_res : TYPE
        DESCRIPTION.

    Returns
    -------
    mod_DBs : TYPE
        DESCRIPTION.

    """
    if IV_DB_loc:
        SS_irr_db_name = '_'.join(['pvsubstr', 'Irr',
                                   mod_name, cell_name, 'NPTS',
                                   str(NPTS),
                                   'NPTS_cell',
                                   str(NPTS_cell)]
                                  ) + '.pickle'
        SS_Id_pre_db_name = '_'.join(['pvsubstr', 'Id_pre',
                                      mod_name, cell_name, 'NPTS',
                                      str(NPTS),
                                      'NPTS_cell',
                                      str(NPTS_cell)]
                                     ) + '.pickle'
        SS_Vd_pre_db_name = '_'.join(['pvsubstr', 'Vd_pre',
                                      mod_name, cell_name, 'NPTS',
                                      str(NPTS),
                                      'NPTS_cell',
                                      str(NPTS_cell)]
                                     ) + '.pickle'
        SS_cts_pre_db_name = '_'.join(['pvsubstr', 'counts',
                                      mod_name, cell_name, 'NPTS',
                                      str(NPTS),
                                      'NPTS_cell',
                                       str(NPTS_cell)]
                                      ) + '.pickle'
        SS_irr_ss_db_name = '_'.join(['pvsubstr', 'Irr_ss',
                                      mod_name, cell_name, 'NPTS',
                                      str(NPTS),
                                      'NPTS_cell',
                                      str(NPTS_cell)]
                                     ) + '.pickle'
        SS_Iss_pre_db_name = '_'.join(['pvsubstr', 'Iss',
                                       mod_name, cell_name, 'NPTS',
                                       str(NPTS),
                                       'NPTS_cell',
                                       str(NPTS_cell)]
                                      ) + '.pickle'
        SS_Vss_pre_db_name = '_'.join(['pvsubstr', 'Vss',
                                       mod_name, cell_name, 'NPTS',
                                       str(NPTS),
                                       'NPTS_cell',
                                       str(NPTS_cell)]
                                      ) + '.pickle'
        SS_cts_ss_db_name = '_'.join(['pvsubstr', 'counts_ss',
                                      mod_name, cell_name, 'NPTS',
                                      str(NPTS),
                                      'NPTS_cell',
                                      str(NPTS_cell)]
                                     ) + '.pickle'
        SS_DBs = [os.path.join(IV_DB_loc,
                               SS_irr_db_name),
                  os.path.join(IV_DB_loc,
                               SS_Id_pre_db_name),
                  os.path.join(IV_DB_loc,
                               SS_Vd_pre_db_name),
                  os.path.join(IV_DB_loc,
                               SS_cts_pre_db_name),
                  os.path.join(IV_DB_loc,
                               SS_irr_ss_db_name),
                  os.path.join(IV_DB_loc,
                               SS_Iss_pre_db_name),
                  os.path.join(IV_DB_loc,
                               SS_Vss_pre_db_name),
                  os.path.join(IV_DB_loc,
                               SS_cts_ss_db_name),
                  IV_res]
    else:
        SS_DBs = None
    return SS_DBs
