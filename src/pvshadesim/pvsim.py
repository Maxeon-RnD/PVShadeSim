# -*- coding: utf-8 -*-
"""Run the shading simulation."""

from pathlib import Path
import os
import time

import pandas as pd

from .db import database
from .pvmod import create_pvmod_dict
from .pvshade import gen_shade_scenarios
from .pvelectric import gen_pvmmvec_shade_results

# Default databases
def_cell_db = r'PVMM_cell_params_DB.csv'
def_mod_db = r'PVMM_mod_params_DB.csv'
def_shd_db = r'PVMM_shade_params_DB.csv'
def_user_cell_idx_f = r'User_cell_index_maps.xlsx'
def_user_cell_pos_f = r'User_cell_pos.xlsx'
sim_config_csv = r'Sim_Config.csv'
db_path = os.path.join(Path(__file__).parent, 'db')


def run(cell_prm_csv=os.path.join(db_path, def_cell_db),
        mod_prm_csv=os.path.join(db_path, def_mod_db),
        shade_prm_csv=os.path.join(db_path, def_shd_db),
        cell_idx_xls=os.path.join(db_path, def_user_cell_idx_f),
        cell_pos_xls=os.path.join(db_path, def_user_cell_pos_f),
        sim_config_csv=os.path.join(db_path, sim_config_csv),
        NPTS=1500, NPTS_cell=100, use_cell_NPT=False,
        Tcell=298.15, irrad_suns=1,
        gen_mod_idx=False, search_idx_name='scenario_definition',
        gen_sh_sce=False, gen_sh_arr=False,
        pickle_fn='Gen_PVMM_Vectorized_Shade_Results.pickle',
        save_detailed=False, TUV_class=False, for_gui=False,
        excel_fn="PVMM_Vectorized_Shade_Simulation_Results.xlsx",
        d_p_fn='Detailed_Data.pickle',
        run_cellcurr=True, c_p_fn='Cell_current.pickle',
        Ee_round=2):
    """
    Run the entire PVShadeSim process.

    Parameters
    ----------
    cell_prm_csv : str, optional
        PV cell database file path.
        The default is os.path.join(db_path, def_cell_db).
    mod_prm_csv : str, optional
        PV module database file path.
        The default is os.path.join(db_path, def_mod_db).
    shade_prm_csv : str, optional
        Shade scenarios database file path.
        The default is os.path.join(db_path, def_shd_db).
    cell_idx_xls : str, optional
        User defined cell index maps file path.
        The default is os.path.join(db_path, def_user_cell_idx_f).
    cell_pos_xls : str, optional
        User defined cell positions file path.
        The default is os.path.join(db_path, def_user_cell_pos_f).
    sim_config_csv : str, optional
        File path for the simulation configuration.
        The default is os.path.join(db_path, sim_config_csv).
    NPTS : int, optional
        Number of points in IV curve. The default is 1500.
    NPTS_cell : int, optional
        Number of points in cell IV curve. The default is 100.
    use_cell_NPT : bool, optional
        Use separate NPTS_cell parameter. The default is False.
    Tcell : float, optional
        Nominal cell temperature in kelvin. The default is 298.15.
    irrad_suns : float, optional
        Nominal irradiance in suns. The default is 1.
    gen_mod_idx : bool, optional
        Generate images of the cell position index within the module.
        The default is False.
    search_idx_name : str
        Which index of database to search for the filter.
    gen_sh_sce : bool, optional
        Generate plot of shade scenarios? The default is False.
    gen_sh_arr : bool, optional
        Generate plot of cell intersection arrays with shade scenarios.
        The default is False.
    pickle_fn : str, optional
        Pickle file containing all the detailed results.
        The default is 'Gen_PVMM_Vectorized_Shade_Results.pickle'.
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
    # Load databases
    dbs = database.import_db(cell_prm_csv, mod_prm_csv, shade_prm_csv,
                             cell_idx_xls, cell_pos_xls)
    pvcell_params, pvmod_params, pvshade_params, cell_idx_xls, cell_pos_xls = dbs

    # Load Simulation configuration file
    sim_config = pd.read_csv(sim_config_csv, index_col=0).T

    # Generate PV Module physical and electrical models data structure
    t0 = time.time()
    mods_sys_dict = create_pvmod_dict(pvmod_params, sim_config, pvcell_params,
                                      cell_idx_xls, cell_pos_xls,
                                      gen_mod_idx=gen_mod_idx,
                                      NPTS=NPTS, Tcell=Tcell)
    print('Time to generate Module Models: ' + str(time.time() - t0) + ' s')

    # Generate required shade scenarios
    t0 = time.time()
    mods_sys_dict = gen_shade_scenarios(mods_sys_dict, pvshade_params,
                                        search_idx_name=search_idx_name,
                                        gen_sh_sce=gen_sh_sce,
                                        gen_sh_arr=gen_sh_arr)
    print('Time to generate Shade Scenarios: ' + str(time.time() - t0) + ' s')

    # Run the electrical model
    dfCases = gen_pvmmvec_shade_results(mods_sys_dict,
                                        pickle_fn=pickle_fn,
                                        irrad_suns=irrad_suns, Tcell=Tcell,
                                        NPTS=NPTS, NPTS_cell=NPTS_cell,
                                        use_cell_NPT=use_cell_NPT,
                                        save_detailed=save_detailed,
                                        TUV_class=TUV_class,
                                        for_gui=for_gui, excel_fn=excel_fn,
                                        d_p_fn=d_p_fn,
                                        run_cellcurr=run_cellcurr,
                                        c_p_fn=c_p_fn, Ee_round=Ee_round)
    return dfCases
