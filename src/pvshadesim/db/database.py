# -*- coding: utf-8 -*-
"""Function to load default or user specified databases."""

import pandas as pd


def import_db(cell_prm_csv, mod_prm_csv, shade_prm_csv, cell_idx_xls, cell_pos_xls):
    """
    Import all the databases allowing for either using default DBs or user inputted DBs.

    This is just a wrapper function.

    Parameters
    ----------
    cell_prm_csv : str, optional
        PV cell database file path.
    mod_prm_csv : str, optional
        PV module database file path.
    shade_prm_csv : str, optional
        Shade scenarios database file path.
    cell_idx_xls : str, optional
        User defined cell index maps file path.
    cell_pos_xls : str, optional
        User defined cell positions file path.

    Returns
    -------
    pvcell_params : pandas.DataFrame
        PV cell database.
    pvmod_params : pandas.DataFrame
        PV module database.
    pvshade_params : pandas.DataFrame
        Shade scenarios database.
    cell_idx_xls : str, optional
        User defined cell index maps file path.
    cell_pos_xls : str, optional
        User defined cell positions file path.

    """
    # Extract cell prm data
    pvcell_params = pd.read_csv(cell_prm_csv, index_col=0).T

    # Extract module prm data
    pvmod_params = pd.read_csv(mod_prm_csv, index_col=0).T

    # Extract shade prm data
    pvshade_params = pd.read_csv(shade_prm_csv, index_col=0).T

    return (pvcell_params, pvmod_params, pvshade_params, cell_idx_xls, cell_pos_xls)
