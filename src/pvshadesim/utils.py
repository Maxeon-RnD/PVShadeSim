# -*- coding: utf-8 -*-
"""Helper functions."""

import pickle


def create_letter_list(num_x):
    """
    Create a list of ordered upper case letters provided a number.

    Eg. 1) num_x = 1 --> O/P = [A]
        2) num_x = 2 --> O/P = [A, B]

    Parameters
    ----------
    num_x : int
        Number of letters to return.

    Returns
    -------
    letter_list : list
        List of upper case letters.

    """
    letter_list = []
    for num in range(num_x):
        letter_list.append(chr(ord('@')+num+1))
    return letter_list


def find_middle(input_list):
    """
    Find middle index of a list. If it is odd, then the output is trivial.

    If it is even then the first of two middle indices is returned.

    Parameters
    ----------
    input_list : list
        Input list containing number of cells per diode.

    Returns
    -------
    int
        Middle index of list.

    """
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return int(middle - .5)
    else:
        return int(middle-1)


def ranges(N, nb):
    """
    Generate list of ranges given N points and the end point.

    Parameters
    ----------
    N : int
        Number of points.
    nb : float
        End point.

    Returns
    -------
    list
        DESCRIPTION.

    """
    step = N / nb
    return [[round(step*i), round(step*(i+1))] for i in range(nb)]


def save_pickle(filename, variable):
    """
    Save pickle file.

    Parameters
    ----------
    filename : str
        File path.
    variable : python variable
        Variable to save to pickle file.

    Returns
    -------
    None.

    """
    with open(filename, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """
    Load data from a pickle file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    db : python variable
        Variable that is stored in the pickle file.

    """
    with open(filename, 'rb') as handle:
        db = pickle.load(handle)
    return db
