import yaml
import pandas as pd
import numpy as np
import os
import yaml

this_dir, this_filename = os.path.split(os.path.abspath(__file__))
DATA_PATH = os.path.join(this_dir, '..', "data")


def load_mismatch_scores(name_or_path):
    """ Loads a formatted penalty matrix.
    Parameters
    ----------
    name_or_path : str
        Specific named reference or path/to/yaml definition

    Returns
    -------

    matrix : pd.DataFrame
        Scores for specific binding pairs
    pams : pd.Series
        Encoding for the second 2 NT in the PAM


    """

    if name_or_path == 'CFD':
        path = os.path.join(DATA_PATH, 'models', 'CFD.yaml')
    else:
        path = name_or_path

    with open(path) as handle:
        obj = next(yaml.load_all(handle))

    matrix = pd.DataFrame(obj['scores'])
    matrix = matrix.reindex(columns=sorted(matrix.columns))

    pams = pd.Series(obj['pams'])
    pams = pams.reindex(sorted(pams.index))

    return matrix, pams

