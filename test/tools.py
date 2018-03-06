import numpy as np


def make_random_seq(bp, restricted = 'AT', alphabet = None):
    """ Utility function for making random sequence
    Parameters
    ----------
    restricted
    alphabet
    bp : int
        Length of sequence

    Returns
    -------
    str

    """
    return ''.join(np.random.choice(list('AT'), size=bp))