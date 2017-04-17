from sklearn.base import BaseEstimator
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import numpy as np
import pandas as pd


def annotate_gRNA_binding(grna, seq_record, estimator, extra_qualifiers=None):
    """ In-place annotatio of gRNA binding location.
    Parameters
    ----------
    grna : str
        gRNA to search for.
    seq_record : SeqRecord
        The sequence to search within
    estimator : BaseEstimator
        Estimator to use to evaluate gRNA binding
    extra_qualifiers : dict
        Extra qualifiers to add to the SeqFeature

    Returns
    -------

    SeqRecord

    """

    pass



def _build_target_feature(start, strand, grna, score=1, extra_quals = None):

    if strand not in {-1, 1}:
        raise ValueError('Strand must be {1, -1}')

    end = start + 23 if strand == 1 else start - 23
    assert end >= 0, 'Cannot have a SeqFeature that goes BEFORE the start location'

    quals = {'gRNA': grna,
             'On Target Score': score}
    if extra_quals is not None:
        quals.update(extra_quals)

    return SeqFeature(FeatureLocation(start=start, end=end, strand=strand),
                      qualifiers = quals)