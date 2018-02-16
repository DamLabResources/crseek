from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import numpy as np
import pandas as pd

from crisprtree.utils import tile_seqrecord, cas_offinder
from crisprtree.estimators import SequenceBase


def annotate_grna_binding(grna, seq_record, estimator, extra_qualifiers=None,
                          exhaustive = False, mismatch_tolerance = 6):
    """ In-place annotation of gRNA binding location.
    Parameters
    ----------
    grna : str
        gRNA to search for.
    seq_record : SeqRecord
        The sequence to search within
    estimator : SequenceBase
        Estimator to use to evaluate gRNA binding
    extra_qualifiers : dict
        Extra qualifiers to add to the SeqFeature
    exhaustive : bool
        If True then all positions within the seq_record are checked.
        If False then a mismatch search is performed first.
    mismatch_tolerance : int
        If using a mismatch search, the tolerance.

    Returns
    -------

    SeqRecord

    """

    if exhaustive:
        tiles = tile_seqrecord(grna, seq_record)
    else:
        tiles = cas_offinder([grna], mismatch_tolerance, seqs = [seq_record])

    pred = estimator.predict(tiles.values)
    pred_ser = pd.Series(pred, index=tiles.index)

    hits = pred_ser[pred_ser]

    for _, strand, left in hits.index:
        seq_record.features.append(_build_target_feature(left, strand, grna, score=1,
                                                         extra_quals = extra_qualifiers))

    return seq_record


def _build_target_feature(left, strand, grna, score=1, extra_quals = None):
    """
    Parameters
    ----------
    left : int
        Left most position of the binding site
    strand : int
        1 or -1 indicating the positive or negative strand
    grna : str
        gRNA that's targetted to this location
    score : float
        Binding score of the gRNA to this location
    extra_quals : dict
        Extra qualifiers to add to the SeqFeature

    Returns
    -------

    SeqFeature

    """

    if strand not in {-1, 1}:
        raise ValueError('Strand must be {1, -1}')

    quals = {'gRNA': grna,
             'On Target Score': score}
    if extra_quals is not None:
        quals.update(extra_quals)

    return SeqFeature(FeatureLocation(start=left, end=left+23, strand=strand),
                      qualifiers = quals)