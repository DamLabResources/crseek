import pandas as pd
from Bio import Alphabet
from Bio.Seq import Seq, reverse_complement
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord

from crseek import exceptions
from crseek.estimators import SequenceBase
from crseek import utils


def annotate_grna_binding(spacer, seq_record, estimator, extra_qualifiers=None,
                          exhaustive=False, mismatch_tolerance=6, openci_devices=None):
    """ In-place annotation of gRNA binding location.
    Parameters
    ----------
    spacer : Seq
        spacer to search for.
    seq_record : SeqRecord
        The sequence to search within
    estimator : SequenceBase or None
        Estimator to use to evaluate spacer binding. If None, exact string matching is used.
    extra_qualifiers : dict
        Extra qualifiers to add to the SeqFeature
    exhaustive : bool
        If True then all positions within the seq_record are checked.
        If False then a mismatch search is performed first.
    mismatch_tolerance : int
        If using a mismatch search, the tolerance.
    openci_devices : str or None
        Formatted string of device-IDs acceptable to cas-offinder. If None
        the first choice is picked from the OpenCI device list.

    Returns
    -------

    SeqRecord

    """

    exceptions._check_seq_alphabet(spacer, base_alphabet=Alphabet.RNAAlphabet)
    exceptions._check_seq_alphabet(seq_record.seq, base_alphabet=Alphabet.DNAAlphabet)

    if estimator is None:
        pos = seq_record.seq.find(spacer)
        strand = 1
        if pos == -1:
            pos = seq_record.seq.find(reverse_complement(spacer))
            strand = -1
            if pos == -1:
                raise ValueError('Could not find exact match on either strand')

        seq_record.features.append(_build_target_feature(pos, strand, spacer, score=1,
                                                         extra_quals=extra_qualifiers))
        return seq_record

    if exhaustive:
        tiles = utils.tile_seqrecord(spacer, seq_record)
    else:
        tiles = utils.cas_offinder([spacer], mismatch_tolerance, locus=[seq_record],
                                   openci_devices=openci_devices)

    pred = estimator.predict(tiles[['spacer', 'target']].values)
    pred_ser = pd.Series(pred, index=tiles.index)

    hits = pred_ser[pred_ser]

    for _, strand, left in hits.index:
        seq_record.features.append(_build_target_feature(left, strand, spacer, score=1,
                                                         extra_quals=extra_qualifiers))

    return seq_record


def _build_target_feature(left, strand, spacer, score=1, extra_quals=None):
    """
    Parameters
    ----------
    left : int
        Left most position of the binding site
    strand : int
        1 or -1 indicating the positive or negative strand
    spacer : Seq
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

    quals = {'spacer': str(spacer),
             'On Target Score': score}
    if extra_quals is not None:
        quals.update(extra_quals)

    return SeqFeature(FeatureLocation(start=left, end=left + 23, strand=strand),
                      qualifiers=quals)
