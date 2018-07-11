from itertools import cycle

import interlap
import numpy as np
import pandas as pd
from Bio import Alphabet
from Bio.Seq import Seq
from sklearn.base import BaseEstimator

from crseek import exceptions
from crseek.preprocessing import locate_hits_in_array
from crseek.utils import _make_record_key

def check_spacer_across_loci(spacer, loci, estimator, index=None):
    """ Simple utility function to check all sequences against a single gRNA

    Parameters
    ----------
    spacer : str or Seq
        The gRNA to scan across all sequences.
    loci : iter[SeqRecord]
        Iterable of sequences to check
    estimator : BaseEstimator
        The estimator to use for evaluation. The estimator should already be *fit*
    index : pd.Index or list
        An index to attach to the result

    Returns
    -------

    pd.Series

    """

    X = np.array(list(zip(cycle([spacer]), loci)))
    if type(loci) == pd.Series:
        index = loci.index

    if index is None:
        index = [_make_record_key(locus) for locus in loci]
    resX, resL, resS = locate_hits_in_array(X, estimator=estimator)

    out = pd.concat([pd.DataFrame(resL, columns=['left', 'strand']),
                     pd.DataFrame(resX, columns=['spacer', 'target'])],
                    axis=1)
    out['score'] = resS
    out.index = index

    return out


def _check_columns(df, columns):
    """ Utility function to test for columns in dataframe

    Parameters
    ----------
    df : pd.DataFrame
    columns : list

    Returns
    -------
    bool
    """

    for col in columns:
        if col not in df.columns:
            return False
    return True


def _iterate_grna_seq_overlaps(seq_df, grna_df, overlap):
    inter_tree = interlap.InterLap()

    for ind, row in seq_df.iterrows():
        inter_tree.add((row['Start'] + overlap, row['Stop'] - overlap, ind))

    for gind, row in grna_df.iterrows():
        res = inter_tree.find((row['Start'], row['Stop']))
        index = pd.Index([ind for _, _, ind in res])
        if len(index) == 0:
            continue

        yield row, seq_df.loc[index, :]


def positional_aggregation(loci_df, spacer_df, estimator, overlap=20):
    """ Utility function to aggregate matching results in a position-specific manner.

    Parameters
    ----------
    loci_df : pd.DataFrame
        Must contain 3 columns: Seq, Start, Stop. These must contain a Bio.Seq object, and two integers indicating
        the start and stop positions of the sequence on the reference chromosome

    spacer_df : pd.DataFrame
        Must contain 2 columns: spacer, Start, Stop. These must contain a Bio.Seq object, and two integers indicating
        the start and stop positions of the gRNA on the reference chromosome

    estimator : BaseEstimator
        The estimator to use for evaluation. The estimator should already be *fit*
    overlap: int
        How much overlap to require when comparing intervals.

    Returns
    -------

    """

    # Simple type checking

    assert _check_columns(loci_df, ['Seq', 'Start', 'Stop']), 'seq_df must contain [Seq, Start, Stop] columns'
    assert _check_columns(spacer_df, ['spacer', 'Start', 'Stop']), 'seq_df must contain [spacer, Start, Stop] columns'

    _ = [exceptions._check_seq_alphabet(seq, base_alphabet=Alphabet.RNAAlphabet) for seq in spacer_df['spacer'].values]
    _ = [exceptions._check_seq_alphabet(seqR.seq, base_alphabet=Alphabet.DNAAlphabet) for seqR in loci_df['Seq'].values]

    all_hits = []
    for grna_row, seq_hits in _iterate_grna_seq_overlaps(loci_df, spacer_df, overlap):

        name_index = [row['Seq'].id + ' ' + row['Seq'].description for _, row in seq_hits.iterrows()]
        all_hits.append(check_spacer_across_loci(grna_row['spacer'],
                                                 seq_hits['Seq'].values,
                                                 estimator,
                                                 index=seq_hits.index))

        if len(all_hits[-1].index) > 0:
            for col in grna_row.index:
                if col not in {'Start', 'Stop', 'spacer'}:
                    all_hits[-1][col] = grna_row[col]
            all_hits[-1]['spacer'] = [grna_row['spacer']] * len(all_hits[-1].index)

            for col in seq_hits.columns:
                if col not in {'Start', 'Stop', 'Seq'}:
                    all_hits[-1].loc[seq_hits.index, col] = seq_hits.loc[seq_hits.index, col]

    try:
        return pd.concat(all_hits, axis=0, ignore_index=True)
    except ValueError:
        return None
