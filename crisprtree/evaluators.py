import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from Bio.Seq import Seq, reverse_complement
import interlap


def check_grna_across_seqs(grna, seqs, estimator, index=None):
    """ Simple utility function to check all sequences against a single gRNA

    Parameters
    ----------
    grna : str or Seq
        The gRNA to scan across all sequences.
    seqs : list or pd.Series
        Iterable of sequences to check
    estimator : BaseEstimator
        The estimator to use for evaluation. The estimator should already be *fit*
    index : pd.Index or list
        An index to attach to the result

    Returns
    -------

    pd.Series

    """

    checks = []
    seq_info = []

    if type(seqs) == type(pd.Series()):
        it = zip(seqs.index, seqs.values)
        index_name = seqs.index.name
    else:
        it = enumerate(seqs)
        index_name = 'SeqNum'

    if type(grna) is Seq:
        grna = str(grna)

    for seq_key, seq in it:
        if len(seq) < 23:
            # deal with short sequences gracefully
            checks.append((grna, 'X'*23))
            seq_info.append({'Index': seq_key,
                             'Position': -1,
                             'Strand': ''})
            continue

        rseq = reverse_complement(seq)
        for n in range(len(seq)-23):
            if all(l.upper() in {'A', 'C', 'G', 'T'} for l in str(seq[n:n+23])):
                checks.append((grna, seq[n:n+23]))
                seq_info.append({'Index': seq_key,
                                 'Position': n,
                                 'Strand': '+'})

            if all(l.upper() in {'A', 'C', 'G', 'T'} for l in str(rseq[n:n+23])):
                checks.append((grna, rseq[n:n+23]))
                seq_info.append({'Index': seq_key,
                                 'Position': n,
                                 'Strand': '-'})

    res = estimator.predict_proba(np.array(checks))

    targets = [targ for grna, targ in checks]

    df = pd.DataFrame(seq_info)
    df['Target'] = targets
    df['Value'] = res

    def fix_agg(rows):
        idx = rows['Value'].idxmax()
        return rows.ix[idx]


    out = df.groupby('Index')[['Value', 'Target', 'Position', 'Strand']].agg(fix_agg)
    out.index.name = index_name

    if index is not None:
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
        inter_tree.add((row['Start']+overlap, row['Stop']-overlap, ind))

    for gind, row in grna_df.iterrows():
        res = inter_tree.find((row['Start'], row['Stop']))
        index = pd.Index([ind for _, _, ind in res])

        yield row, seq_df.loc[index, :]



def positional_aggregation(seq_df, grna_df, estimator, overlap = 20):
    """ Utility function to aggregate matching results in a position-specific manner.

    Parameters
    ----------
    seq_df : pd.DataFrame
        Must contain 3 columns: Seq, Start, Stop. These must contain a Bio.Seq object, and two integers indicating
        the start and stop positions of the sequence on the reference chromosome

    grna_df : pd.DataFrame
        Must contain 2 columns: gRNA, Start, Stop. These must contain a Bio.Seq object, and two integers indicating
        the start and stop positions of the gRNA on the reference chromosome

    estimator : BaseEstimator
        The estimator to use for evaluation. The estimator should already be *fit*
    overlap: int
        How much overlap to require when comparing intervals.

    Returns
    -------

    """

    # Simple type checking

    assert _check_columns(seq_df, ['Seq', 'Start', 'Stop']), 'seq_df must contain [Seq, Start, Stop] columns'
    assert _check_columns(grna_df, ['gRNA', 'Start', 'Stop']), 'seq_df must contain [gRNA, Start, Stop] columns'

    is_seq = seq_df['Seq'].map(lambda x: type(x) is Seq)
    assert is_seq.all(), 'All items in the Seq column must be Bio.Seq objects'

    is_seq = grna_df['gRNA'].map(lambda x: type(x) is Seq)
    assert is_seq.all(), 'All items in the gRNA column must be Bio.Seq objects'

    for grna_row, seq_hits in _iterate_grna_seq_overlaps(seq_df, grna_df, overlap):
        pass