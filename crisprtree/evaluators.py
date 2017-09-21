import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from Bio.Seq import reverse_complement


def check_grna_across_seqs(grna, seqs, estimator, index=None):
    """ Simple utility function to check all sequences against a single gRNA

    Parameters
    ----------
    grna : str
        The gRNA to scan across all sequences.
    seqs : list or pd.Series
        Iterable of sequences to check
    estimator : BaseEstimator
        The estimator to use for evaluation. The estimator should already be *fit*
    index : pd.Index
        An index to attach to the result

    Returns
    -------

    pd.Series

    """

    checks = []
    seq_info = []
    orig_place = []
    orig_position = []
    orig_strand = []

    if type(seqs) == type(pd.Series()):
        it = zip(seqs.index, seqs.values)
        index_name = seqs.index.name
    else:
        it = enumerate(seqs)
        index_name = 'SeqNum'

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


def positional_aggregation():
    pass