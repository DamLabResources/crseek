import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from Bio.Seq import reverse_complement


def check_grna_across_seqs(grna, seqs, estimator, aggfunc='max', index=None):
    """ Simple utility function to check all sequences against a single gRNA

    Parameters
    ----------
    grna : str
        The gRNA to scan across all sequences.
    seqs : list or pd.Series
        Iterable of sequences to check
    estimator : BaseEstimator
        The estimator to use for evaluation. The estimator shuold already be *fit*
    aggfunc : str or func
        How to aggregate the results from the same sequence. Anything understood by pandas.agg
    index : pd.Index
        An index to attach to the result

    Returns
    -------

    pd.Series

    """

    checks = []
    orig_place = []

    if type(seqs) == type(pd.Series()):
        it = zip(seqs.index, seqs.values)
    else:
        it = enumerate(seqs)

    for seq_key, seq in it:
        if len(seq) < 23:
            # deal with short sequences gracefully
            checks.append((grna, 'X'*23))
            orig_place.append(seq_key)
            continue

        rseq = reverse_complement(seq)
        for n in range(len(seq)-23):
            checks.append((grna, seq[n:n+23]))
            checks.append((grna, rseq[n:n+23]))
            orig_place += [seq_key, seq_key]

    res = estimator.predict(np.array(checks))

    out = pd.DataFrame({'SeqNum': orig_place, 'Value': res}).groupby('SeqNum')['Value'].agg(aggfunc)
    return out

