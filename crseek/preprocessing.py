from itertools import product
import numpy as np
import pandas as pd
from Bio.Alphabet import generic_dna, generic_rna, RNAAlphabet, DNAAlphabet, IUPAC
from Bio.Seq import Seq
from sklearn.base import BaseEstimator
from crseek import exceptions
from crseek import utils


class MatchingTransformer(BaseEstimator):
    """ MatchingTransformer
    This class is used to transform pairs of gRNA, Target rows into the simple matching encoding strategy. This strategy
    represents the pair as a set of 20 + 1 binary elements. Each position is simply scored as "matching" or not. The 21st
    position represents the NGG PAM site.
        """

    def fit(self, X, y):
        """ fit
        In this context nothing happens.
        Parameters
        ----------
        X : np.array
        y : np.array

        Returns
        -------
        OneHotTransformer

        """
        return self

    def transform(self, X):
        """ transform
        Transforms the array into the one-hot encoded format.

        Parameters
        ----------
        X : np.array
            This should be an Nx2 vector in which the first column is the gRNA
            and the second column is the target sequence (+PAM).

        Returns
        -------
        np.array

        """

        check_proto_target_input(X)

        encoded = []
        for row in range(X.shape[0]):
            encoded.append(match_encode_row(X[row, 0].upper(), X[row, 1].upper()))

        return np.array(encoded)


class OneHotTransformer(BaseEstimator):
    """ OneHotTransformer
    This class is used to transform pairs of gRNA, Target rows into the One-Hot encoding strategy. This strategy
    represents the pair as a set of 20*16 + 1 binary elements. Each match/mismatch is encoded individually A:A, A:C, A:G
    A:T, C:A, etc.
        """

    def __init__(self, spacer_alphabet=IUPAC.unambiguous_rna, target_alphabet=IUPAC.unambiguous_dna):
        self.spacer_alphabet = spacer_alphabet
        self.target_alphabet = target_alphabet

    def fit(self, X, y):
        """ fit
        In this context nothing happens.
        Parameters
        ----------
        X : np.array
        y : np.array

        Returns
        -------
        OneHotTransformer

        """
        return self

    def transform(self, X):
        """ transform
        Transforms the array into the one-hot encoded format.

        Parameters
        ----------
        X : np.array
            This should be an Nx2 vector in which the first column is the gRNA
            and the second column is the target sequence (+PAM).

        Returns
        -------
        np.array

        """

        check_proto_target_input(X)

        encoded = []
        for row in range(X.shape[0]):
            encoded.append(one_hot_encode_row(X[row, 0].upper(), X[row, 1].upper(),
                                              spacer_alphabet=self.spacer_alphabet,
                                              target_alphabet=self.target_alphabet))

        return np.array(encoded)


def locate_hits_in_array(X, estimator, exhaustive=False, mismatches=6, openci_devices=None):
    """ Utilizes cas-offinder to find the likeliest hit of the gRNA in a long
    sequence. It uses the provided estimator to rank each potential hit.

    Parameters
    ----------
    X : np.array
        This should be an Nx2 array in which the first column is the spacer
        and the second column is the locus.

    estimator : SequenceBase
        Estimator to use when scoring potential hits

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

    X : np.array
        An Nx2 array in which the first column is the gRNA and the second
        column is the target sequence (+PAM). Suitable for use with other tools.
    loc : np.array
        The position in each array of the maximal target hit. An Nx2 array in
        which the first column is the start and the second is the strand.
    scores: np.array
        The score returned by the estimator for this binding.

    """

    def pick_best(df):
        best = df['score'].idxmax()
        return df.loc[best, :]

    seqs = list(X[:, 1])
    seq_ids = [utils._make_record_key(s) for s in X[:, 1]]
    spacers = np.unique(X[:, 0])

    if exhaustive:
        result = pd.concat([utils.tile_seqrecord(spacer, seq) for seq in seqs for spacer in spacers],
                           axis=0)
    else:
        result = utils.cas_offinder(spacers, mismatches, locus=seqs,
                                    openci_devices=openci_devices)

    if len(result.index) > 0:
        result['score'] = estimator.predict_proba(result.values)
    else:
        result['score'] = []

    best_hits = result.reset_index().groupby('name').agg(pick_best)
    best_hits = best_hits.reindex(seq_ids)

    X = best_hits[['spacer', 'target']]
    loc = best_hits[['left', 'strand']]
    scores = best_hits['score']

    return X.values, loc.values, scores.values


def check_proto_target_input(X):
    """ Basic input parameter checking.
    Parameters
    ----------
    X : np.array
        This should be an Nx2 vector in which the first column is the spacer
        and the second column is the target.

    Returns
    -------
    bool

    """

    assert X.shape[1] == 2

    spacer_lens = np.array([len(val) for val in X[:, 0]])
    target_lens = np.array([len(val) for val in X[:, 1]])

    assert np.all(spacer_lens == 20)
    assert np.all(target_lens == 23)

    _ = [exceptions._check_seq_alphabet(spacer, base_alphabet=RNAAlphabet) for spacer in X[:, 0]]
    _ = [exceptions._check_seq_alphabet(target, base_alphabet=DNAAlphabet) for target in X[:, 1]]

    return True


def match_encode_row(spacer, target):
    """ Does the actual match-based encoding.

    Parameters
    ----------
    spacer : Seq
    target : Seq

    Returns
    -------
    np.array

    """

    # TODO: Deal with different PAMs

    features = [g == l for g, l in zip(spacer.back_transcribe(), target)]
    features.append(target[-2:] == 'GG')

    return np.array(features)


def one_hot_encode_row(spacer, target,
                       spacer_alphabet=IUPAC.unambiguous_rna,
                       target_alphabet=IUPAC.unambiguous_dna):
    """ Does the actual one-hot encoding using a set of nested for-loops.

    Parameters
    ----------
    spacer : Seq
    target : Seq
    spacer_alphabet : RNAAlphabet
    target_alphabet : DNAAlphabet

    Returns
    -------
    np.array

    """

    spacer_order = sorted(spacer_alphabet.letters)
    target_order = sorted(target_alphabet.letters)
    features = []
    for pos in range(20):
        for g in spacer_order:
            for t in target_order:
                features.append((spacer[pos] == g) and (target[pos] == t))

    for m22, m23 in product(target_order, repeat=2):
        features.append((target[21] == m22) and (target[22] == m23))

    feats = np.array(features) == 1
    assert feats.sum() == 21, 'Nonstandard nucleotide detected.'

    return feats
