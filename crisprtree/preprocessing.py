from sklearn.base import BaseEstimator
import numpy as np


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
            encoded.append(match_encode_row(X[row, 0], X[row, 1]))

        return np.array(encoded)


class OneHotTransformer(BaseEstimator):
    """ OneHotTransformer
    This class is used to transform pairs of gRNA, Target rows into the One-Hot encoding strategy. This strategy
    represents the pair as a set of 20*16 + 1 binary elements. Each match/mismatch is encoded individually A:A, A:C, A:G
    A:T, C:A, etc.
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
            encoded.append(one_hot_encode_row(X[row, 0], X[row, 1]))

        return np.array(encoded)


def check_proto_target_input(X):
    """ Basic input parameter checking.
    Parameters
    ----------
    X : np.array
        This should be an Nx2 vector in which the first column is the gRNA
        and the second column is the target sequence (+PAM).

    Returns
    -------
    bool

    """

    assert X.shape[1] == 2

    gRNA_lens = np.array([len(val) for val in X[:,0]])
    hit_lens = np.array([len(val) for val in X[:,1]])

    assert np.all(gRNA_lens == 20)
    assert np.all(hit_lens == 23)

    return True


def match_encode_row(gRNA, target):
    """ Does the actual match-based encoding.

    Parameters
    ----------
    gRNA : str
    target : str

    Returns
    -------
    np.array

    """

    seq_order = 'ACGT'

    features = [g == l for g, l in zip(gRNA, target)]
    features.append(target[-2:] == 'GG')

    return np.array(features)



def one_hot_encode_row(gRNA, target):
    """ Does the actual one-hot encoding using a set of nested for-loops.

    Parameters
    ----------
    gRNA : str
    target : str

    Returns
    -------
    np.array

    """

    seq_order = 'ACGT'
    features = []
    for pos in range(20):
        for g in seq_order:
            for t in seq_order:
                features.append((gRNA[pos] == g) and (target[pos] == t))

    for m22 in seq_order:
        for m23 in seq_order:
            features.append((target[21] == m22) and (target[22] == m23) )

    return np.array(features)==1
