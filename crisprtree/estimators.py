from sklearn.base import BaseEstimator


class MITEstimator(BaseEstimator):

    def __init__(self, dampening=False, cutoff = 0.75):
        self.dampening=dampening
        self.cutoff = cutoff

    def fit(self, X, y=None):
        return self

    def predict(self, X):

        pass

    def predict_probs(self, X):

        pass



def check_matching_input(X):
    """
    Parameters
    ----------
    X : np.array
        Must be an Nx21 of booleans

    Returns
    -------
    bool

    """

    assert X.shape[1] == 21
    #assert np.




def MIT_penalty_score(matches, dampening=False):
    """Score gRNA hit

    This scores the gRNA hit based on the method described by Hsu et al
    as implemented in http://crispr.mit.edu/about.

    This takes into account that mutations at PAM-distal sites have
    minimal effects while those in the middle and PAM-proximal sites have
    more effect on binding. With the `dampening` flag sequences with more
    mismatches are more drastically penalized.

    As such, it is important to provide the gRNA and hit_seq in the proper
    orientation. If the gRNA is arranged like this:

    ATG ATG ATG ATC ATC ATC T NGG <- PAM

    Then it is in the correct orientation. If instead the gRNA is arranged
    like this:

    GGN ATG ATG ATG ATC ATC ATC T

    Then both the gRNA and hit_seq need to be reversed. Use the gRNA[::-1]
    idiom to do so easily.

    Parameters
    ----------

    matches : str
        This should be the protospacer region of the gRNA. Do not include the
        PAM region. The PAM is assumed to immediately follow this region.

    hit_seq : str
        This is the potential binding site of the protospacer region of the
        gRNA. The PAM is assumed to immediately follow this region.

    dampening : bool, optional
        If True, this implements the extra penality for high-mismatch items.
        This is often done when search for genomic off-targets but not in
        the desired region.

    Returns
    -------

    score : float
        The likelihood of binding as estimated by the Hsu et al method.

    """

    penalties = [0, 0, 0.014, 0, 0, 0.395, 0.317, 0,
                 0.389, 0.079, 0.445, 0.508, 0.613,
                 0.851, 0.732, 0.828, 0.615, 0.804,
                 0.685, 0.583]

    #Do some error checking
    for n, s in [('gRNA', guide_seq), ('hit', hit_seq)]:
        if len(s) != len(penalties):
            msg = 'len(%s) != %i, got %i' % (n, len(penalties),
                                             len(guide_seq))
            raise ValueError(msg)

    score = 1.0
    nmiss = 0
    miss_sum = 0.0
    for p, g, h in zip(penalties,
                       seq_to_str(guide_seq),
                       seq_to_str(hit_seq)):
        if g != h:
            nmiss += 1
            score *= 1-p
            miss_sum += p
    if dampening and (nmiss > 0):
        d = miss_sum/nmiss
        score *= (1/(4*(19-d)/19 + 1))*(1/nmiss**2)

    return score