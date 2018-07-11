from Bio import Alphabet
from Bio.Seq import Seq


def _check_seq_alphabet(seq, base_alphabet=Alphabet.DNAAlphabet):
    """ Checks objects to make sure they are Bio.Seq objects and they have
    the correct alphabet
    Parameters
    ----------
    seq : Seq
    base_alphabet : Alphabet.Alphabet

    """

    try:
        base = Alphabet._get_base_alphabet(seq.alphabet)
        if not isinstance(base, base_alphabet):
            raise WrongAlphabetException()
    except AttributeError:
        raise ValueError('Alphabets must be defined')


class WrongAlphabetException(BaseException):
    pass
