import os
from itertools import cycle, product
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import yaml
from Bio import Alphabet
from Bio.Seq import Seq

from crisprtree import loaders


def test_load_mismatch_penalties_CFD():

    matrix, pams = loaders.load_mismatch_scores('CFD')

    np.testing.assert_array_equal(matrix.columns,
                                  [''.join(tup) for tup in product('ACGU', 'ACGT')])

    assert matrix.loc[4, 'AC'] == 0.5
    assert matrix.loc[4, 'UG'] == 0.64
