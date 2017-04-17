from Bio.SeqRecord import SeqRecord
from Bio.Seq import reverse_complement
import pandas as pd
import numpy as np


def tile_seqrecord(grna, seq_record):
    """ Simple utility function to convert a sequence and gRNA into something
    the preprocessing tools can seal with.

    Parameters
    ----------
    grna : str
    seq_record : SeqRecord
        Sequence to tile

    Returns
    -------

    pd.DataFrame

    """

    tiles = []
    str_seq = str(seq_record.seq)
    for n in range(len(str_seq)-23):
        tiles.append({'Name': seq_record.id,
                      'Left': n,
                      'Strand': 1,
                      'gRNA': grna,
                      'Seq': str_seq[n:n+23]})
        tiles.append({'Name': seq_record.id,
                      'Left': n,
                      'Strand': -1,
                      'gRNA': grna,
                      'Seq': reverse_complement(str_seq[n:n+23])})

    df = pd.DataFrame(tiles)

    return df.groupby(['Name', 'Strand', 'Left'])[['gRNA', 'Seq']].first()