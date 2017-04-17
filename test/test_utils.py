from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq, reverse_complement
from crisprtree import utils


class TestTiling(object):

    def test_basic(self):

        grna = 'A'*20
        bseq = 'ACTG'*20
        seqR = SeqRecord(Seq(bseq), id='checking')

        res = utils.tile_seqrecord(grna, seqR)

        assert len(res) > 1
        assert (res['gRNA'] == grna).all()

        for (name, strand, start), row in res.iterrows():
            assert name == 'checking'
            if strand == 1:
                assert row['Seq'] == bseq[start:start+23]
            else:
                assert row['Seq'] == reverse_complement(bseq[start:start+23])