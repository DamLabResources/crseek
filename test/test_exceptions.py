import pytest
from Bio import Alphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from crseek import exceptions


class TestCheckAlphabet(object):
    def test_correct_dna(self):
        seq = Seq('A' * 10, alphabet=Alphabet.generic_dna)
        exceptions._check_seq_alphabet(seq)

    def test_correct_rna(self):
        seq = Seq('A' * 10, alphabet=Alphabet.generic_rna)
        exceptions._check_seq_alphabet(seq, base_alphabet=Alphabet.RNAAlphabet)

    def test_incorrect_dna(self):
        seq = Seq('A' * 10, alphabet=Alphabet.generic_dna)
        with pytest.raises(exceptions.WrongAlphabetException):
            exceptions._check_seq_alphabet(seq, base_alphabet=Alphabet.RNAAlphabet)

    def test_incorrect_rna(self):
        seq = Seq('A' * 10, alphabet=Alphabet.generic_rna)
        with pytest.raises(exceptions.WrongAlphabetException):
            exceptions._check_seq_alphabet(seq, base_alphabet=Alphabet.DNAAlphabet)

    def test_str(self):
        seq = 'A' * 10
        with pytest.raises(ValueError):
            exceptions._check_seq_alphabet(seq, base_alphabet=Alphabet.DNAAlphabet)

    def test_seqrecord(self):
        seq = SeqRecord(Seq('A' * 10, alphabet=Alphabet.generic_dna), id='test')
        with pytest.raises(ValueError):
            exceptions._check_seq_alphabet(seq, base_alphabet=Alphabet.DNAAlphabet)
