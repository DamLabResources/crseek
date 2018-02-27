from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq, reverse_complement
from Bio.Alphabet import generic_dna, generic_rna, _get_base_alphabet, RNAAlphabet, DNAAlphabet
from Bio import SeqIO
import pandas as pd
import numpy as np
import shlex
from io import StringIO, BytesIO
import subprocess
from subprocess import STDOUT, CalledProcessError
from tempfile import TemporaryDirectory, NamedTemporaryFile
from Bio.SeqUtils import nt_search
from Bio.Seq import reverse_complement

from crisprtree import exceptions

import csv

import os


def extract_possible_targets(seq_record, pams = ('NGG',), both_strands = True):
    """
    Parameters
    ----------
    seq_record : SeqRecord
        Sequence to check
    pams : iter
        Set of PAMs to search. Allows ambigious nucleotides.
    both_strands : bool
        Check both strands?

    Returns
    -------

    list
        Targets excluding PAM in 5-3 orientation.

    """

    st_seq = str(seq_record.seq)

    found = set()
    for pam in pams:
        for res in nt_search(st_seq, pam)[1:]:
            found.add(Seq(st_seq[res-20:res], alphabet = generic_dna).transcribe())

    if both_strands:
        rseq = reverse_complement(st_seq)
        for pam in pams:
            for res in nt_search(rseq, pam)[1:]:
                found.add(Seq(rseq[res-20:res], alphabet = generic_dna).transcribe())

    return sorted(f for f in found if len(f) == 20)


def tile_seqrecord(spacer, seq_record):
    """ Simple utility function to convert a sequence and gRNA into something
    the preprocessing tools can deal with.

    Parameters
    ----------
    spacer : Seq
    seq_record : SeqRecord
        Sequence to tile

    Returns
    -------

    pd.DataFrame

    """

    exceptions._check_seq_alphabet(spacer, RNAAlphabet)
    exceptions._check_seq_alphabet(seq_record.seq, DNAAlphabet)

    tiles = []
    str_seq = str(seq_record.seq)
    for n in range(len(str_seq)-23):
        tiles.append({'name': seq_record.id,
                      'left': n,
                      'strand': 1,
                      'spacer': spacer,
                      'target': Seq(str_seq[n:n+23], alphabet = generic_dna)})
        tiles.append({'name': seq_record.id,
                      'left': n,
                      'strand': -1,
                      'spacer': spacer,
                      'target': Seq(reverse_complement(str_seq[n:n+23]),
                                    alphabet = generic_dna)})

    df = pd.DataFrame(tiles)

    return df.groupby(['name', 'strand', 'left'])[['spacer', 'target']].first()


def cas_offinder(spacers, mismatches, locus = None, direc = None,
                 pam = 'NRG', openci_devices = 'G0',
                 keeptmp = False):
    """ Call the cas-offinder tool and return the relevant info
    Parameters
    ----------
    spacers : list
        spacers to search for
    mismatches : int
        Number of mismatches to allow
    locus : list
        SeqRecords to search.
    direc : str
        Path to a directory containing fasta-files to search
    pam : str
        PAM to use when searching
    openci_devices : str
        Formatted string of device-IDs acceptable to cas-offinder
    keeptmp : bool
        Keep the temporary director? Useful for debugging

    Returns
    -------

    pd.DataFrame

    """

    msg = 'Must provide either sequences or a directory path'
    assert (locus is not None) or (direc is not None), msg

    _ = [exceptions._check_seq_alphabet(s, base_alphabet = RNAAlphabet) for s in spacers]

    with TemporaryDirectory() as tmpdir:

        if direc is None:
            fasta_path = os.path.join(tmpdir, 'search_seqs.fasta')
            with open(fasta_path, 'w') as handle:
                try:
                    SeqIO.write(locus, handle, 'fasta')
                except AttributeError:
                    raise ValueError('locus must be Bio.Seq objects')
            direc = tmpdir
        elif (direc is not None) and (locus is not None):
            tmpfile = NamedTemporaryFile(dir = direc,
                                         suffix = '.fasta',
                                         buffering = 1,
                                         mode = 'w')
            SeqIO.write(locus, tmpfile, 'fasta')

        input_path = os.path.join(tmpdir, 'input.txt')
        out_path = os.path.join(tmpdir, 'outdata.tsv')

        with open(input_path, 'w') as handle:
            handle.write(direc + '\n')
            handle.write('NNNNNNNNNNNNNNNNNNNN'+pam + '\n')
            for grna in spacers:

                if type(grna) != Seq:
                    raise ValueError('spacers must be Bio.Seq objects')

                # Not sure why, must be NNN no matter what the PAM length is
                handle.write('%sNNN %i\n' % (grna.back_transcribe(),
                                             mismatches))

        tdict = {'ifile': input_path,
                 'ofile': out_path,
                 'dev': openci_devices}
        cmd = 'cas-offinder %(ifile)s %(dev)s %(ofile)s'

        FNULL = open(os.devnull, 'w')
        call_args = shlex.split(cmd % tdict)
        try:
            subprocess.check_call(call_args, stdout=FNULL, stderr=STDOUT)
        except FileNotFoundError:
            raise AssertionError('cas-offinder not installed on the path')

        out_res = []
        with open(out_path) as handle:
            reader = csv.reader(handle, delimiter='\t')
            for row in reader:
                out_res.append({'spacer': Seq(row[0][:-3], alphabet=generic_dna).transcribe(),
                                'name': row[1],
                                'left': int(row[2]),
                                'target': Seq(row[3].upper(), alphabet = generic_dna),
                                'strand': 1 if row[4] == '+' else -1})
        if len(out_res) > 0:
            return pd.DataFrame(out_res).groupby(['name', 'strand', 'left'])[['spacer', 'target']].first()
        else:
            index = pd.MultiIndex.from_tuples([], names = ['name', 'strand', 'left'])
            return pd.DataFrame([], index = index, columns = ['spacer', 'target'])


def overlap_regions(hits, bed_path):
    """ Utility function to overlap hits with genomic regions
    Parameters
    ----------
    hits : pd.DataFrame
        Dataframe as returned by tile_seqrecord or cas_offinder
    bed_path : str
        Path to bedfile of gene regions

    Returns
    -------

    pd.Series
        A boolean series with the same index as hits.
        True indicates that the region is within a defined region.

    """

    if not os.path.exists(bed_path):
        raise IOError('%s does not exist.' % bed_path)

    with NamedTemporaryFile(mode = 'w',buffering = 1,
                            newline = '') as handle:
        writer = csv.writer(handle, delimiter = '\t',
                            quoting=csv.QUOTE_NONE,
                            dialect = csv.unix_dialect)
        for name, strand, left in hits.index:
            if strand not in {1, -1}:
                raise TypeError('Strand must be {1, -1}, found %s' % strand)
            if type(left) is not int:
                raise TypeError('Left must be an integer, found %s' % type(left))

            st = '+' if strand > 0 else '-'
            writer.writerow([name, left, left+23, None, None, st])

        tdict = {'genes': bed_path, 'hit': handle.name}
        cmd = 'bedtools intersect -a %(hit)s -b %(genes)s -loj -u'
        call_args = shlex.split(cmd % tdict)
        out = subprocess.check_output(call_args)
        cols = ['Name', 'Left', 'Right', '_', '_', 'Strand']
        df = pd.read_csv(BytesIO(out), sep='\t',
                         header = None,
                         names = cols)
    df['Strand'] = df['Strand'].map(lambda x: 1 if x == '+' else -1)
    df['Region'] = True
    ser = df.groupby(['Name', 'Strand', 'Left'])['Region'].any()
    _, found = hits.align(ser, axis=0, join='left')

    return found.fillna(False)