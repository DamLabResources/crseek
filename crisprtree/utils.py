from Bio.SeqRecord import SeqRecord
from Bio.Seq import reverse_complement, Seq
from Bio import SeqIO
import pandas as pd
import numpy as np
import shlex
from io import StringIO, BytesIO
from subprocess import check_call, STDOUT, check_output,CalledProcessError
from tempfile import TemporaryDirectory, NamedTemporaryFile
from Bio.SeqUtils import nt_search
from Bio.Alphabet import generic_alphabet

import csv

import os


def smrt_seq_convert(outfmt, seqs, default_phred=None,
                     alphabet=generic_alphabet):
    """Converts iterable of any sequence type into another format.

    This function transparently converts sequence objects (even mixed lists)
    into any other sequence representation. Input and output data can be
    Bio.SeqRecord, Bio.Seq, str, or (name, str) tuples. If the `default_phred`
    kwarg is provided then it will transparently add default quality scores.

    If names are needed for the output format but are not included ('str'
    and 'Seq' inputs) then the ids are generated using the position in the
    iterable.

    Parameters
    ----------
    outfmt : { 'str', 'SeqRecord', 'Seq', 'tuple' }
        The desired output format.
    seqs : iterable
        An iterable of sequences. They can be any of the above formats.
    default_phred : int, optional
        The default phred-score to give to every call when outputing a
        SeqRecord format.
    alphabet : Bio.Alphabet.generic_nucleotide, optional.
        The alphabet to use when generating the Seq and SeqRecord objects.

    Returns
    -------
    generator
        A generator object yielding the records in the desired format.
    """

    possible_formats = {'str': lambda x: str(x.seq),
                        'SeqRecord': lambda x: x,
                        'Seq': lambda x: x.seq,
                        'tuple': lambda x: (x.id, str(x.seq))}
    assert outfmt in possible_formats
    for num, seq_obj in enumerate(seqs):
        if isinstance(seq_obj, SeqRecord):
            seq_rec = seq_obj
        elif isinstance(seq_obj, Seq):
            seq_rec = SeqRecord(seq_obj, id='Seq-%i' % num)
        elif isinstance(seq_obj, str):
            seq_rec = SeqRecord(Seq(seq_obj, alphabet=alphabet),
                                          id='Seq-%i' % num)
        elif isinstance(seq_obj, tuple):
            seq_rec = SeqRecord(Seq(seq_obj[1], alphabet=alphabet),
                                          id=seq_obj[0])
        else:
            raise AssertionError("Don't understand obj of type: %s" % type(seq_obj))

        if default_phred and ('phred_quality' not in seq_rec.letter_annotations):
            seq_rec.letter_annotations['phred_quality'] = [default_phred]*len(seq_rec)

        yield possible_formats[outfmt](seq_rec)



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
            found.add(st_seq[res-20:res])

    if both_strands:
        rseq = reverse_complement(st_seq)
        for pam in pams:
            for res in nt_search(rseq, pam)[1:]:
                found.add(rseq[res-20:res])

    return sorted(f for f in found if len(f) == 20)


def tile_seqrecord(grna, seq_record):
    """ Simple utility function to convert a sequence and gRNA into something
    the preprocessing tools can deal with.

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



def _run_casoffinder(input_path, out_path, openci_devices):

    raise NotImplementedError

    tdict = {'ifile': input_path,
             'ofile': out_path,
             'dev': openci_devices}
    cmd = 'cas-offinder %(ifile)s %(dev)s %(ofile)s'

    FNULL = open(os.devnull, 'w')
    call_args = shlex.split(cmd % tdict)
    try:
        check_call(call_args, stdout=FNULL, stderr=STDOUT)
    except FileNotFoundError:
        raise AssertionError('cas-offinder not installed on the path')


def _build_cas_offinder_input_file(handle, gRNAs, fasta_direc,
                                   mismatches,
                                   template = 'NNNNNNNNNNNNNNNNNNNNNRG'):
    """ Utility to build the input file template for cas-offinder
    Parameters
    ----------
    handle : file
        A file like object opened for writing
    gRNAs : list
        Any list of Seq-like objects
    fasta_direc : str
        Path to the directory of fasta-files for searching
    missmatches : int
        Number of mismatches allowed
    template : str
        The template for cas-offinder searching.

    Returns
    -------
    None
    """

    handle.write(fasta_direc + '\n')
    handle.write(template + '\n')
    for grna in gRNAs:
        handle.write('%sNNN %i\n' % (grna, mismatches))


def cas_offinder(gRNAs, mismatches, seqs = None, direc = None,
                      openci_devices = 'G0', keeptmp = False,
                 template = 'NNNNNNNNNNNNNNNNNNNNNRG'):
    """ Call the cas-offinder tool and return the relevant info
    Parameters
    ----------
    gRNAs : list
        gRNAs to search for
    mismatches : int
        Number of mismatches to allow
    seqs : list
        SeqRecords to search.
    direc : str
        Path to a directory containing fasta-files to search
    openci_devices : str
        Formatted string of device-IDs acceptable to cas-offinder
    keeptmp : bool
        Keep the temporary director? Useful for debugging
    template : str
        The template for cas-offinder searching.

    Returns
    -------

    pd.DataFrame

    """

    msg = 'Must provide either sequences or a directory path'
    assert (seqs is not None) or (direc is not None), msg

    with TemporaryDirectory() as tmpdir:

        if direc is None:
            fasta_path = os.path.join(tmpdir, 'search_seqs.fasta')
            with open(fasta_path, 'w') as handle:
                SeqIO.write(seqs, handle, 'fasta')
            direc = tmpdir
        elif (direc is not None) and (seqs is not None):
            tmpfile = NamedTemporaryFile(dir = direc,
                                         suffix = '.fasta',
                                         buffering = 1,
                                         mode = 'w')
            SeqIO.write(seqs, tmpfile, 'fasta')

        input_path = os.path.join(tmpdir, 'input.txt')
        out_path = os.path.join(tmpdir, 'outdata.tsv')

        with open(input_path, 'w') as handle:

            _build_cas_offinder_input_file(handle, gRNAs, direc, mismatches,
                                           template = template)

        _run_casoffinder(input_path, out_path, openci_devices)

        out_res = []
        with open(out_path) as handle:
            reader = csv.reader(handle, delimiter='\t')
            for row in reader:
                out_res.append({'gRNA': row[0][:-3],
                                'Name': row[1],
                                'Left': int(row[2]),
                                'Seq': row[3].upper(),
                                'Strand': 1 if row[4] == '+' else -1})

    return pd.DataFrame(out_res).groupby(['Name', 'Strand', 'Left'])[['gRNA', 'Seq']].first()


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
        out = check_output(call_args)
        cols = ['Name', 'Left', 'Right', '_', '_', 'Strand']
        df = pd.read_csv(BytesIO(out), sep='\t',
                         header = None,
                         names = cols)
    df['Strand'] = df['Strand'].map(lambda x: 1 if x == '+' else -1)
    df['Region'] = True
    ser = df.groupby(['Name', 'Strand', 'Left'])['Region'].any()
    _, found = hits.align(ser, axis=0, join='left')

    return found.fillna(False)