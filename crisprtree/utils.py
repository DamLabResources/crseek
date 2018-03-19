import csv
import os
import shlex
import subprocess
import warnings
from io import BytesIO
from subprocess import STDOUT, CalledProcessError, check_output
from tempfile import TemporaryDirectory, NamedTemporaryFile

import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_alphabet
from Bio.Alphabet import generic_dna, RNAAlphabet, DNAAlphabet
from Bio.Seq import Seq, reverse_complement, BiopythonWarning
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import nt_search

from crisprtree import exceptions


def smrt_seq_convert(outfmt, seqs, default_phred = None,
                     alphabet = generic_alphabet):
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
            seq_rec = SeqRecord(seq_obj, id = 'Seq-%i' % num)
        elif isinstance(seq_obj, str):
            seq_rec = SeqRecord(Seq(seq_obj, alphabet = alphabet),
                                id = 'Seq-%i' % num)
        elif isinstance(seq_obj, tuple):
            seq_rec = SeqRecord(Seq(seq_obj[1], alphabet = alphabet),
                                id = seq_obj[0])
        else:
            raise AssertionError("Don't understand obj of type: %s" % type(seq_obj))

        if default_phred and ('phred_quality' not in seq_rec.letter_annotations):
            seq_rec.letter_annotations['phred_quality'] = [default_phred] * len(seq_rec)

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

    st_seq = str(seq_record.seq.upper())

    with warnings.catch_warnings():
        # We're using the new BioPython Seq comparison, don't need the
        # warning EVERY time.
        warnings.simplefilter('ignore', category = BiopythonWarning)

        found = set()
        for pam in pams:
            for res in nt_search(st_seq, pam)[1:]:
                found.add(Seq(st_seq[res - 20:res], alphabet = generic_dna).transcribe())

        if both_strands:
            rseq = reverse_complement(st_seq)
            for pam in pams:
                for res in nt_search(rseq, pam)[1:]:
                    found.add(Seq(rseq[res - 20:res], alphabet = generic_dna).transcribe())

    return sorted(f for f in found if len(f) == 20)


def _make_record_key(seqR):
    """
    Parameters
    ----------
    seqR : SeqRecord

    Returns
    -------
    str
    """

    # clean function in Bio.SeqIO.Interfaces
    id = seqR.id.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    description = seqR.description.replace("\n", " ").replace("\r", " ").replace("  ", " ")

    # Class FastaWriter in Bio.SeqIO.FastaIO
    if description and description.split(None, 1)[0] == id:
        # The description includes the id at the start
        title = description
    elif description:
        title = "%s %s" % (id, description)
    else:
        title = id
    return title


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
    str_seq = str(seq_record.seq.upper())
    for n in range(len(str_seq) - 23):
        tiles.append({'name': _make_record_key(seq_record),
                      'left': n,
                      'strand': 1,
                      'spacer': spacer,
                      'target': Seq(str_seq[n:n + 23], alphabet = generic_dna)})
        tiles.append({'name': _make_record_key(seq_record),
                      'left': n,
                      'strand': -1,
                      'spacer': spacer,
                      'target': Seq(reverse_complement(str_seq[n:n + 23]),
                                    alphabet = generic_dna)})

    df = pd.DataFrame(tiles)

    return df.groupby(['name', 'strand', 'left'])[['spacer', 'target']].first()


def _run_casoffinder(input_path, out_path, openci_devices):
    """

    Parameters
    ----------
    input_path : file
        An input file that follows the input format for cas-offinder
    out_path : file
        An output file from cas-offinder
    openci_devices : str or None
        A letter for openci usage. ['C'|'G'] for CPU or GPU, a numeric order following the device letter may apply
        e.g. 'C0' or 'G1'

    Returns
    -------
    None
    """

    if openci_devices is None:
        openci_devices = _guess_openci_devices()

    tdict = {'ifile': input_path,
             'ofile': out_path,
             'dev': openci_devices}
    cmd = 'cas-offinder %(ifile)s %(dev)s %(ofile)s'

    FNULL = open(os.devnull, 'w')
    call_args = shlex.split(cmd % tdict)
    try:
        subprocess.check_call(call_args, stdout = FNULL, stderr = STDOUT)
    except FileNotFoundError:
        raise AssertionError('cas-offinder not installed on the path')


def _build_cas_offinder_input_file(handle, spacers, fasta_direc,
                                   mismatches,
                                   template = 'NNNNNNNNNNNNNNNNNNNNNRG'):
    """ Utility to build the input file template for cas-offinder
    Parameters
    ----------
    handle : file
        A file like object opened for writing
    spacers : list
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
    for spacer in spacers:
        handle.write('%sNNN %i\n' % (spacer.back_transcribe(), mismatches))


def _guess_openci_devices():
    """ Guess which OpenCI device to use.
    Assumes they're ranked and the first is the "best".
    May need a better strategy.
    Returns
    -------
    str
    """

    out = subprocess.check_output(['cas-offinder'], stderr = subprocess.STDOUT)
    out = out.decode('ascii')
    pos = out.find('Available device list:')
    assert pos != -1
    for line in out[pos:].split('\n'):
        if line.startswith('Type:'):
            _pu, _id, _name = line.split(', ')
            pu = 'G' if 'GPU' in _pu else 'C'
            id = _id.split(' ')[1]
            return pu + id


def cas_offinder(spacers, mismatches, locus = None, direc = None,
                 openci_devices = None, keeptmp = False,
                 template = 'NNNNNNNNNNNNNNNNNNNN',
                 pam = 'NRG'):
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
    openci_devices : str or None
        Formatted string of device-IDs acceptable to cas-offinder. If None
        the first choice is picked from the OpenCI device list.
    keeptmp : bool
        Keep the temporary director? Useful for debugging
    template : str
        The template for cas-offinder searching.

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
            _build_cas_offinder_input_file(handle, spacers, direc, mismatches,
                                           template = template + pam)
        _run_casoffinder(input_path, out_path, openci_devices)

        out_res = []
        with open(out_path) as handle:
            reader = csv.reader(handle, delimiter = '\t')
            for row in reader:
                out_res.append({'spacer': Seq(row[0][:-3], alphabet = generic_dna).transcribe(),
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

    with NamedTemporaryFile(mode = 'w', buffering = 1,
                            newline = '') as handle:
        writer = csv.writer(handle, delimiter = '\t',
                            quoting = csv.QUOTE_NONE,
                            dialect = csv.unix_dialect)
        for name, strand, left in hits.index:
            if strand not in {1, -1}:
                raise TypeError('Strand must be {1, -1}, found %s' % strand)
            if type(left) is not int:
                raise TypeError('Left must be an integer, found %s' % type(left))

            st = '+' if strand > 0 else '-'
            writer.writerow([name, left, left + 23, None, None, st])

        tdict = {'genes': bed_path, 'hit': handle.name}
        cmd = 'bedtools intersect -a %(hit)s -b %(genes)s -loj -u'
        call_args = shlex.split(cmd % tdict)
        out = subprocess.check_output(call_args)
        cols = ['Name', 'Left', 'Right', '_', '_', 'Strand']
        df = pd.read_csv(BytesIO(out), sep = '\t',
                         header = None,
                         names = cols)
    df['Strand'] = df['Strand'].map(lambda x: 1 if x == '+' else -1)
    df['Region'] = True
    ser = df.groupby(['Name', 'Strand', 'Left'])['Region'].any()
    _, found = hits.align(ser, axis = 0, join = 'left')

    return found.fillna(False)


def _missing_casoffinder():
    """ Returns True if cas-offinder is not on the path"""

    try:
        out = check_output(['which', 'cas-offinder'])
        return len(out.strip()) == 0
    except CalledProcessError:
        return True
