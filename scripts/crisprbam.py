import sys
import os
pacPath = os.path.abspath(os.path.join(os.path.dirname(__file__), (os.pardir)))
sys.path.append(pacPath)

import argparse
import pandas as pd
import numpy as np
from crisprtree.evaluators import positional_aggregation
from crisprtree import estimators
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from pysam import AlignedSegment, Samfile

from itertools import islice



def load_grna_file(path):
    """ Imports gRNA, chomr, and start positions
    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame
    """

    grnas = pd.read_csv(path)
    for col in ('gRNA', 'Chrom', 'Start', 'Name'):
        assert col in grnas.columns

    assert len(grnas['Chrom'].unique()) == 1, 'Can only check gRNAs targeting one chromosome'

    grnas['gRNA'] = grnas['gRNA'].map(lambda x: Seq(x, alphabet=generic_dna))

    return grnas


def convert_reads_to_rows(read):
    """ Extracts info from pysam into a pd.Series
    Parameters
    ----------
    read : AlignedSegment

    Returns
    -------
    pd.Series

    """

    rec = Seq(read.query_sequence, alphabet=generic_dna)

    return pd.Series({'Seq': rec,
                      'Start': read.reference_start,
                      'Stop': read.reference_end},
                     name = read.query_name)


def batch_iterate_reads(path, chrom, batchsize):
    """ Generates batches of pre-screened reads
    Parameters
    ----------
    path : str
        Path to bam-file for extraction
    chrom : str
        Chromosome in bam-file to process
    batchsize : int
        Number of sequences to process at once

    Returns
    -------
    generator

    """

    bamfile = Samfile(path, mode='rb')
    it = bamfile.fetch(chrom)
    rows = []
    while True:
        for read in islice(it, batchsize):
            rows.append(convert_reads_to_rows(read))
        yield pd.DataFrame(rows)
        if len(rows) < batchsize:
            break
        rows = []


def aggregate_batch(batch_res):
    """ Calculate stats on each batch
    Parameters
    ----------
    batch_res : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """


    agg_dict = {'min':'min',
                'mean':'mean',
                'max': 'max',
                'count': 'count'}

    for cut in [0, 0.25, 0.5, 0.75, 0.9, 0.99]:
        agg_dict['FracAbove_%02i' % (100*cut)] = lambda x: (x > cut).mean()


    return batch_res.groupby(['Name', 'gRNA'], as_index=False)['Value'].agg(agg_dict)


def combine_aggregated_results(agg_df):
    """ Use weighting to get final results
    Parameters
    ----------
    agg_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """

    agg_dict = {'min':'min',
                'max': 'max',
                'count': 'sum'}

    big_agg = agg_df.groupby(['Name', 'gRNA']).agg(agg_dict)

    weighted_mean_cols = ['mean'] + [col for col in agg_df.columns if col.startswith('FracAbove_')]

    weights = pd.pivot_table(agg_df,
                             index = 'BatchNum',
                             columns = ['Name', 'gRNA'],
                             values = 'count',
                             aggfunc = 'first').fillna(0)

    for col in weighted_mean_cols:
        values = pd.pivot_table(agg_df,
                            index = 'BatchNum',
                            columns = ['Name', 'gRNA'],
                            values = col,
                            aggfunc = 'first')
        vl = np.ma.array(values.values,
                         mask = values.isnull().values)
        normed = np.ma.average(vl, weights=weights.values,
                               axis = 0)
        big_agg[col] = pd.Series(normed, index=values.columns)

    return big_agg


def main(bam_path, chrom, grnas, est, batchsize=5000):
    """
    Parameters
    ----------
    bam_path : str
    chrom : str
    grnas : pd.DataFrame
    est : estimators.BaseEstimator
    batchsize : int or None

    Returns
    -------
    pd.DataFrame

    """

    read_batches = batch_iterate_reads(bam_path, chrom, batchsize)

    aggregated_results = []
    for num, batch in enumerate(read_batches):
        res = positional_aggregation(batch, grnas, est)
        if res is not None:
            aggregated_results.append(aggregate_batch(res))
            aggregated_results[-1]['BatchNum'] = num
        if num > 10:
            break

    agg_res = pd.concat(aggregated_results, axis=0, ignore_index=True)
    return combine_aggregated_results(agg_res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Check hits of gRNA targets')
    parser.add_argument('file',
                        help = 'BAM files to search')
    parser.add_argument('--gRNA', dest='gRNA',
                        help = 'Path to CSV file that contains the gRNA, Chrom, and Start columns')
    parser.add_argument('--out', dest='out')
    parser.add_argument('--method', dest='method',
                        choices = ['MIT', 'CFD', 'missmatch'],
                        default = 'MIT')
    parser.add_argument('--batch-size', dest='batch',
                        type=int,
                        default = 5000,
                        help = 'How many reads to process in one batch.')
    args = parser.parse_args()

    if args.method == 'MIT':
        est = estimators.MITEstimator.build_pipeline()
    elif args.method == 'CFD':
        est = estimators.CFDEstimator.build_pipeline()
    elif args.method == 'missmatch':
        est = estimators.MismatchEstimator.build_pipeline()
    else:
        raise ValueError('Unknown method provided.')


    grnas = load_grna_file(args.gRNA)
    chrom = grnas['Chrom'].iloc[0]

    comb_agg = main(args.file, chrom, grnas, est, batchsize=5000)

    order = ['Name', 'gRNA', 'max', 'min', 'count', 'mean',
             'FracAbove_00', 'FracAbove_25', 'FracAbove_50', 'FracAbove_75'
             'FracAbove_90', 'FracAbove_99']

    comb_agg.reset_index()[order].to_csv(args.out)



