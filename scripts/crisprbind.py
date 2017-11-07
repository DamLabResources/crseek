import sys
import os
pacPath = os.path.abspath(os.path.join(os.path.dirname(__file__), (os.pardir)))
sys.path.append(pacPath)

import argparse
import pandas as pd
from crisprtree.evaluators import check_grna_across_seqs
from crisprtree import estimators
from Bio import SeqIO


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Check hits of gRNA targets')
    parser.add_argument('files', nargs = '+',
                        help = 'Fasta files to search')
    parser.add_argument('--gRNA', dest='gRNA')
    parser.add_argument('--out', dest='out')
    parser.add_argument('--topN', dest='topN', default=20,
                        help = 'How many gRNAs to return')
    parser.add_argument('--method', dest='method',
                        choices = ['MIT', 'CFD', 'missmatch'],
                        default = 'MIT')
    args = parser.parse_args()

    seqs = []
    names = []
    for f in args.files:
        with open(f) as handle:
            for seqR in SeqIO.parse(handle, 'fasta'):
                seqs.append(str(seqR.seq.ungap('-')))
                names.append(seqR.id)

    if args.method == 'MIT':
        est = estimators.MITEstimator.build_pipeline()
    elif args.method == 'CFD':
        est = estimators.CFDEstimator.build_pipeline()
    elif args.method == 'missmatch':
        est = estimators.MismatchEstimator.build_pipeline()
    else:
        raise ValueError('Unknown method provided.')

    res = check_grna_across_seqs(args.gRNA, seqs, est)
    res.index = pd.Index(names, name = 'Sequence')
    res.to_csv(args.out)
