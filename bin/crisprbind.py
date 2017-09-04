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
    args = parser.parse_args()

    seqs = []
    names = []
    for f in args.files:
        with open(f) as handle:
            for seqR in SeqIO.parse(handle, 'fasta'):
                seqs.append(str(seqR.seq.ungap('-')))
                names.append(seqR.id)

    est = estimators.MITEstimator.build_pipeline(cutoff=0.5)

    res = check_grna_across_seqs(args.gRNA, seqs, est)
    res.index = pd.Index(names, name = 'Sequence')
    res.to_csv(args.out)
