import argparse
from crseek.utils import extract_possible_targets
from collections import Counter
from Bio import SeqIO

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Find gRNA targets')
    parser.add_argument('files', nargs = '+',
                        help = 'Fasta files to search')
    parser.add_argument('--topN', dest='topN', default=20,
                        help = 'How many gRNAs to return')
    parser.add_argument('--fasta', dest='fasta',
                        help = 'Output results in FASTA format')
    args = parser.parse_args()

    target_counter = Counter()

    for f in args.files:
        with open(f) as handle:
            for seqR in SeqIO.parse(handle, 'fasta'):
                seqR.seq = seqR.seq.ungap('-')
                found = Counter(target[:-3] for target in extract_possible_targets(seqR))
                target_counter += found

    if args.fasta:
        with open(args.fasta, 'w') as handle:
            for gRNA, count in target_counter.most_common(args.topN):
                handle.write('>%i-hits\n%s\n' % (count, gRNA))

    for gRNA, count in target_counter.most_common(args.topN):
        print(gRNA, '\t', count)