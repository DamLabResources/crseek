
import svgwrite
import os
import sys
import pandas as pd
import argparse

boxWidth = 10
box_size = 15
v_spacing = 3
nameSpace =80

colors = {'G': '#F5F500', 'A': '#FF5454', 'T': '#00D118', 'C': '#26A8FF', 'N': '#B3B3B3'}


def parseSitesFile(infile):
    offtargets = []
    with open(infile, 'r') as f:
        f.readline()
        for line in f:
            line_items = line.split('\t')
            offtarget_sequence = line_items[21]
            offtarget_reads = line_items[11]
            ref_seq = line_items[32]
            if offtarget_sequence != '':
                offtargets.append({'seq': offtarget_sequence.strip(),
                                   'reads': int(offtarget_reads.strip())})
    offtargets = sorted(offtargets, key=lambda x: x['reads'], reverse=True)
    return offtargets, ref_seq

def parseSeq(infile):
    targets=[]
    dfRec = pd.read_csv(infile, sep=",", header=0)
    dfRec = dfRec.round(3)
    #ref_seq = dfRec.loc[dfRec.Sequence == 'HXB2', 'Target'].values[0]
    #print('ref_seq is %s' % ref_seq)
    for idx, row in dfRec.iterrows():
        offSeq = row.Target
        offScore = row.Value
        #ref_seq = 'TAGGGAACCCACTGCTTAAGCCTCAATAAAGCTTGCCTTG'
        if offSeq != '':
            targets.append({'seq': offSeq,
                            'reads': offScore,
                            'name': row.Sequence})
    return targets # , ref_seq

def visualizeOfftargets(infile, outfile, gRNA, title=None):
    output_folder = os.path.dirname(outfile)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get offtargets array from file
    #offtargets, ref_seq = parseSitesFile(infile)
    #offtargets, ref_seq = parseSeq(infile)
    offtargets = parseSeq(infile)
    ref_seq = str(gRNA) + 'NGG'
    # Initiate canvas
    dwg = svgwrite.Drawing(outfile + '.svg', profile='full', size = ("600px", "400px"))

    if title is not None:
        # Define top and left margins
        x_offset = 20
        y_offset = 50
        dwg.add(dwg.text(title, insert=(x_offset, 30), style="font-size:20px; font-family:Courier"))
    else:
        # Define top and left margins
        x_offset = 20
        y_offset = 20

        # Draw ticks
    tick_locations = [1, len(ref_seq)]
    tick_locations += range(len(ref_seq) + 1)[::10][1:]
    for x in tick_locations:
        dwg.add(dwg.text(str(x), insert=(x_offset + (x - 1) * box_size + 2 + nameSpace, y_offset - 2),
                         style="font-size:10px; font-family:Courier"))

    # Draw reference sequence row
    for i, c in enumerate(ref_seq):
        y = y_offset
        x = x_offset + i * box_size + nameSpace
        dwg.add(dwg.rect((x , y), (box_size, box_size), fill=colors[c]))
        dwg.add(
            dwg.text(c, insert=(x + 3, y + box_size - 3), fill='black', style="font-size:15px; font-family:Courier"))

    dwg.add(dwg.text('Activity Score', insert=(x_offset + box_size * len(ref_seq) + 16 + nameSpace, y_offset + box_size - 3),
                     style="font-size:15px; font-family:Courier"))

    # Draw aligned sequence rows
    y_offset += 10  # leave some extra space after the reference row
    for j, seq in enumerate(offtargets):
        y = y_offset + j * box_size
        x = x_offset
        dwg.add(dwg.text(str(seq['name']), insert=(x, 2* box_size + y -4), fill="black",
                         style="font-size:15px; font-family:Courier"))
        for i, c in enumerate(seq['seq']):
            x = x_offset + i * box_size +nameSpace
            if c == ref_seq[i] or ref_seq[i] == 'N':
                dwg.add(dwg.text(u"\u2022", insert=(x + 4.5, 2 * box_size + y - 4), fill='black',
                                 style="font-size:10px; font-family:Courier"))
            else:
                dwg.add(dwg.rect((x, box_size + y), (box_size, box_size), fill=colors[c]))
                dwg.add(dwg.text(c, insert=(x + 3, 2 * box_size + y - 3), fill='black',
                                 style="font-size:15px; font-family:Courier"))

        reads_text = dwg.text(str(seq['reads']),
                              insert=(box_size * (len(ref_seq) + 1) + 20 + nameSpace, y_offset + box_size * (j + 2) - 2),
                              fill='black', style="font-size:15px; font-family:Courier")
        dwg.add(reads_text)

    dwg.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize mismatches on aligned DNA sequences.')
    parser.add_argument('infile',
                        help='CSV file output from crisprbind.py')
    parser.add_argument('--out', dest='out', required=False)
    parser.add_argument('--gRNA', dest='gRNA', required=True)
    parser.add_argument('--title', dest='title', required=False)
    args = parser.parse_args()
    visualizeOfftargets(args.infile, args.out, args.gRNA, args.title)
    #main()
