# Introduction

## Cas9 history

## Cas9 binding rules

However, recent research has shown that there are still unknown aspects of the binding site recognition of the Cas9 system [PMID: ].
Early target search strategies involved using rule based methods that excluded potiential targets by defining either a total maximum of mismatches [PMID: ] or distinguishing between seed and tail regions [PMID: ].
Later targeting methods employ a data driven approach by using a dataset of experimentally determined targeting probabilities and extracting a more nuanced rule methodology.
Work by Hsu et al [PMID: ] created a position specific binding matrix by testing *X* hundred gRNAs engineered to have mismatches to their intended target.
Deunch et al [PMID: ] followed this work by creating an even larger dataset of >80,000 gRNAs targeting many positions across the same gene.
Klien et al [PMID: ] further refined this method by developing a kinetic model of Cas9 binding and cleavage using these datasets and showing that a small number of parameters can explain these experimentally determined binding rules.

## Cas9 uses
The CRISPR/Cas9 system has revolutionized the gene editing field [PMID: ].
It has democratized the field by lowering the barrier to entry for editing a specific genomic loci [PMID: ].
Various pre-made Cas9 systems have been developed and released which have useful properties such as an inducible expression [PMID: , Addgene: ], tissue specificity [PMID: , Addgene: ], as well as advanced editing strategies [PMID: , Addgene: ].
In practice, targeting a specific region genome with any of these systems simply involves finding a protospacer adjacent motif (PAM) next to unique 20-bp segment of the DNA [PMID: ] and performing basic molecular biology techniques.

## Current prediction software
There are numerous currently employed tools for designing gRNAs. These have been extensively reviewed by [PMIDs:].
All of these tools use the same basic strategy.
First, the target DNA is searched for PAM recognition sites and the adjacent protospacers are extracted.
Next, the potential protospacers are searched against a reference database allowing for multiple mismatches and are scored for potential off-target sites using one of the many scoring methods described above.
Finally, any the protospacers are ranked by their potential off-target risk.
These tools can be served either as a webserver and occasionally as a locally installable toolset.

## Lacking features
However, all of these tools lack many of the features needed from a bioinformatics perspective.
Webtools are often limited to searching model organisms.
Most tools cannot search for non-standard Cas9 variants such as Cpf1, SaCas9, or other species specific Cas9s.
Currently distributed software lacks unit-testing, one-command install of the tool and dependencies, continious integration frameworks, or a documented API.
These features are needed in order for bioinformaticians to design larger scale experiments or develop automation methods for more advanced gene editing strategies.

## Our stuff
In this manuscript we present CRISPRTree, a Python library mirroring the popular Scikit-Learn API.
It provides both high- and low-level methods for designing CRISPR gene editing strategies.
We believe it will provide an invaluable resource for computational biologists looking to employ CRISPR based gene editing techniques.

# Implementation

We present here a collection of tools for the biomedical software developer that needs to design advance CRISPR/Cas9 strategies.
This toolset was built with a developer perspective to aid biologists with programming experience who are looking to perform non-standard CRISPR/Cas9 gene editting.
The toolset, and its dependencies, are installable using the Ananconda environment system [REF] and is hosted in the BioConda repository [REF].

## Terminology

Throughout the research field there is a great deal of variability in the nomeclature of the various parts of the CRISPR/Cas9 complex.
Many of these nomeclatures do not conform to the Python PEP-8 standard [REF].
For consistency we refer to the complex parts in the following ways:
  - The `gRNA` refers to the entire strand of RNA which binds to the CRISPR/Cas9 complex. This molecule is not directly represented in crisprtree.
  - The `pam` refers to the, potentially degenerate, nucleotide recognition site of the particular Cas9. This requires a DNA alphabet.
  - The `spacer` refers to the 20-nucleotide region of the gRNA which is used for target matching by the CRISPR/Cas9 complex. This requires an RNA alphabet.
  - The `target` refers to the 20-nucleotide region of the DNA which, potentially, matches the `spacer`.
  For various implementation details the `target` includes the PAM recognition site. This requires a DNA alphabet.
  - The `loci` refers to a >20-nucleotide segment of the DNA which may contain potential `target`s.

## Class schematic

We have intentionally mirrored the Scikit-Learn API.
We subclass the `BaseEstimator` class and overload the `fit`, `transform`, and `predict` methods.
This allows us to seamlessly use all of the tools of the Scikit-Learn package such as cross-validation, normalization, or other prediciton strategies.
We have decomposed this methodology into three main tasks: Searching, Encoding, and Predicting.
We use the BioPython library to enforce appropriate nucleotide alphabets [PMID: ] as inputs and outputs of the various tools.

## Searching

We define `search`ing as the act of locating potential `target` sites in a `loci`.
DNA segments can be any files, or directories of files, readable by `Bio.SeqIO` (fasta, genbank, etc), lists of `Bio.SeqRecord` objects, or `np.array`s of charachers.
Our tool provides two stragies for this searching: exhaustive or mismatch based.
In an exhausitve search all potential positions are considered as potiential targets.
For mismatch searching we have created a wrapper around the popular `cas-offinder` library [PMID: ].
This library allows for rapid mismatch searching, even employing the GPU if present and properly installed.
The `utils.tile_seqrecord` and `utils.cas_offinder` return arrays of (spacer, target) pairs for downstream analysis.

## Preprocessing

Once targets have been found on the genomic loci we must encode the (spacer, target) pairs for downstream analysis.
As of this writing, there are two main strategies for this encoding: missmatch vs one-hot encoding.
Rule-based mismatching and the Hsu et al [PMID: ] strategies both give the same weight independenet of the mismatch identity; as such this encoding is a binary vector of length 21 (20bp for spacer, 1 for PAM matcing).
One-hot encoding is used for strategies such as Deunch et al [PMID: ] which give different binding penalities based on the mismatch identity; for example an A:T mismatch may have a larger penalty then a A:C mismatch.

These to strategies have been encapusulated into the `preprocessing.MatchingTransformer` and `preprocessing.OneHotEncoder` classes.
These classes take pairs of `spacer`, `target` and return binary vectors for downstream processing.
By implementing these as sublasses of the `sklearn.BaseEstimator` one can use these classes in `sklearn.Pipeline` instances.
This downstream processing can either employ one of the pre-built `estimator` classes or as preprocessing for other machine learning algorithms.

## Estimators

We have collected multiple pre-built algorithms for determining the activity given a pair of `spacer`, `targets`.
These estimators as input take the binary vectors produced by the various preprocessing madules.
We built a flexible `MismatchEstimator` that allows one to specificy the seed-length, number of seed or non-seed mismatches, and the PAM identity.
These parameters can also be loaded from user-supplied yaml files to allow for the exploration of non-standard Cas9 variants.
We have implemented the penalty scoring strategy described by Hsu et al [PMID: ] as the `MITEstimator` and the Deunch et al [PMID: ] method as the `CFDEstimator`.
The newly proposed kinect model developed by Klien et al [PMID: ] has also been implemented as the `KineticEstimator`.
New estimators can be easily added by subclassing the `SequenceBase` abstract base-class and overloading the relevant methods.

## Additional Features

In order to account for many of the idiosyncrasies of designing CRSIPR/Cas9 tools we have added a few optional features.
Degeneterate bases can be used across all tools.
These bases are assumed to be each relevant nucleotide at equal likelihoods and the penalty scores are scaled accordingly.
SAM/BAM files can be used in all input instances.
This allows for search and estimation across variable populations. This is useful when targeting microbiomes, metagenomes, or highly mutable viral genomes.

# Uses

In order to showcase the many uses of the `crisprtree` module we present four simple and realistics CRISPR/Cas9 design tasks.
These tasks would not be reasonably possible with currently available tools.

## Task

In this project you are tasked with introducing an eGFP plasmid (Addgene-54622) into the newly sequenced Clostridioides difficile genome (Genbank: NC_009089.1).
You want to make an spCas9 positive control knocking out the eGFP gene while missing the rest of the newly sequenced genome.
The following code-snippet walks through a basic execution strategy.

```python
# Build the CFD estimator
estimator = estimators.CFDEstimator.build_pipeline()

# Load the relevant sequences
with open('data/Clostridioides_difficile_630.gb') as handle:
    genome = list(SeqIO.parse(handle, 'genbank'))[0]

with open('data/addgene-plasmid-54622-sequence-158642.gbk') as handle:
    plasmid = list(SeqIO.parse(handle, 'genbank'))[0]

for rec in plasmid.features:
    if rec.qualifiers.get('product', [''])[0].startswith('enhanced GFP'):
        egfp_feature = rec
        break

egfp_record = egfp_feature.extract(plasmid)

# Extract all possible NGG targets
possible_targets = utils.extract_possible_targets(egfp_record, pams=('NGG',))

# Find all targets across the host genome
possible_binding = utils.cas_offinder(possible_targets, 5, locus = [genome])

# Score each hit across the genome
possible_binding['Score'] = estimator.predict_proba(possible_binding.values)

# Find the maximum score for each spacer
results = possible_binding.groupby('spacer')['Score'].agg('max')
results.sort_values(inplace=True)
```

We find that there are at least 5 potential spacers with no off-target potential.
Using the BioPython GenomeDiagram module we can show where these spacers will bind across the plasmid.

 - Summary Figure
 
## Task

The previous task involved searching for the spacers against a single gene.
However, one may want to generate a CRISPR/Cas9 knockout library.
In this instance we want to find spacers targeting each gene individually while avoiding the rest of the genome.
For this example we will continue with spCas9 and generate the knock-out library for each gene in the Clostridioides difficile genome.
With the crisprtree library this is a trivial expansion of the previous task.

```python
library_grnas = []
gene_info = []

genome_key = genome.id + ' ' + genome.description
for feat in genome.features:

    # Only target genes
    if feat.type == 'CDS':
        # Get info about this gene
        product = feat.qualifiers['product'][0]
        tag = feat.qualifiers['locus_tag'][0]
        gene_record = feat.extract(genome)

        # Get potential spacers
        possible_targets = utils.extract_possible_targets(gene_record)

        # Score hits
        possible_binding = utils.cas_offinder(possible_targets, 3, locus = [genome])
        possible_binding['Score'] = estimator.predict_proba(possible_binding.values)

        # Set hits within the gene to np.nan
        slic = pd.IndexSlice[genome_key, :, feat.location.start:feat.location.end ]
        possible_binding.loc[slic, 'Score'] = np.nan

        # Aggregate and sort off-target scan
        offtarget_scores = possible_binding.groupby('spacer')['Score'].agg('max')
        offtarget_scores.fillna(0, inplace=True) # Spacers which only have intragenic hit
        offtarget_scores.sort_values(inplace=True)

        # Save information
        gene_info.append({'Product': product,
                          'Tag': tag,
                          'UsefulGuides': (offtarget_scores<=0.25).sum()})


        for protospacer, off_score in offtarget_scores.head().to_dict().items():
            location = genome.seq.find(protospacer.back_transcribe())
            strand = '+'
            if location == -1:
                location = genome.seq.find(reverse_complement(protospacer.back_transcribe()))
                strand = '-'

            library_grnas.append({'Product': product,
                                  'Tag': tag,
                                  'Protospacer': protospacer,
                                  'Location': location,
                                  'Strand': strand,
                                  'Off Target Score': off_score})

gene_df = pd.DataFrame(gene_info)
library_df = pd.DataFrame(library_grnas)
```

 - Summary Figure

From this we learn that most gene loci can be targeted by spacers which are unique to that single gene.
However, there are a handful of genes which cannot be uniquely targeted by SpCas9.
In this instance one can simply ignore those genes or utlize one of the many Cas9 variants implemented in the `data/cas-variants` folder.

## Task

Leaving the C. diff genome we explore another potential task.
In this we will explore the effect of spacer targeting against a variable population.
We will use data from a recently published anti-HIV-1 CRISPR/Cas9 treatment paper by Sullivan et al [PMID: ].
In this assay equimolor ratios of plasmids containing a mixture of patient derived HIV-1 sequences were cleaved with an IVT CRISPR/Cas9 complex.
The fraction of cleaved plasmids was measured using a Bioanalyzer when exposed to different spacers.
We will use crisprtree to evaluate how well each of the three estimators: MITEstimator, CFDEstimator, and the KineticEstimator predicts the observed values.

```python
# Load in the known data
cutting_data = pd.read_csv('data/IVCA_gRNAs_efficiency.csv')

# Text objects need to be converted to Bio.Seq objects with the correct Alphabet
cutting_data['target'] = list(utils.smrt_seq_convert('Seq',
                                                     cutting_data['Target'].values,
                                                     alphabet=Alphabet.generic_dna))

spacer_dna = utils.smrt_seq_convert('Seq',
                                    cutting_data['gRNA Sequence'].values,
                                    alphabet=Alphabet.generic_dna)
cutting_data['spacer'] = [s.transcribe() for s in spacer_dna]

# Predict the score using each model across all sequences
ests = [('MIT', estimators.MITEstimator.build_pipeline()),
        ('CFD', estimators.CFDEstimator.build_pipeline()),
        ('Kinetic', estimators.KineticEstimator.build_pipeline())]

for name, est in ests:
    # Models only require the gRNA and Target sequence columns
    data = cutting_data[['spacer', 'target']].values
    cutting_data[name] = est.predict_proba(data)

# Aggregate the predicted cutting data across each sample
predicted_cleavage = pd.pivot_table(cutting_data,
                                    index = 'Patient',
                                    columns = 'gRNA Name',
                                    values = ['MIT', 'CFD', 'Kinetic'],
                                    aggfunc = 'mean')
```

After aggregation these results can be plotted against the known results.

 - Summary Figure
 
## Task 4

Finally, we will explore merging crisprtree modules with the rest of the Scikit-learn.
We will attempt to build a new predictor using the results from a GUIDE-Seq experiment by Tsai et al. [PMID: ]
In this data the researchers used *X* spacers targeting genes in the human genome.

```python

files = sorted(glob.glob('data/GUIDESeq/*/*.tsv'))

hit_data = pd.concat([pd.read_csv(f, sep='\t').reset_index() for f in files],
                     axis=0, ignore_index=True)

# Multiple data loading/cleaning steps have been omitted.
# See Supplemental files for the entire script.

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# Convert gRNA-Target pairs into a binary "matching" vector
X = preprocessing.MatchingTransformer().transform(hit_data[['spacer', 'target']].values)
y = (hit_data['NormSum'] >= cutoff).values

# Evaluate the module using 3-fold cross-validation
grad_res = cross_validate(GradientBoostingClassifier(), X, y,
                          scoring = ['accuracy', 'precision', 'recall'],
                          cv = StratifiedKFold(random_state=0),
                          return_train_score=True)
grad_res = pd.DataFrame(grad_res)

# Evaluate MITEstimator using 3-fold cross-validation
mit_res = cross_validate(estimators.MITEstimator(), X, y,
                         scoring = ['accuracy', 'precision', 'recall'],
                         cv = StratifiedKFold(random_state=0),
                         return_train_score=True)
mit_res = pd.DataFrame(mit_res)
```

 - Summary Figure
 
# Discussion & Conclusion

We believe that as the CRISPR/Cas9 gene-editting field expands the number of potential strategies will increase as well.
It is infeasable to build single purpose-driven tool to account for all of these potential strategies.
For example, nickase based strategies require finding targets with potential spacers on each strand at the same position while homologous recombination strategies require finding targets a specific distance away.
While it may be easy to develop single instances of these strategies by hand, it would be impractile to do across many (potentially thousands) of genes.
As such, we have built this library in a modular way to allow researchers to leverage the power of Python to design gene-editting strategies.

The crisprtree module can also be used to explore the use of Cas9 gene-editting in non-model organisms.
Even "draft quality" genome sequences containing degenerate bases can be searched.
This further includes the use of differing Cas9 variants that can be explored easily.
Researchers can also integrate crisprtree into the Scikit-Learn ecosystem to create more advanced estimators as data becomes available.

We also show that crisprtree can be used to evaluate the effectiveness of CRISPR/Cas9 gene editting on variable populations.
This will be an invaluable tool for fields editting highly mutable genomes or meta-genomes of mixed populations.
For example, HIV-1 has such a high mutation rate that within a single individual there exists a swarm of distinct genomes; crisprtree can be used to measure the effectiveness across the spectrum of available genomes.
Additionally, CRISPR gene editting has been proposed as a strategy for shifting the human microbiome by selectively targetting specific species.
The crisprtree module can measure the effectiveness across each composed genome and measure arbitrarly complicated targetting rules (eg. target genomes A-E, miss genomes F-J).

We also believe that crisprtree will be useful to researchers exploring the binding rules of non-model Cas9 variants.
In Task 4 we showed the ease of training a new estimator given a set of binding data.
Using a Cas9 variant in an genome-wide binding experiment such as GUIDE-Seq, CIRLCE-Seq, or SITE-Seq, can provide invaluable training data.
The crisprtree module can be easily merged with the Scikit-learn architechure or even the more advanced Deep Learning frameworks such as Keras [REF] or Tensorflow [REF].

All of these features make crisprtree an invaluable resource for computational biologists exploring the intricacies of CRISPR/Cas9 gene editing.