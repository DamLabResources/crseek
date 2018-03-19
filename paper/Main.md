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
Currently distributed software lacks unit-testing, continious integration frameworks, or a documented API.
These features are needed in order for bioinformaticians to design larger scale experiments or develop automation methods for more advanced gene editing strategies.

## Our stuff
In this manuscript we present CRISPRTree, a Python library mirroring the popular Scikit-Learn API.
It provides both high- and low-level methods for designing CRISPR gene editing strategies.
We believe it will provide an invaluable resource for computational biologists looking to employ CRISPR based gene editing techniques.

# Implementation

## Class schematic

We have intentionally mirrored the Scikit-Learn API.
We subclass the `BaseEstimator` class and overload the `fit`, `transform`, and `predict` methods.
This allows us to seamlessly use all of the tools of the Scikit-Learn package such as cross-validation, normalization, or other prediciton strategies.
We have decomposed this methodology into three main tasks: Searching, Encoding, and Predicting.

## Searching

We define `search`ing as the act of locating potential binding sites of a Cas9/protospacer in a DNA segment.
DNA segments can be any files, or directories of files, readable by `Bio.SeqIO` (fasta, genbank, etc), lists of `Bio.SeqRecord` objects, or `np.array`s of charachers.
Our tool provides two stragies for this searching: exhaustive or mismatch based.
In an exhausitve search all potential positions are considered as potiential targets.
For mismatch searching we have created a wrapper around the popular `cas-offinder` library [PMID: ].
This library allows for rapid mismatch searching, even employing the GPU if present and properly installed.
The `utils.tile_seqrecord` and `utils.cas_offinder` return arrays of (spacer, target) pairs for downstream analysis.

## Encoding

Once targets have been found on the genomic loci we must encode the (spacer, target) pairs for downstream analysis.
As of this writing, there are two main strategies for this encoding: missmatch vs one-hot encoding.
Rule-based mismatching and the Hsu et al [PMID: ] strategies both give the same weight independenet of the mismatch identity; as such this encoding is a binary vector of length 21 (20bp for spacer, 1 for PAM matcing).


 -  Accepting many types of Sequence objects
 -  Subclasses the Transform object
 -  Mismatch encoder
 -  One-hot encoder

## Prediction
 - Subclass of BaseEstimator
 - Rule based mismatch
 - MIT estimator
 - CFD eestimator
 
## Ranking Metrics
 - MIT off target score
 - Hairpin score

## IO Options
 - Anything readable by Bio.SeqIO
 - Anything readable by PySam


# Uses

## Task
 You are tasked with introducing an eGFP plasmid into the newly sequenced Clostridioides difficile genome.
 You want to make an spCas9 positive control knocking out the eGFP gene while missing the rest of the newly 
 sequenced genome. The following code-snippet walks through a basic execution strategy.
 
 1. Load the plasmid genbank file and find the eGFP gene.
 2. Find all possible protospacers within the gene.
 3. Load in the Thermofilum genome.
 4. Scan all possible protospacers against the genome using the CFD matrix
 5. Annotate the 5 with the lowest off-target likelihood.
 
 - Code Snippet using API
 - Code Snippet using scripts
 - Summary Figure
 
## Task
 You are tasked with creating an spCas9 knock-out library for each gene in the Clostridioides difficile genome.
 You want to find 5 protospacers for each gene which have a minimum level of off-target potential.
 
 1. Load in the Clostridioides difficile genome.
 2. Iterate through all CDS annotations in the file.
     1. Extract all possible protospacers
     2. Scan all possible protospacers against the genome using the CFD matrix
     3. Annotate the 5 with the lowest off-target likelihood.
 3. Re-annotate the Genbank record.
 4. Output new genbank record and spreadsheet.
 
 - Code Snippet using API
 - Summary Figure

## Task
 Evaluating gRNA binding in a variable population
 
 1. Load in plasmid sequences
 2. Extract best hit from each sequence
 3. Evaluate effectiveness using multiple estimators
 4. Compare predicted vs observed in IVT-cutting assay
 
 - Code Snippet using API
 - Summary Figure
 
## Task 4
 Training new predictor using GUIDE-Seq data
 
 1. Load in GS data
 2. Normalize counts using a rank scale
 3. Select random positions on the genome as negatives
 4. Make pipeline of one-hot encoder and GrandientBoosting
 5. Train using cross validation
 6. Compare with CFD, MIT, and MissMatch rules
 
 - Code Snippet using API
 - Summary Figure
 
# Discussion

## Training against a variable population
 - HIV
 - Microbiome

## Developing Cas9 knockout libraries for non-standard things
 - Different Cas9 variants
 - Draft genomes
 
## Training off-target rules for novel Cas9s
 - Exploring different Cas9 variants
 
# Conclusion
  
  Crisprtree is awesome. Everyone should use it.