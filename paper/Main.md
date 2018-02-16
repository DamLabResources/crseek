# Introduction

- Cas9 uses
- Cas9 binding rules

## Current prediction software
 - BLAST
 - MIT webserver
 - Cas9-off finder

## Lacking features
 -  Pythonic Api
 -  Evaluation of NGS data
 -  Dealing with variation

# Implementation

## Class schematic

## Use of sklearn api
 -  Well tested library
 -  Division of Searching vs Encoding vs Prediction

## Searching
 -  Python api wrapper for cas-offinder
 -  Can load yaml definitions of cas9 variants

## Encoding
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
 You are tasked with introducing an eGFP plasmid into the newly sequenced Thermofilum sp. NZ13-TE1 genome.
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
 You are tasked with creating an spCas9 knock-out library for each gene in the Thermofilum genome. 
 You want to find 5 protospacers for each gene which have a minimum level of off-target potential.
 
 1. Load in the Thermofilum genome.
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
 3. Evaluate effectivness using multiple estimators
 4. Compare predicted vs observed in IVT-cutting assay
 
 - Code Snippet using API
 - Summary Figure
 
## Task 4
 Training new predictor using GUIDE-Seq data
 
 1. Load in CS data
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