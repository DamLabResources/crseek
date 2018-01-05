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
 Finding a gRNA cutting a gene of interest but missing the rest of the genome
 
 1. Load gene and find all hits (what gene?)
 2. Scan genome (using non-standard e-coli genome)
 3. Quantify off-targets
 
 - Code Snippet using API
 - Code Snippet using scripts
 - Summary Figure
 
## Task
 Creating a knock-out for every gene in the genome
 
 1. Genome annotation file
 2. Iterate through all genes
 3. Quantify off-targets
 4. Save top X for each gene
 
 - Code Snippet using API
 - Summary Figure

## Task
 Evaluating gRNA binding in a variable population
 
 1. Load in LANL data for region
 2. Extract best hit from each sequence
 3. Evaluate effectivness
 4. Compare predicted vs observed in IVT-cutting assay
 
 - Code Snippet using API
 - Summary Figure
 
## Task 4
 Training new predictor using CIRCLE-Seq data
 
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