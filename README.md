crseek
======

.. image:: https://travis-ci.org/DamLabResources/crseek.png?branch=master
    :target: https://travis-ci.org/DamLabResources/crseek

.. image:: https://coveralls.io/repos/github/DamLabResources/crseek/badge.svg?branch=master)
    :target: https://coveralls.io/github/DamLabResources/crseek?branch=master

`crseek` is a tool for designing complex CRISPR-Cas9 workflows using the popular `SKlearn` API.


Quick Start
-----------

For the impatient:

```python
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna, generic_rna

from crseek.preprocessing import OneHotTransformer
from crseek.estimators import CFDEstimator
from crseek.utils import cas_offinder

gRNA1 = Seq('ATGTTGAGTCAGTGAAGGTG', alphabet = generic_rna)
gRNA2 = Seq('ATGTTCAGTCAGTAAAGGTG', alphabet = generic_rna)

possible_hits = cas_offinder([gRNA1, gRNA2], 5, direc='/path/to/genome/files/')

cfd_est =  CFDEstimator.build_pipeline()

target_scores = cfd_est.predict_proba(possible_hits.values)
```

See the notebooks in the `paper/` folder for more examples.

Features
--------


### Preprocessing

 - `crseek.preprocessing.MatchingTransformer`
 - `crseek.preprocessing.OneHotTransformer`

These modules are modeled after the preprocessing modules in `Sklearn` and are capable of comparing (`spacer`, `target`) pairs. The
 outputs of these modules can either be fed further into the `crseek` estimation tools or used for building one's own machine
 learning methods.

### Estimation

 - `crseek.estimators.MismatchEstimator`
 - `crseek.estimators.MITEstimator`
 - `crseek.estimators.CFDEstimator`
 - `crseek.estimators.KineticEstimator`

These modules can take the outputs of preprocessed (`spacer`, `target`) pairs and estimate the likelihood of cleavage under numerous
assumptions. These modules implement `.fit()`, `.predict()`, and `.predict_proba()` methods and are usable in all `Sklearn` tools.

### Utilities

 - `crseek.utils.smrt_seq_convert`
 - `crseek.utils.extract_possible_targets`
 - `crseek.utils.tile_seqrecord`
 - `crseek.utils.cas_offinder`

There is also a collection of utilities for manipulating sequence types, finding potential targets in sequences, and searching
long (potentially genome scale) sequences for cutting events. This includes a wrapper around the popular [`cas_offinder`](https://github.com/snugel/cas-offinder)
tool for mismatch searches of large genomes using OpenCL-enabled devices. The searching methods have also been adapted to work with any
 potential PAM allowing the tool to be used with non-standard Cas9 variants.

### Additional Features

 - Tight integration with `BioPython` allows for the easy import, search, manipulation, and export of sequences.
 - Integration with `PySam` allows for the search of SAM/BAM files.
 - Preprocessing and Estimation tools intelligently deal with ambigious base calls in sequences allowing the tool to be
   used on draft-quality genomes.


Installation
------------

Currently the installable modules are split between `pip` and `conda` tools due to multiple non-python dependencies.

```shell
conda config --add channels conda-forge
conda config --add channels defaults
conda config --add channels r
conda config --add channels bioconda
conda create -yq -n crseek-environment --file requirements.conda
source activate crseek-environment
pip install -q -r requirements.pip
python setup.py install
```
