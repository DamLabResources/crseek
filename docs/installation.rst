============
Installation
============

Basic Installation
-----------------

    $ conda config --add channels conda-forge
    $ conda config --add channels bioconda
    $ conda install crisprtree

Due to the nature of the `cas-offinder`, in order to use a GPU for mismatch searching the openci library must be
installed by hand. Instructions can be found by searching for your particular graphics card.

Installing Dependencies Independently
-------------------------------------

Both `conda` and `pip` are needed to install all dependencies.

    $ conda create -n crisprtree python=3.6
    $ source activate crisprtree
    $ conda config --add channels conda-forge
    $ conda config --add channels defaults
    $ conda config --add channels r
    $ conda config --add channels bioconda
    $ conda install --file requirements.conda
    $ pip install -r requirement.txt
