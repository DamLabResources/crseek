#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.md').read()
doclink = """
Documentation
-------------

The full documentation is at http://crseek.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='crseek',
    version='0.1.0',
    description='Search for CRISPR-Cas9 targets in a genome.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Will Dampier',
    author_email='wnd22@drexel.edu',
    url='https://github.com/DamLabResources/crseek',
    packages=[
        'crseek',
    ],
    package_dir={'crseek': 'crseek'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='crseek',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.3',
    ],
)
