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
    description='Deploy static HTML sites to S3 with the simple "alotofeffort" command.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Will Dampier',
    author_email='judowill@gmail.com',
    url='https://github.com/judowill/crseek',
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
