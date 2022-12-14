#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:52:04 2022

@author: cibo
"""

from distutils.util import convert_path
from typing import Any, Dict

from setuptools import setup

meta: Dict[str, Any] = {}
with open(convert_path('SUPer/__metadata__.py'), encoding='utf-8') as f:
    exec(f.read(), meta)

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', encoding='utf-8') as fh:
    install_requires = fh.read()

NAME = 'BDSUPer'

setup(
    name=NAME,
    version=meta['__version__'],
    author=meta['__author__'],
    #author_email=meta['__email__'],
    description='Blu-Ray SUP editor and optimiser.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['SUPer',],
    package_data={
        'BDSUPer': ['py.typed'],
    },
    url='https://github.com/cubicibo/SUPer',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=install_requires,
)