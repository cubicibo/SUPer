#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 cibo
This file is part of SUPer <https://github.com/cubicibo/SUPer>.

SUPer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SUPer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SUPer.  If not, see <http://www.gnu.org/licenses/>.
"""

from distutils.util import convert_path
from typing import Any

import sys
from setuptools import setup, Extension

NAME = 'SUPer'

#%% metadata plumbing
meta: dict[str, Any] = {}
with open(convert_path(f"{NAME}/__metadata__.py"), encoding='utf-8') as f:
    exec(f.read(), meta)

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', encoding='utf-8') as fh:
    install_requires = fh.read()

#%% Final setup
def setup_pkg():
    setup(
        name=NAME,
        version=meta['__version__'],
        author=meta['__author__'],
        description='Blu-ray HDMV PGS encoder.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=[NAME,],
        package_dir={NAME: '.'},
        package_data={
            NAME: ['py.typed'],
        },
        license="GPLv3",
        url='https://github.com/cubicibo/SUPer',
        zip_safe=False,
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.10',
        install_requires=install_requires,
    )
####
if __name__ == "__main__":
    setup_pkg()
