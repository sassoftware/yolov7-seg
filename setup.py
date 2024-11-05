#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  This project is licensed under the GNU GENERAL PUBLIC LICENSE 3.0 License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.gnu.org/licenses/gpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' Install the SAS YOLOv7 Seg package '''

import io
import os
from setuptools import setup, find_packages


def get_file(fname):
    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
                 encoding='utf8') as infile:
        return infile.read()


setup(
    name='sas-yolov7-seg',
    version='1.0.3',
    description='SAS YOLOv7 Seg',
    long_description=get_file('README.md'),
    long_description_content_type='text/markdown',
    author='SAS',
    author_email='support@sas.com',
    url='https://github.com/sassoftware/yolov7-seg/',
    license='GNU GENERAL PUBLIC LICENSE 3.0',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
)