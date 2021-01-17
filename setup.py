#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import setuptools

with open('apps.txt') as f:
    apps = f.read().splitlines()

apps = ['bin/'+ app for app in apps ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hyperion-jvillalba", # Replace with your own username
    version="0.0.1",
    author="Jesus Villalba",
    author_email="jesus.antonio.villalba@gmail.com",
    description="Toolkit for speaker recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyperion-ml/hyperion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.1',
        'pysoundfile>=0.9.0',
        'h5py>=2.10.0',
        'matplotlib>=3.1.3',
        'pandas>=1.0.1',
        'scikit-learn>=0.22.1',
        'scipy>=1.4.1',
        'sphinx_rtd_theme'],
    scripts=apps
)
