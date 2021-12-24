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
from pathlib import Path

project_root = Path(__file__).parent

with open(project_root / "apps.txt") as f:
    apps = f.read().splitlines()

apps = [str(project_root / "hyperion" / "bin" / app) for app in apps]

with open(project_root / "requirements.txt") as f:
    requirements = f.read().splitlines()

with open(project_root / "README.md", "r") as fh:
    long_description = fh.read()


def get_version():
    with open(project_root / "hyperion" / "__init__.py") as f:
        for line in f.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]

        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    # This name will decide what users will type when they install your package.
    name="hyperion-ml",
    # The version of your project.
    # Usually, it would be in the form of:
    # major.minor.patch
    # eg: 1.0.0, 1.0.1, 3.0.2, 5.0-beta, etc.
    # You CANNOT upload two versions of your package with the same version number
    version=get_version(),
    author="Jesus Villalba",
    author_email="jesus.antonio.villalba@gmail.com",
    # The description that will be shown on PyPI.
    # Keep it short and concise
    # This field is OPTIONAL
    description="Toolkit for speaker recognition",
    # The content that will be shown on your project page.
    # In this case, we're displaying whatever is there in our README.md file
    # This field is OPTIONAL
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="speaker recognition, adversarial attacks, NIST SRE, x-vectors",
    url="https://github.com/hyperion-ml/hyperion",
    # The packages that constitute your project.
    # For my project, I have only one - "pydash".
    # Either you could write the name of the package, or
    # alternatively use setuptools.findpackages()
    #
    # If you only have one file, instead of a package,
    # you can instead use the py_modules field instead.
    # EITHER py_modules OR packages should be present.
    packages=setuptools.find_packages(exclude="tests"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    scripts=apps,
)
