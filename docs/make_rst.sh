#!/bin/bash

export SPHINX_APIDOC_OPTIONS="members,private-members,undoc-members,show-inheritance"

sphinx-apidoc -P -l -d 10 -f -o ./hyperion ../hyperion

