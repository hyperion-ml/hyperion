#!/usr/bin/env python

"""
Loads data to train UBM, i-vector
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time
import copy

import numpy as np

from ..io import HypDataReader
from ..utils.scp_list import SCPList
from ..utils.tensors import to3D_by_class
from ..transforms import TransformList

class SequenceLoader(object):

    
