
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from abc import ABCMeta, abstractmethod
import json

from ..hyp_model import HypModel

class TFModel(HypModel):

    def __init__(self):
        self.loss = None

    
        
