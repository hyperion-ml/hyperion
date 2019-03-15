"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import torch
import torch.nn as nn


class TorchModel(nn.Module):

    def get_config(self):
        config = {
            'class_name': self.__class__.__name__}
        
        return config


    def save(self, file_path):
        file_dir = os.path.dirname(file_path)
        if not(os.path.isdir(file_dir)):
            os.makedirs(file_dir, exist_ok=True)

        config = self.get_config()
        torch.save({'config': self.get_config(),
                    'model_state_dict': self.state_dict()})


    def get_reg_loss(self):
        return 0


    def get_loss(self):
        return 0

