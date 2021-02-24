"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser

import torch.nn as nn

from ..layers import AudioFeatsFactory as AFF
from ..layers import MeanVarianceNorm as MVN
from .net_arch import NetArch


class AudioFeatsMVN(NetArch):
    """Acoustic Feature Extractor + STMVN
    """
    def __init__(self, audio_feats, mvn=None, trans=False):
        super().__init__()
        
        audio_feats = AFF.filter_args(**audio_feats)
        if mvn is not None:
            mvn = MVN.filter_args(**mvn)

        self.audio_feats_cfg = audio_feats
        self.mvn_cfg = {} if mvn is None else mvn
        self.trans = trans

        self.audio_feats = AFF.create(**audio_feats)
        self.mvn = None
        if mvn is None:
            return
    
        if mvn['norm_mean'] or mvn_args['norm_var']:
            self.mvn = MVN(**mvn)


    def forward(self, x):
        f = self.audio_feats(x)
        if self.mvn is not None:
            f = self.mvn(f)

        if self.trans:
            f = f.transpose(1,2).contiguous()
        return f


    def get_config(self):
        config = {
            'audio_feats': self.audio_feats_cfg,
            'mvn': self.mvn_cfg,
            'trans': self.trans
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @staticmethod
    def filter_args(**kwargs):
        valid_args = ('audio_feats', 'mvn', 'trans')
        return dict((k, kwargs[k])
                    for k in valid_args if k in kwargs)


    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog='')

        AFF.add_class_args(parser, prefix='audio_feats')
        MVN.add_class_args(parser, prefix='mvn')
        if prefix is not None:
            outer_parser.add_argument(
                '--' + prefix,
                action=ActionParser(parser=parser),
                help='feature extraction options')

        
