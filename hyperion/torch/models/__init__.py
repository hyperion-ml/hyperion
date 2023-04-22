"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

from .vae.vae import VAE
from .vae.vq_vae import VQVAE
from .transducer import RNNTransducer, RNNRNNTransducer
from .wav2languageid import HFWav2Vec2ResNet1dLanguageID
from .wav2transducer import (HFWav2Vec2RNNRNNTransducer,
                             HFWav2Vec2RNNTransducer, HFWav2Vec2Transducer)
from .wav2xvectors import (HFHubert2ResNet1dXVector, HFWav2Vec2ResNet1dXVector,
                           HFWavLM2ResNet1dXVector)
from .wav2transducer_languageid import HFWav2Vec2RNNTransducerResnet1D
from .xvectors.efficient_net_xvector import EfficientNetXVector
from .xvectors.resnet1d_xvector import ResNet1dXVector
from .xvectors.resnet_xvector import ResNetXVector
from .xvectors.spinenet_xvector import SpineNetXVector
from .xvectors.tdnn_xvector import TDNNXVector
from .xvectors.transformer_xvector_v1 import TransformerXVectorV1
from .xvectors.xvector import XVector
