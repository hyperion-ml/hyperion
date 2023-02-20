"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

from .xvectors.xvector import XVector
from .xvectors.tdnn_xvector import TDNNXVector
from .xvectors.resnet_xvector import ResNetXVector
from .xvectors.efficient_net_xvector import EfficientNetXVector
from .xvectors.transformer_xvector_v1 import TransformerXVectorV1
from .xvectors.spinenet_xvector import SpineNetXVector
from .xvectors.resnet1d_xvector import ResNet1dXVector

from .wav2xvectors import (
    HFWav2Vec2ResNet1dXVector,
    HFHubert2ResNet1dXVector,
    HFWavLM2ResNet1dXVector,
)


from .wav2transducer import HFWav2Vec2Transducer
from .wav2languageid import HFWav2Vec2ResNet1dLanguageID

from .vae.vae import VAE
from .vae.vq_vae import VQVAE
