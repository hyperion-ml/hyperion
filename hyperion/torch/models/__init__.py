"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

from .transducer import RNNRNNTransducer, RNNTransducer
from .vae.vae import VAE
from .vae.vq_vae import VQVAE
from .wav2transducer import (
    HFWav2Vec2ConformerV1RNNTransducer,
    HFWav2Vec2RNNRNNTransducer,
    HFWav2Vec2RNNTransducer,
    HFWav2Vec2Transducer,
)
from .wav2xvectors import (
    HFHubert2ConformerV1XVector,
    HFHubert2ResNet1dXVector,
    HFWav2Vec2ConformerV1XVector,
    HFWav2Vec2ResNet1dXVector,
    HFWavLM2ConformerV1XVector,
    HFWavLM2ResNet1dXVector,
    Wav2ConformerV1XVector,
    Wav2ResNet1dXVector,
    Wav2ResNetXVector,
)
from .xvectors.conformer_v1_xvector import ConformerV1XVector
from .xvectors.efficient_net_xvector import EfficientNetXVector
from .xvectors.resnet1d_xvector import ResNet1dXVector
from .xvectors.resnet_xvector import ResNetXVector
from .xvectors.spinenet_xvector import SpineNetXVector
from .xvectors.tdnn_xvector import TDNNXVector
from .xvectors.transformer_xvector_v1 import TransformerXVectorV1
from .xvectors.xvector import XVector
