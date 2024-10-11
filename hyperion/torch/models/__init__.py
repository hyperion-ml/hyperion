"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

from .transducer import RNNRNNTransducer, RNNTransducer
from .vae.vae import VAE
from .vae.vq_vae import VQVAE
from .wav2transducer import (  # HFWav2Vec2Transducer,
    HFWav2Vec2ConformerV1RNNTransducer,
    HFWav2Vec2RNNRNNTransducer,
    HFWav2Vec2RNNTransducer,
    Wav2ConformerV1RNNTransducer,
    Wav2RNNRNNTransducer,
)
from .wav2xvectors import (
    HFHubert2ConformerV1XVector,
    HFHubert2ResNet1dXVector,
    HFWav2Vec2ConformerV1XVector,
    HFWav2Vec2ResNet1dXVector,
    HFWavLM2ConformerV1XVector,
    HFWavLM2ResNet1dXVector,
    Wav2ConformerV1XVector,
    Wav2ConvNext1dXVector,
    Wav2ConvNext2dXVector,
    Wav2ResNet1dXVector,
    Wav2ResNetXVector,
)
from .xvectors import (
    ConformerV1XVector,
    ConvNext1dXVector,
    ConvNext2dXVector,
    EfficientNetXVector,
    ResNet1dXVector,
    ResNetXVector,
    SpineNetXVector,
    TDNNXVector,
    TransformerV1XVector,
    XVector,
)
