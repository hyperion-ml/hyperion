"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

# from .wav2tdnn_xvector import Wav2TDNNXVector
from .wav2resnet_xvector import Wav2ResNetXVector

# from .wav2efficient_net_xvector import Wav2EfficientNetXVector
# from .wav2transformer_xvector_v1 import Wav2TransformerXVectorV1
# from .wav2spinenet_xvector import Wav2SpineNetXVector
from .wav2resnet1d_xvector import Wav2ResNet1dXVector

from .hf_wav2vec2resnet1d_xvector import HFWav2Vec2ResNet1dXVector
from .hf_hubert2resnet1d_xvector import HFHubert2ResNet1dXVector
from .hf_wavlm2resnet1d_xvector import HFWavLM2ResNet1dXVector
