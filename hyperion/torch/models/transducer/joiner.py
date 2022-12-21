# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
import torch
import torch.nn as nn


class Joiner(nn.Module):
    def __init__(self, in_feats: int, out_dims: int, num_layers: int):
        super().__init__()
        self.in_feats = in_feats
        self.out_dims = out_dims
        self.num_layers = num_layers

        self.output_linear = nn.Linear(in_feats, out_dims)

    def forward(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, C).
          decoder_out:
            Output from the decoder. Its shape is (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        # print("encoder_out",encoder_out.shape)
        # print("decoder_out",decoder_out.shape)
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == decoder_out.size(2)

        encoder_out = encoder_out.unsqueeze(2)
        # Now encoder_out is (N, T, 1, C)

        decoder_out = decoder_out.unsqueeze(1)
        # Now decoder_out is (N, 1, U, C)

        logit = encoder_out + decoder_out
        logit = torch.tanh(logit)

        output = self.output_linear(logit)

        return output


    def get_config(self):
        config = {
            "in_feats" : self.in_feats,
            "out_dims": self.out_dims,
            "num_layers": self.num_layers,
        }

        # base_config = super().get_config()
        return dict(list(config.items()))


    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "in_feats",
            "out_dims",
            "num_layers",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args


    @staticmethod
    def add_class_args(parser, prefix=None, skip=set(["in_feats", "out_dims"])):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats", type=int, required=True, help=("input feature dimension")
            )

        if "out_dims" not in skip:
            parser.add_argument(
                "--out-dims", type=int, required=True, help=("output feature dimension (vocab size)")
            )
        parser.add_argument(
            "--num-layers", default=1, type=int, help=("layers of the joiner")
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


    # @staticmethod
    # def add_class_args(parser, prefix=None, skip=set()):

    #     parser.add_argument(
    #         "--encoder-out-dim", default=512, type=int, help=("")
    #     )