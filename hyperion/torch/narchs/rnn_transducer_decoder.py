"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from jsonargparse import ActionParser, ArgumentParser
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

from ...utils.misc import filter_func_args
from ...utils.text import add_sos
from ..layer_blocks import TransducerPredictor as Predictor, TransducerJoiner as Joiner
from .net_arch import NetArch


@dataclass
class Hypothesis:
    ys: List[int]  # predicted sequences
    log_prob: float  # log prob of ys

    # Optional LSTM predictor state.
    pred_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class RNNTransducerDecoder(NetArch):
    """ RNN-T Decoder composed of Predictor and Joiner networks
    Implementation based on 
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/transducer.py

    Attributes:
      in_feats: input features dimension (encoder output)
      vocab_size: Number of tokens of the modeling unit including blank.
      embed_dim: Dimension of the predictor input embedding.
      blank_id: The ID of the blank symbol.
      num_layers: Number of LSTM layers.
      hid_feats: Hidden dimension for predictor layers.
      embed_dropout_rate: Dropout rate for the embedding layer.
      rnn_dropout_rate: Dropout for LSTM layers.

    """

    def __init__(self,
                 in_feats: int,
                 vocab_size: int,
                 embed_dim: int,
                 num_pred_layers: int,
                 pred_hid_feats: int,
                 embed_dropout_rate: float = 0.0,
                 rnn_dropout_rate: float = 0.0,
                 rnn_type: str = "lstm",
                 blank_id: int = 0):

        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_pred_layers = num_pred_layers
        self.pred_hid_feats = pred_hid_feats
        self.embed_dropout_rate = embed_dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.rnn_type = rnn_type
        self.blank_id = blank_id

        pred_args = filter_func_args(Predictor.__init__, locals())
        pred_args["num_layers"] = num_pred_layers
        pred_args["hid_feats"] = pred_hid_feats
        pred_args["out_feats"] = in_feats
        self.predictor = Predictor(**pred_args)
        self.joiner = Joiner(in_feats, vocab_size)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor,
                y: k2.RaggedTensor) -> torch.Tensor:

        # get y_lengths
        row_splits = y.shape.row_splits(1)
        y_lengths = row_splits[1:] - row_splits[:-1]
        # shift y adding <sos> token
        sos_y = add_sos(y, sos_id=self.blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=self.blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)
        # apply predictor and joiner
        pred_out, _ = self.predictor(sos_y_padded)
        logits = self.joiner(x, pred_out)
        # rnnt_loss requires 0 padded targets
        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)
        x_lengths = x_lengths.to(torch.int32)
        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded.to(torch.int32),
            logit_lengths=x_lengths,
            target_lengths=y_lengths,
            blank=self.blank_id,
            reduction="sum",
        )
        return logits, loss

    def decode(self,
               x: torch.Tensor,
               x_lengths: torch.Tensor = None,
               method="time_sync_beam_search",
               beam_width: int = 5,
               max_sym_per_frame: int = 3,
               max_sym_per_utt: int = 1000) -> List[int]:
        if method == "time_sync_beam_search":
            return self.decode_time_sync_beam_search(x,
                                                     x_lengths,
                                                     beam_width=beam_width)
        elif method == "align_length_sync_beam_search":
            return self.decode_align_length_sync_beam_search(
                x,
                x_lengths,
                beam_width=beam_width,
                max_sym_per_utt=max_sym_per_utt)
        elif method == "greedy":
            return self.decode_greedy(x,
                                      x_lengths,
                                      max_sym_per_frame=max_sym_per_frame,
                                      max_sym_per_utt=max_sym_per_utt)

    def decode_greedy(self,
                      x: torch.Tensor,
                      x_lengths: torch.Tensor = None,
                      max_sym_per_frame: int = 3,
                      max_sym_per_utt: int = 1000) -> List[int]:
        """
        Args:
          x: encoder embeddings with shape = (N, T, C)
        Returns:
          Decoded tokens
        """
        assert x.ndim == 3

        # support only batch_size == 1 for now
        assert x.size(0) == 1, x.size(0)
        blank_id = self.blank_id
        device = x.device

        sos = torch.tensor([blank_id], device=device,
                           dtype=torch.int64).reshape(1, 1)
        pred_out, (h, c) = self.predictor(sos)
        T = x.size(1)
        t = 0
        hyp = []

        sym_per_frame = 0
        sym_per_utt = 0

        while t < T and sym_per_utt < max_sym_per_utt:
            x_t = x[:, t:t + 1, :]
            logits = self.joiner(x_t, pred_out)  # (1, 1, 1, vocab_size)
            # logits is

            log_prob = logits.log_softmax(dim=-1)  # (1, 1, 1, vocab_size)
            # TODO: Use logits.argmax()
            y = log_prob.argmax()
            if y != blank_id:
                hyp.append(y.item())
                y = y.reshape(1, 1)
                pred_out, (h, c) = self.predictor(y, (h, c))

                sym_per_utt += 1
                sym_per_frame += 1

            if y == blank_id or sym_per_frame > max_sym_per_frame:
                sym_per_frame = 0
                t += 1

        return hyp

    def decode_time_sync_beam_search(self,
                                     x: torch.Tensor,
                                     x_lengths: torch.Tensor = None,
                                     beam_width: int = 5) -> List[int]:
        assert x.ndim == 3
        assert x.size(0) == 1, x.size(0)

        blank_id = self.blank_id
        device = x.device

        sos = torch.tensor([blank_id], device=device).reshape(1, 1)
        pred_out, (h, c) = self.predictor(sos)
        T = x.size(1)
        t = 0
        B = [Hypothesis(ys=[blank_id], log_prob=0.0, pred_state=None)]
        max_u = 20000  # terminate after this number of steps
        u = 0

        cache: Dict[str, Tuple[torch.Tensor, Tuple[torch.Tensor,
                                                   torch.Tensor]]] = {}

        while t < T and u < max_u:
            x_t = x[:, t:t + 1, :]
            A = B
            B = []

            while u < max_u:
                y_star = max(A, key=lambda hyp: hyp.log_prob)
                A.remove(y_star)

                # Note: y_star.ys is unhashable, i.e., cannot be used
                # as a key into a dict
                cached_key = "_".join(map(str, y_star.ys))

                if cached_key not in cache:
                    pred_in = torch.tensor([y_star.ys[-1]],
                                           device=device).reshape(1, 1)

                    pred_out, pred_state = self.predictor(
                        pred_in,
                        y_star.pred_state,
                    )
                    cache[cached_key] = (pred_out, pred_state)
                else:
                    pred_out, pred_state = cache[cached_key]

                logits = self.joiner(x_t, pred_out)
                log_prob = logits.log_softmax(dim=-1)
                # log_prob is (1, 1, 1, vocab_size)
                log_prob = log_prob.squeeze()
                # Now log_prob is (vocab_size,)

                # If we choose blank here, add the new hypothesis to B.
                # Otherwise, add the new hypothesis to A

                # First, choose blank
                skip_log_prob = log_prob[blank_id]
                new_y_star_log_prob = y_star.log_prob + skip_log_prob.item()
                # print("tuAB0", t, u, len(y_star.ys), y_star.log_prob,
                #       skip_log_prob.item(), new_y_star_log_prob)
                # ys[:] returns a copy of ys
                new_y_star = Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    # Caution: Use y_star.decoder_state here
                    pred_state=y_star.pred_state,
                )
                B.append(new_y_star)

                topk_log_prob = log_prob.topk(beam_width, dim=-1)

                # Second, choose other labels
                #for i, v in enumerate(log_prob.tolist()):
                for v, i in zip(*topk_log_prob):
                    v = v.item()
                    i = i.item()
                    if i == blank_id:
                        continue
                    new_ys = y_star.ys + [i]
                    new_log_prob = y_star.log_prob + v
                    new_hyp = Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        pred_state=pred_state,
                    )
                    A.append(new_hyp)

                u += 1
                # check whether B contains more than "beam" elements more probable
                # than the most probable in A
                A_most_probable = max(A, key=lambda hyp: hyp.log_prob)
                #print("tuAB1", t, u, len(A), A_most_probable.log_prob, len(B))
                B = sorted(
                    [
                        hyp
                        for hyp in B if hyp.log_prob > A_most_probable.log_prob
                    ],
                    key=lambda hyp: hyp.log_prob,
                    reverse=True,
                )
                # print("tuAB2",
                #       t,
                #       u,
                #       len(A),
                #       A_most_probable.log_prob,
                #       len(B),
                #       flush=True)
                if len(B) >= beam_width:
                    B = B[:beam_width]
                    break
            t += 1

        best_hyp = max(B,
                       key=lambda hyp: hyp.log_prob / max(1, len(hyp.ys[1:])))
        ys = best_hyp.ys[1:]  # [1:] to remove the blank
        return ys

    def decode_align_length_sync_beam_search(
            self,
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            beam_width: int = 5,
            max_sym_per_utt: int = 1000) -> List[int]:
        assert x.ndim == 3
        assert x.size(0) == 1, x.size(0)

        blank_id = self.blank_id
        device = x.device

        sos = torch.tensor([blank_id], device=device).reshape(1, 1)
        pred_out, (h, c) = self.predictor(sos)
        T = x.size(1)
        #t = 0
        B = [Hypothesis(ys=[blank_id], log_prob=0.0, pred_state=None)]
        #max_u = 20000  # terminate after this number of steps
        #u = 0

        cache: Dict[str, Tuple[torch.Tensor, Tuple[torch.Tensor,
                                                   torch.Tensor]]] = {}
        F = []
        #for t < T and u < max_u:
        for i in range(T + max_sym_per_utt):
            A = []
            for y_star in B:
                #while u < max_u:
                u = len(y_star.ys) - 1
                t = i - u
                if t >= T:
                    continue

                #y_star = max(A, key=lambda hyp: hyp.log_prob)
                #A.remove(y_star)
                x_t = x[:, t:t + 1, :]
                # Note: y_star.ys is unhashable, i.e., cannot be used
                # as a key into a dict
                cached_key = "_".join(map(str, y_star.ys))

                if cached_key not in cache:
                    pred_in = torch.tensor([y_star.ys[-1]],
                                           device=device).reshape(1, 1)

                    pred_out, pred_state = self.predictor(
                        pred_in,
                        y_star.pred_state,
                    )
                    cache[cached_key] = (pred_out, pred_state)
                else:
                    pred_out, pred_state = cache[cached_key]

                logits = self.joiner(x_t, pred_out)
                log_prob = logits.log_softmax(dim=-1)  # (1, 1, 1, vocab_size)
                log_prob = log_prob.squeeze()  # (vocab_size,)

                # First, choose blank
                skip_log_prob = log_prob[blank_id]
                new_y_star_log_prob = y_star.log_prob + skip_log_prob.item()
                # print("tuAB0", t, u, len(y_star.ys), y_star.log_prob,
                #       skip_log_prob.item(), new_y_star_log_prob)
                # ys[:] returns a copy of ys
                new_y_star = Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    # Caution: Use y_star.pred_state here
                    pred_state=y_star.pred_state,
                )
                A.append(new_y_star)
                if t == T - 1:
                    F.append(y_star)

                topk_log_prob = log_prob.topk(beam_width, dim=-1)

                # Second, choose other labels
                #for i, v in enumerate(log_prob.tolist()):
                for v, i in zip(*topk_log_prob):
                    v = v.item()
                    i = i.item()
                    if i == blank_id:
                        continue
                    new_ys = y_star.ys + [i]
                    new_log_prob = y_star.log_prob + v
                    new_hyp = Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        pred_state=pred_state,
                    )
                    A.append(new_hyp)

                # check whether B contains more than "beam_width" elements more probable
                # than the most probable in A
                #A_most_probable = max(A, key=lambda hyp: hyp.log_prob)
                #print("tuAB1", t, u, len(A), A_most_probable.log_prob, len(B))
                B0 = sorted(
                    [hyp for hyp in A],
                    key=lambda hyp: hyp.log_prob,
                    reverse=True,
                )
                B = []
                B_ys = set()
                for hyp in B0:
                    hyp_ys = tuple(hyp.ys)  # to make ys hashable
                    if hyp_ys not in B_ys:
                        B.append(hyp)
                        B_ys.add(hyp_ys)
                # print("tuAB2",
                #       t,
                #       u,
                #       len(A),
                #       A_most_probable.log_prob,
                #       len(B),
                #       flush=True)
                if len(B) >= beam_width:
                    B = B[:beam_width]
                    break

        best_hyp = max(F,
                       key=lambda hyp: hyp.log_prob / max(1, len(hyp.ys[1:])))
        ys = best_hyp.ys[1:]  # [1:] to remove the blank
        return ys

    def change_config(
        self,
        override_dropouts=False,
        embed_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
    ):
        logging.info("changing decoder config")
        self.predictor.change_config(override_dropouts, embed_dropout_rate,
                                     rnn_dropout_rate)

    def get_config(self):

        config = {
            "in_feats": self.in_feats,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_pred_layers": self.num_pred_layers,
            "pred_hid_feats": self.pred_hid_feats,
            "embed_dropout_rate": self.embed_dropout_rate,
            "rnn_dropout_rate": self.rnn_dropout_rate,
            "blank_id": self.blank_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(RNNTransducerDecoder.__init__, kwargs)
        return args

    @staticmethod
    def filter_finetune_args(**kwargs):
        args = filter_func_args(RNNTransducerDecoder.change_config, kwargs)
        return args

    @staticmethod
    def add_class_args(parser,
                       prefix=None,
                       skip=set(["in_feats", "blanck_id", "vocab_size"])):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument("--in-feats",
                                type=int,
                                required=True,
                                help=("input feature dimension"))
        if "blank_id" not in skip:
            parser.add_argument("--blank-id",
                                type=int,
                                default=0,
                                help=("blank id from tokenizer model"))
        if "vocab_size" not in skip:
            parser.add_argument("--vocab-size",
                                type=int,
                                required=True,
                                help=("output prediction dimension"))
        parser.add_argument("--embed-dim",
                            default=1024,
                            type=int,
                            help=("token embedding dimension"))
        parser.add_argument(
            "--embed-dropout-rate",
            default=0.0,
            type=float,
            help=("dropout prob for predictor input embeddings"))
        parser.add_argument("--rnn-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder RNN "))
        parser.add_argument(
            "--rnn-type",
            default="lstm",
            choices=["lstm", "gru"],
            help=(
                "type of recurrent network for thep predictor in [lstm, gru]"))

        parser.add_argument("--num-pred-layers",
                            default=2,
                            type=int,
                            help="""number of layers of the predictor """)

        parser.add_argument("--pred-hid-feats",
                            default=512,
                            type=int,
                            help="""hidden features of the predictor""")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def add_finetune_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--override-dropouts",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to use the dropout probabilities passed in the "
                "arguments instead of the defaults in the pretrained model."))
        parser.add_argument("--embed-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder input embeddings"))
        parser.add_argument("--rnn-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder RNN "))

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
