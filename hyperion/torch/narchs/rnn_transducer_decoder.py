"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional
from jsonargparse import ActionParser, ArgumentParser, ActionYesNo

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

from ...utils.misc import filter_func_args
from ...utils.text import add_sos
from ..layer_blocks import TransducerJoiner as Joiner
from ..layer_blocks import TransducerRNNPredictor as RNNPredictor, TransducerConvPredictor as ConvPredictor
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
      predictor: Dictionary with the predictor options.
      joiner: Dictionary with the joiner options.
      blank_id: id of the null symbol.
      rnnt_loss: type of rnn-t loss between torchaudio, k2 or k2_pruned.
      rnnt_type: rnn-t variation between regular, modified or constrained.
      delay_penalty: penalize symbol delay, which is used to make symbol 
        emit earlier.
      reduction: type of reduction for rnn-t loss between sum or mean
      prune_range: how many symbols to keep for each frame in k2 rnn-t 
        pruned loss.
      lm_scale: language model scale in rnn-t smoothed loss.
      am_scale: acoustic model scale in rnn-t smoothed loss.
      simple_loss_scale: weight of rnn-t simple loss when using k2 pruned loss.
      pruned_warmup_steps: number of steps to warm up the k2 rnn-t pruned loss 
        from 0.1 to 1.
    """

    def __init__(
        self,
        in_feats: int,
        vocab_size: int,
        predictor: Dict,
        joiner: Dict,
        blank_id: int = 0,
        rnnt_loss: str = "k2_pruned",
        rnnt_type: str = "regular",
        delay_penalty: float = 0.0,
        reduction: str = "sum",
        prune_range: int = 5,
        lm_scale: float = 0.25,
        am_scale: float = 0.0,
        simple_loss_scale: float = 0.5,
        pruned_warmup_steps: int = 2000,
        # film: bool=False,
    ):

        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size
        self.predictor_args = predictor
        self.joiner_args = joiner
        self.blank_id = blank_id
        self.rnnt_loss = rnnt_loss
        self.rnnt_type = rnnt_type
        self.delay_penalty = delay_penalty
        self.reduction = reduction
        self.prune_range = prune_range
        self.lm_scale = lm_scale
        self.am_scale = am_scale
        self.simple_loss_scale = simple_loss_scale
        self.pruned_warmup_steps = pruned_warmup_steps

        self._make_predictor()
        self._make_joiner()

        if self.rnnt_loss == "k2_pruned":
            self.simple_am_proj = nn.Linear(in_feats, vocab_size)
            self.simple_lm_proj = nn.Linear(self.predictor.out_feats,
                                            vocab_size)
            self.register_buffer("cur_step", torch.as_tensor(0,
                                                             dtype=torch.int))

    def _make_predictor(self):
        pred_type = self.predictor_args["pred_type"]
        self.predictor_args["in_feats"] = self.in_feats
        self.predictor_args["vocab_size"] = self.vocab_size
        self.predictor_args["blank_id"] = self.blank_id
        if pred_type == "rnn":
            pred_args = filter_func_args(RNNPredictor.__init__,
                                         self.predictor_args)
            self.predictor = RNNPredictor(**pred_args)
        elif pred_type == "conv":
            pred_args = filter_func_args(ConvPredictor.__init__,
                                         self.predictor_args)
            self.predictor = ConvPredictor(**pred_args)
        else:
            raise ValueError(f"Unknown predictor type {pred_type}")

    def _make_joiner(self):
        joiner_type = self.joiner_args["joiner_type"]

        if joiner_type == "basic":
            pred_feats = self.predictor_args["out_feats"]
            hid_feats = self.joiner_args["hid_feats"]
            self.joiner = Joiner(self.in_feats, pred_feats, hid_feats,
                                 self.vocab_size)
        else:
            raise ValueError(f"Unknown joiner type {joiner_type}")

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "vocab_size": self.vocab_size,
            "predictor": self.predictor_args,
            "joiner": self.joiner_args,
            "blank_id": self.blank_id,
            "rnnt_loss": self.rnnt_loss,
            "rnnt_type": self.rnnt_type,
            "delay_penalty": self.delay_penalty,
            "reduction": self.reduction,
            "prune_range": self.prune_range,
            "lm_scale": self.lm_scale,
            "am_scale": self.am_scale,
            "simple_loss_scale": self.simple_loss_scale,
            "pruned_warmup_steps": self.pruned_warmup_steps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _rnnt_loss_torchaudio(self, x: torch.Tensor, x_lengths: torch.Tensor,
                              y: torch.Tensor, y_lengths: torch.Tensor,
                              pred_out: torch.Tensor):
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
            reduction=self.reduction,
        )
        return loss

    def _rnnt_loss_k2(self, x: torch.Tensor, x_lengths: torch.Tensor,
                      y: torch.Tensor, y_lengths: torch.Tensor,
                      pred_out: torch.Tensor):
        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4),
                               dtype=torch.int64,
                               device=x.device)
        boundary[:, 2] = y_lengths
        boundary[:, 3] = x_lengths

        logits = self.joiner(x, pred_out)

        with torch.cuda.amp.autocast(enabled=False):
            loss = k2.rnnt_loss(
                logits=logits.float(),
                symbols=y_padded,
                termination_symbol=self.blank_id,
                boundary=boundary,
                rnnt_type=self.rnnt_type,
                delay_penalty=self.delay_penalty,
                reduction=self.reduction,
            )
        return loss

    def _rnnt_loss_k2_pruned(self, x: torch.Tensor, x_lengths: torch.Tensor,
                             y: torch.Tensor, y_lengths: torch.Tensor,
                             pred_out: torch.Tensor):

        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4),
                               dtype=torch.int64,
                               device=x.device)
        boundary[:, 2] = y_lengths
        boundary[:, 3] = x_lengths

        am_simple = self.simple_am_proj(x)
        lm_simple = self.simple_lm_proj(pred_out)
        with torch.cuda.amp.autocast(enabled=False):
            loss_simple, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm_simple.float(),
                am=am_simple.float(),
                symbols=y_padded,
                termination_symbol=self.blank_id,
                lm_only_scale=self.lm_scale,
                am_only_scale=self.am_scale,
                boundary=boundary,
                rnnt_type=self.rnnt_type,
                delay_penalty=self.delay_penalty,
                reduction=self.reduction,
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=self.prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.enc_proj(x),
            lm=self.joiner.pred_proj(pred_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            loss_pruned = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=self.blank_id,
                boundary=boundary,
                rnnt_type=self.rnnt_type,
                delay_penalty=self.delay_penalty,
                reduction=self.reduction,
            )

        if self.cur_step > self.pruned_warmup_steps:
            simple_loss_scale = self.simple_loss_scale
            pruned_loss_scale = 1.0
        else:
            r = self.cur_step / self.pruned_warmup_steps
            simple_loss_scale = 1.0 - r * (1.0 - self.simple_loss_scale)
            pruned_loss_scale = 0.1 + 0.9 * r
            self.cur_step += 1
            print(simple_loss_scale, pruned_loss_scale)

        loss = simple_loss_scale * loss_simple + pruned_loss_scale * loss_pruned

        return loss, loss_simple, loss_pruned

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, y: k2.RaggedTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # get y_lengths
        row_splits = y.shape.row_splits(1)
        y_lengths = row_splits[1:] - row_splits[:-1]
        # shift y adding <sos> token
        sos_y = add_sos(y, sos_id=self.blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=self.blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)
        # apply predictor and joiner
        pred_out, _ = self.predictor(sos_y_padded)
        loss_simple = loss_pruned = None
        if self.rnnt_loss == "k2_pruned":
            loss, loss_simple, loss_pruned = self._rnnt_loss_k2_pruned(
                x, x_lengths, y, y_lengths, pred_out)
        elif self.rnnt_loss == "k2":
            loss = self._rnnt_loss_k2(x, x_lengths, y, y_lengths, pred_out)
        elif self.rnnt_loss == "torchaudio":
            loss_simple = loss_pruned = None
            loss = self._rnnt_loss_torchaudio(x, x_lengths, y, y_lengths,
                                              pred_out)

        return loss, loss_simple, loss_pruned

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

        try:
            best_hyp = max(B,
                            key=lambda hyp: hyp.log_prob / max(1, len(hyp.ys[1:])))
        except:
            return ""
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
        prune_range: Optional[int] = None,
        reduction: Optional[str] = None,
    ):
        logging.info("changing decoder config")
        self.predictor.change_config(override_dropouts, embed_dropout_rate,
                                     rnn_dropout_rate)
        if prune_range is not None:
            self.prune_range = prune_range
        if reduction is not None:
            self.reduction = reduction

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(RNNTransducerDecoder.__init__, kwargs)
        return args

    @staticmethod
    def filter_finetune_args(**kwargs):
        args = filter_func_args(RNNTransducerDecoder.change_config, kwargs)
        return args

    @staticmethod
    def add_pred_args(parser):

        pred_parser = ArgumentParser(prog="")
        pred_parser.add_argument(
            "--pred-type",
            default="rnn",
            choices=["rnn", "conv"],
            help=
            """type of predictor between RNN and Convolutional [rnn, conv]""")
        pred_parser.add_argument("--embed-dim",
                                 default=1024,
                                 type=int,
                                 help=("token embedding dimension"))
        pred_parser.add_argument(
            "--embed-dropout-rate",
            default=0.0,
            type=float,
            help=("dropout prob for predictor input embeddings"))
        pred_parser.add_argument("--rnn-dropout-rate",
                                 default=0.0,
                                 type=float,
                                 help="""dropout prob for decoder RNN """)
        pred_parser.add_argument(
            "--rnn-type",
            default="lstm",
            choices=["lstm", "gru"],
            help=
            """type of recurrent network for thep predictor in [lstm, gru]""")

        pred_parser.add_argument("--num-layers",
                                 default=2,
                                 type=int,
                                 help="""number of layers of the predictor """)

        pred_parser.add_argument("--hid-feats",
                                 default=512,
                                 type=int,
                                 help="""hidden features of the predictor""")
        pred_parser.add_argument("--out-feats",
                                 default=512,
                                 type=int,
                                 help="""output features of the predictor""")
        pred_parser.add_argument("--context-size",
                                 default=2,
                                 type=int,
                                 help="""context length of the convolutional 
                                 predictor, 1->bigram, 2-> trigram,...""")

        parser.add_argument("--predictor",
                            action=ActionParser(parser=pred_parser))

    @staticmethod
    def add_joiner_args(parser):

        pred_parser = ArgumentParser(prog="")
        pred_parser.add_argument(
            "--joiner-type",
            default="basic",
            choices=["basic"],
            help=
            """type of joiner network, there is only basic joiner for now""")
        pred_parser.add_argument("--hid-feats",
                                 default=512,
                                 type=int,
                                 help="""hidden features of the joiner""")
        parser.add_argument("--joiner",
                            action=ActionParser(parser=pred_parser))

    @staticmethod
    def add_class_args(parser,
                       prefix=None,
                       skip=set(["in_feats", "blank_id", "vocab_size"])):

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

        RNNTransducerDecoder.add_pred_args(parser)
        RNNTransducerDecoder.add_joiner_args(parser)
        parser.add_argument(
            "--rnnt-loss",
            default="k2_pruned",
            choices=["torchaudio", "k2", "k2_pruned"],
            help="""type of rnn-t loss between torchaudio, k2 or k2_pruned.""")
        parser.add_argument(
            "--rnnt-type",
            default="regular",
            choices=["regular", "modified", "constrained"],
            help=
            """type of rnn-t loss between regular, modified or constrained.""")
        parser.add_argument(
            "--delay-penalty",
            default=0.0,
            type=float,
            help=
            """penalize symbol delay, which is used to make symbol emit earlier
            for streaming models.""")
        parser.add_argument(
            "--reduction",
            default="sum",
            choices=["sum", "mean"],
            help="""type of reduction for rnn-t loss between sum or mean""")
        parser.add_argument(
            "--prune-range",
            default=None,
            type=Optional[int],
            help="""how many symbols to keep for each frame in k2 rnn-t 
            pruned loss.""")
        parser.add_argument(
            "--lm-scale",
            default=0.25,
            type=float,
            help="""language model scale in rnn-t smoothed loss""")
        parser.add_argument(
            "--am-scale",
            default=0.0,
            type=float,
            help="""acoustic model scale in rnn-t smoothed loss""")
        parser.add_argument(
            "--simple-loss-scale",
            default=0.5,
            type=float,
            help="""weight of rnn-t simple loss when using k2 pruned loss""")
        parser.add_argument(
            "--pruned-warmup-steps",
            default=2000,
            type=int,
            help="""number of steps to warm up the k2 rnn-t pruned loss 
            from 0.1 to 1""")

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


        parser.add_argument(
            "--reduction",
            default="sum",
            choices=["sum", "mean"],
            help="""type of reduction for rnn-t loss between sum or mean""")
            
        parser.add_argument(
            "--prune-range",
            default=5,
            type=int,
            help="""how many symbols to keep for each frame in k2 rnn-t 
            pruned loss.""")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
