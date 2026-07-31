"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

try:
    import k2
except ModuleNotFoundError:
    from ..utils import dummy_k2 as k2

from ...utils.misc import filter_func_args
from ...utils.text import add_sos
from ..layer_blocks import TransducerConvPredictor as ConvPredictor
from ..layer_blocks import TransducerJoiner as Joiner
from ..layer_blocks import TransducerRNNPredictor as RNNPredictor
from .net_arch import NetArch


@dataclass
class Hypothesis:
    """Beam-search hypothesis container.

    Attributes:
      ys: Token sequence including the leading blank/symbol prefix used by
        the search helpers.
      log_prob: Accumulated log probability for the hypothesis.
      pred_state: Optional cached predictor state associated with `ys`.
    """

    ys: List[int]  # predicted sequences
    log_prob: float  # log prob of ys

    # Optional predictor state.
    pred_state: Optional[Tuple[torch.Tensor, ...]] = None


class RNNTransducerDecoder(NetArch):
    """RNN-T decoder composed of predictor and joiner networks.

    Implementation based on
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/transducer.py

    Attributes:
      in_feats: Encoder output feature dimension.
      vocab_size: Number of tokens in the modeling unit, including blank.
      predictor_args: Predictor configuration dictionary.
      joiner_args: Joiner configuration dictionary.
      blank_id: ID of the blank symbol.
      rnnt_loss: RNN-T loss backend, one of `torchaudio`, `k2`, or
        `k2_pruned`.
      rnnt_type: RNN-T variation, one of `regular`, `modified`, or
        `constrained`.
      delay_penalty: Symbol delay penalty used by the k2 losses.
      reduction: Reduction mode for the selected RNN-T loss.
      prune_range: Number of symbols kept per frame for k2 pruned loss.
      lm_scale: Language-model scale in the smoothed k2 loss.
      am_scale: Acoustic-model scale in the smoothed k2 loss.
      simple_loss_scale: Weight of the simple loss term when using k2 pruned
        loss.
      pruned_warmup_steps: Number of warmup steps for k2 pruned loss scaling.
      predictor: Instantiated predictor module.
      joiner: Instantiated joiner module.
      simple_am_proj: Auxiliary acoustic projection used only for
        `k2_pruned`.
      simple_lm_proj: Auxiliary predictor projection used only for
        `k2_pruned`.
      cur_step: Warmup step counter used only for `k2_pruned`.
    """

    def __init__(
        self,
        in_feats: int,
        vocab_size: int,
        predictor: Dict[str, Any],
        joiner: Dict[str, Any],
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
    ) -> None:
        """Initialize the decoder and its submodules.

        Args:
          in_feats: Encoder output feature dimension.
          vocab_size: Number of output symbols, including blank.
          predictor: Predictor configuration dictionary.
          joiner: Joiner configuration dictionary.
          blank_id: ID of the blank symbol.
          rnnt_loss: RNN-T loss backend to use.
          rnnt_type: RNN-T variant to use with the selected backend.
          delay_penalty: Symbol delay penalty for the k2 losses.
          reduction: Reduction mode for the selected RNN-T loss.
          prune_range: Number of candidate symbols kept per frame for k2
            pruned loss.
          lm_scale: Language-model scale for the smoothed k2 loss.
          am_scale: Acoustic-model scale for the smoothed k2 loss.
          simple_loss_scale: Weight of the simple loss term during warmup.
          pruned_warmup_steps: Number of warmup steps for the k2 pruned loss.
        """
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
            self.simple_lm_proj = nn.Linear(self.predictor.out_feats, vocab_size)
            self.register_buffer("cur_step", torch.as_tensor(0, dtype=torch.int))

    def _make_predictor(self) -> None:
        """Instantiate the predictor module from the configured type."""
        pred_type = self.predictor_args["pred_type"]
        self.predictor_args["in_feats"] = self.in_feats
        self.predictor_args["vocab_size"] = self.vocab_size
        self.predictor_args["blank_id"] = self.blank_id
        if pred_type == "rnn":
            pred_args = filter_func_args(RNNPredictor.__init__, self.predictor_args)
            self.predictor = RNNPredictor(**pred_args)
        elif pred_type == "conv":
            pred_args = filter_func_args(ConvPredictor.__init__, self.predictor_args)
            self.predictor = ConvPredictor(**pred_args)
            self.predictor_args["out_feats"] = self.predictor.out_feats
        else:
            raise ValueError(f"Unknown predictor type {pred_type}")

    def _make_joiner(self) -> None:
        """Instantiate the joiner module from the configured type."""
        joiner_type = self.joiner_args["joiner_type"]

        if joiner_type == "basic":
            pred_feats = self.predictor_args["out_feats"]
            hid_feats = self.joiner_args["hid_feats"]
            self.joiner = Joiner(self.in_feats, pred_feats, hid_feats, self.vocab_size)
        else:
            raise ValueError(f"Unknown joiner type {joiner_type}")

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration for this decoder.

        Args:
          no_class_name: If `True`, omit the base `class_name` field.

        Returns:
          Dictionary containing the decoder configuration.
        """
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
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def _rnnt_loss_torchaudio(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        pred_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the torchaudio RNN-T loss.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Encoder sequence lengths.
          y: Target ragged tensor without a leading SOS symbol.
          y_lengths: Target sequence lengths.
          pred_out: Predictor activations aligned to `y`.

        Returns:
          Scalar loss tensor.
        """
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

    def _rnnt_loss_k2(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        pred_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the k2 RNN-T loss.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Encoder sequence lengths.
          y: Target ragged tensor without a leading SOS symbol.
          y_lengths: Target sequence lengths.
          pred_out: Predictor activations aligned to `y`.

        Returns:
          Scalar loss tensor.
        """
        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lengths
        boundary[:, 3] = x_lengths

        logits = self.joiner(x, pred_out)

        with torch.amp.autocast(enabled=False, device_type=x.device.type):
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

    def _rnnt_loss_k2_pruned(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        pred_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the k2 pruned RNN-T loss and its auxiliary terms.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Encoder sequence lengths.
          y: Target ragged tensor without a leading SOS symbol.
          y_lengths: Target sequence lengths.
          pred_out: Predictor activations aligned to `y`.

        Returns:
          Tuple ``(loss, loss_simple, loss_pruned)``.
        """
        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lengths
        boundary[:, 3] = x_lengths

        am_simple = self.simple_am_proj(x)
        lm_simple = self.simple_lm_proj(pred_out)
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
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

        with torch.amp.autocast(enabled=False, device_type=x.device.type):
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
            # print(simple_loss_scale, pruned_loss_scale)

        loss = simple_loss_scale * loss_simple + pruned_loss_scale * loss_pruned

        return loss, loss_simple, loss_pruned

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, y: k2.RaggedTensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute the selected RNN-T loss for a batch.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Encoder sequence lengths.
          y: Target sequences as a ragged tensor.

        Returns:
          Tuple ``(loss, loss_simple, loss_pruned)``. The auxiliary losses are
          only populated when `rnnt_loss == "k2_pruned"`.
        """
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
                x, x_lengths, y, y_lengths, pred_out
            )
        elif self.rnnt_loss == "k2":
            loss = self._rnnt_loss_k2(x, x_lengths, y, y_lengths, pred_out)
        elif self.rnnt_loss == "torchaudio":
            loss_simple = loss_pruned = None
            loss = self._rnnt_loss_torchaudio(x, x_lengths, y, y_lengths, pred_out)
        else:
            raise ValueError(f"Unknown rnnt_loss {self.rnnt_loss}")

        return loss, loss_simple, loss_pruned

    def decode(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        method: str = "time_sync_beam_search",
        beam_width: int = 5,
        max_sym_per_frame: int = 3,
        max_sym_per_utt: int = 1000,
    ) -> List[int]:
        """Decode encoder activations into a token sequence.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Optional valid-frame lengths for each utterance.
          method: Decoding strategy to use.
          beam_width: Beam size for beam-search methods.
          max_sym_per_frame: Maximum symbols per frame for greedy decoding.
          max_sym_per_utt: Maximum number of emitted symbols per utterance.

        Returns:
          Decoded token IDs for the first utterance in the batch.
        """
        if method == "time_sync_beam_search":
            return self.decode_time_sync_beam_search(
                x, x_lengths, beam_width=beam_width
            )
        elif method == "align_length_sync_beam_search":
            return self.decode_align_length_sync_beam_search(
                x, x_lengths, beam_width=beam_width, max_sym_per_utt=max_sym_per_utt
            )
        elif method == "greedy":
            return self.decode_greedy(
                x,
                x_lengths,
                max_sym_per_frame=max_sym_per_frame,
                max_sym_per_utt=max_sym_per_utt,
            )
        raise ValueError(f"Unknown decode method {method}")

    def decode_greedy(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        max_sym_per_frame: int = 3,
        max_sym_per_utt: int = 1000,
    ) -> List[int]:
        """Greedy decode a single utterance.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Optional valid-frame lengths for each utterance.
          max_sym_per_frame: Maximum symbols to emit per frame.
          max_sym_per_utt: Maximum number of emitted symbols per utterance.

        Returns:
          Decoded token IDs for the first utterance in the batch.
        """
        assert x.ndim == 3

        # support only batch_size == 1 for now
        assert x.size(0) == 1, x.size(0)
        blank_id = self.blank_id
        device = x.device
        T = int(x_lengths[0].item()) if x_lengths is not None else x.size(1)

        sos = torch.tensor([blank_id], device=device, dtype=torch.int64).reshape(1, 1)
        pred_out, state = self.predictor(sos)
        t = 0
        hyp = []

        sym_per_frame = 0
        sym_per_utt = 0

        while t < T and sym_per_utt < max_sym_per_utt:
            x_t = x[:, t : t + 1, :]
            logits = self.joiner(x_t, pred_out)  # (1, 1, 1, vocab_size)
            # logits is

            log_prob = logits.log_softmax(dim=-1)  # (1, 1, 1, vocab_size)
            # TODO: Use logits.argmax()
            y = log_prob.argmax()
            if y != blank_id:
                hyp.append(y.item())
                y = y.reshape(1, 1)
                pred_out, state = self.predictor(y, state)

                sym_per_utt += 1
                sym_per_frame += 1

            if y == blank_id or sym_per_frame >= max_sym_per_frame:
                sym_per_frame = 0
                t += 1

        return hyp

    def decode_time_sync_beam_search(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        beam_width: int = 5,
    ) -> List[int]:
        """Decode with time-synchronous beam search.

        This variant iterates encoder frames explicitly. At each frame `t`, it
        keeps expanding non-blank symbols on that same frame until enough
        blank-completed hypotheses are available to advance to frame `t + 1`.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Optional valid-frame lengths for each utterance.
          beam_width: Beam size.

        Returns:
          Decoded token IDs for the first utterance in the batch.
        """
        # Expect encoder output with shape (batch, time, feat).
        assert x.ndim == 3
        # This implementation only supports single-utterance decoding.
        assert x.size(0) == 1, x.size(0)

        # Cache a few frequently used values.
        blank_id = self.blank_id
        device = x.device
        # Use the valid number of frames when lengths are provided.
        T = int(x_lengths[0].item()) if x_lengths is not None else x.size(1)

        # `t` is the encoder time index.
        t = 0
        # `B` stores hypotheses that have consumed the current frame via blank.
        B = [Hypothesis(ys=[blank_id], log_prob=0.0, pred_state=None)]
        # Safety cap on total symbol-expansion steps.
        max_u = 20000  # terminate after this number of steps
        # `u` counts how many hypothesis expansions we have performed.
        u = 0

        # Cache predictor outputs/states by token prefix to avoid recomputation.
        cache: Dict[str, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]] = {}

        # Advance one encoder frame at a time while we still have search budget.
        while t < T and u < max_u:
            # Current encoder frame slice with shape (1, 1, feat).
            x_t = x[:, t : t + 1, :]
            # `A` holds hypotheses that may still emit non-blank symbols at frame `t`.
            A = B
            # Reset `B` for the next round of blank transitions at frame `t`.
            B = []

            # Keep expanding the best active hypothesis until `B` is full enough.
            while u < max_u:
                # Pick the most likely active hypothesis.
                y_star = max(A, key=lambda hyp: hyp.log_prob)
                # Remove it from the active set before expanding it.
                A.remove(y_star)

                # Note: y_star.ys is unhashable, i.e., cannot be used
                # as a key into a dict
                cached_key = "_".join(map(str, y_star.ys))

                # Reuse predictor state for an existing prefix when possible.
                if cached_key not in cache:
                    # Feed only the last emitted symbol for incremental prediction.
                    pred_in = torch.tensor([y_star.ys[-1]], device=device).reshape(1, 1)

                    # Run the predictor conditioned on the cached recurrent state.
                    pred_out, pred_state = self.predictor(
                        pred_in,
                        y_star.pred_state,
                    )
                    # Save the incremental predictor result for this prefix.
                    cache[cached_key] = (pred_out, pred_state)
                else:
                    # Predictor output/state already computed for this prefix.
                    pred_out, pred_state = cache[cached_key]

                # Combine encoder and predictor streams to obtain vocabulary logits.
                logits = self.joiner(x_t, pred_out)
                # Convert logits to log-probabilities over next symbols.
                log_prob = logits.log_softmax(dim=-1)
                # log_prob is (1, 1, 1, vocab_size)
                # Remove singleton dimensions so we index with token IDs directly.
                log_prob = log_prob.squeeze()
                # Now log_prob is (vocab_size,)

                # If we choose blank here, add the new hypothesis to B.
                # Otherwise, add the new hypothesis to A

                # First, choose blank
                # Score the transition that advances time without emitting a symbol.
                skip_log_prob = log_prob[blank_id]
                # Accumulate the blank-transition score.
                new_y_star_log_prob = y_star.log_prob + skip_log_prob.item()
                # print("tuAB0", t, u, len(y_star.ys), y_star.log_prob,
                #       skip_log_prob.item(), new_y_star_log_prob)
                # ys[:] returns a copy of ys
                # Blank keeps the same token history and predictor state.
                new_y_star = Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    # Caution: Use y_star.pred_state here
                    pred_state=y_star.pred_state,
                )
                # This hypothesis is ready to move to the next encoder frame.
                B.append(new_y_star)

                # Keep only the best candidate non-blank symbols from this state.
                topk_log_prob = log_prob.topk(beam_width, dim=-1)

                # Second, choose other labels
                # for i, v in enumerate(log_prob.tolist()):
                # Expanding a non-blank stays on the same encoder frame.
                for v, i in zip(*topk_log_prob):
                    # Convert scalar tensors to Python values for bookkeeping.
                    v = v.item()
                    i = i.item()
                    # Blank was already handled above.
                    if i == blank_id:
                        continue
                    # Append the emitted symbol to the token prefix.
                    new_ys = y_star.ys + [i]
                    # Accumulate the emitted-symbol score.
                    new_log_prob = y_star.log_prob + v
                    # Non-blank expansions use the updated predictor state.
                    new_hyp = Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        pred_state=pred_state,
                    )
                    # Keep the hypothesis active at the current time step.
                    A.append(new_hyp)

                # Count this hypothesis expansion step.
                u += 1
                # If no active same-frame continuations remain, the frame is done.
                if not A:
                    B = sorted(B, key=lambda hyp: hyp.log_prob, reverse=True)
                    if len(B) > beam_width:
                        B = B[:beam_width]
                    break

                # Check whether `B` already contains enough hypotheses that
                # outrank every remaining active continuation in `A`.
                # Best remaining active hypothesis, used as the pruning threshold.
                A_most_probable = max(A, key=lambda hyp: hyp.log_prob)
                # print("tuAB1", t, u, len(A), A_most_probable.log_prob, len(B))
                # Retain only blank-completed hypotheses that already beat all
                # remaining active hypotheses.
                B = sorted(
                    [hyp for hyp in B if hyp.log_prob > A_most_probable.log_prob],
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
                # Once we have enough completed hypotheses, stop expanding frame `t`.
                if len(B) >= beam_width:
                    # Enforce the requested beam size.
                    B = B[:beam_width]
                    break
            # Move to the next encoder frame.
            t += 1

        # Choose the final hypothesis by length-normalized score.
        best_hyp = max(B, key=lambda hyp: hyp.log_prob / max(1, len(hyp.ys[1:])))
        # Drop the initial blank prefix symbol before returning.
        ys = best_hyp.ys[1:]  # [1:] to remove the blank
        return ys

    def decode_align_length_sync_beam_search(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        beam_width: int = 5,
        max_sym_per_utt: int = 1000,
    ) -> List[int]:
        """Decode with alignment-length-synchronous beam search.

        This variant iterates over alignment index `i = t + u`, where `t` is
        the encoder frame index and `u` is the number of emitted symbols. In
        contrast to time-synchronous decoding, hypotheses are compared across a
        shared alignment step instead of fully exhausting one encoder frame
        before moving to the next.

        Args:
          x: Encoder activations with shape `(batch, time, feat)`.
          x_lengths: Optional valid-frame lengths for each utterance.
          beam_width: Beam size.
          max_sym_per_utt: Maximum number of emitted symbols per utterance.

        Returns:
          Decoded token IDs for the first utterance in the batch.
        """
        # Expect encoder output with shape (batch, time, feat).
        assert x.ndim == 3
        # This implementation only supports single-utterance decoding.
        assert x.size(0) == 1, x.size(0)

        # Cache frequently used values.
        blank_id = self.blank_id
        device = x.device
        # Use the valid number of frames when lengths are provided.
        T = int(x_lengths[0].item()) if x_lengths is not None else x.size(1)

        # Start the predictor with the blank token as the initial symbol.
        # t = 0
        # `B` stores the current beam of hypotheses.
        B = [Hypothesis(ys=[blank_id], log_prob=0.0, pred_state=None)]

        # Cache predictor outputs/states by token prefix to avoid recomputation.
        cache: Dict[str, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}
        # `F` stores hypotheses that have reached the last encoder frame.
        F = []
        # Iterate over alignment index `i = t + u`.
        for i in range(T + max_sym_per_utt):
            # `A` collects all candidates generated at this alignment index.
            A = []
            # Expand each hypothesis currently in the beam.
            for y_star in B:
                # Number of emitted symbols for this hypothesis, excluding prefix blank.
                u = len(y_star.ys) - 1
                # Recover the encoder frame from the alignment index.
                t = i - u
                # Skip alignments that fall outside the valid encoder frames.
                if t < 0 or t >= T:
                    continue

                # Current encoder frame slice with shape (1, 1, feat).
                x_t = x[:, t : t + 1, :]
                # Note: y_star.ys is unhashable, i.e., cannot be used
                # as a key into a dict
                cached_key = "_".join(map(str, y_star.ys))

                # Reuse predictor state for an existing prefix when possible.
                if cached_key not in cache:
                    # Feed only the last emitted symbol for incremental prediction.
                    pred_in = torch.tensor([y_star.ys[-1]], device=device).reshape(1, 1)

                    # Run the predictor conditioned on the cached recurrent state.
                    pred_out, pred_state = self.predictor(
                        pred_in,
                        y_star.pred_state,
                    )
                    # Save the incremental predictor result for this prefix.
                    cache[cached_key] = (pred_out, pred_state)
                else:
                    # Predictor output/state already computed for this prefix.
                    pred_out, pred_state = cache[cached_key]

                # Combine encoder and predictor streams to obtain vocabulary logits.
                logits = self.joiner(x_t, pred_out)
                # Convert logits to log-probabilities over next symbols.
                log_prob = logits.log_softmax(dim=-1)  # (1, 1, 1, vocab_size)
                # Remove singleton dimensions so we index with token IDs directly.
                log_prob = log_prob.squeeze()  # (vocab_size,)

                # First, choose blank
                # Score the transition that advances time without emitting a symbol.
                skip_log_prob = log_prob[blank_id]
                # Accumulate the blank-transition score.
                new_y_star_log_prob = y_star.log_prob + skip_log_prob.item()
                # print("tuAB0", t, u, len(y_star.ys), y_star.log_prob,
                #       skip_log_prob.item(), new_y_star_log_prob)
                # ys[:] returns a copy of ys
                # Blank keeps the same token history and predictor state.
                new_y_star = Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    # Caution: Use y_star.pred_state here
                    pred_state=y_star.pred_state,
                )
                # Add the blank-expanded hypothesis to the candidate pool.
                A.append(new_y_star)
                # Remember hypotheses that have consumed the final encoder frame.
                if t == T - 1:
                    F.append(new_y_star)

                # Keep only the best candidate non-blank symbols from this state.
                topk_log_prob = log_prob.topk(beam_width, dim=-1)

                # Second, choose other labels
                # Expanding a non-blank increases label length at the same alignment.
                for v, i in zip(*topk_log_prob):
                    # Convert scalar tensors to Python values for bookkeeping.
                    v = v.item()
                    i = i.item()
                    # Blank was already handled above.
                    if i == blank_id:
                        continue
                    # Append the emitted symbol to the token prefix.
                    new_ys = y_star.ys + [i]
                    # Accumulate the emitted-symbol score.
                    new_log_prob = y_star.log_prob + v
                    # Non-blank expansions use the updated predictor state.
                    new_hyp = Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        pred_state=pred_state,
                    )
                    # Add the emitted-symbol hypothesis to the candidate pool.
                    A.append(new_hyp)

            # Sort all candidates by descending score before deduplication.
            B0 = sorted(
                [hyp for hyp in A],
                key=lambda hyp: hyp.log_prob,
                reverse=True,
            )
            # Rebuild the beam from the sorted candidates.
            B = []
            # Deduplicate equivalent token prefixes in the beam.
            B_ys = set()
            for hyp in B0:
                # Convert the token list to a hashable key.
                hyp_ys = tuple(hyp.ys)  # to make ys hashable
                # Keep only the first occurrence of each prefix.
                if hyp_ys not in B_ys:
                    B.append(hyp)
                    B_ys.add(hyp_ys)

            # Stop once the deduplicated beam reaches the requested width.
            if len(B) >= beam_width:
                # Enforce the requested beam size.
                B = B[:beam_width]

        # Choose the final hypothesis by length-normalized score.
        best_hyp = max(F, key=lambda hyp: hyp.log_prob / max(1, len(hyp.ys[1:])))
        # Drop the initial blank prefix symbol before returning.
        ys = best_hyp.ys[1:]  # [1:] to remove the blank
        return ys

    def change_config(
        self,
        override_dropouts: bool = False,
        embed_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
    ) -> None:
        """Update predictor dropout settings.

        Args:
          override_dropouts: If `True`, apply the supplied dropout values.
          embed_dropout_rate: New embedding dropout probability.
          rnn_dropout_rate: New recurrent dropout probability.
        """
        logging.info("changing decoder config")
        self.predictor.change_config(
            override_dropouts, embed_dropout_rate, rnn_dropout_rate
        )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments for the decoder constructor.

        Args:
          **kwargs: Candidate keyword arguments.

        Returns:
          Subset of arguments accepted by :meth:`__init__`.
        """
        args = filter_func_args(RNNTransducerDecoder.__init__, kwargs)
        return args

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments for fine-tuning configuration.

        Args:
          **kwargs: Candidate keyword arguments.

        Returns:
          Subset of arguments accepted by :meth:`change_config`.
        """
        args = filter_func_args(RNNTransducerDecoder.change_config, kwargs)
        return args

    @staticmethod
    def add_pred_args(parser: ArgumentParser) -> None:
        """Add predictor configuration arguments to a parser.

        Args:
          parser: Parser to extend.
        """
        pred_parser = ArgumentParser(prog="")
        pred_parser.add_argument(
            "--pred-type",
            default="rnn",
            choices=["rnn", "conv"],
            help="""type of predictor between RNN and Convolutional [rnn, conv]""",
        )
        pred_parser.add_argument(
            "--embed-dim", default=1024, type=int, help=("token embedding dimension")
        )
        pred_parser.add_argument(
            "--embed-dropout-rate",
            default=0.0,
            type=float,
            help=("dropout prob for predictor input embeddings"),
        )
        pred_parser.add_argument(
            "--rnn-dropout-rate",
            default=0.0,
            type=float,
            help="""dropout prob for decoder RNN """,
        )
        pred_parser.add_argument(
            "--rnn-type",
            default="lstm",
            choices=["lstm", "gru"],
            help="""type of recurrent network for thep predictor in [lstm, gru]""",
        )

        pred_parser.add_argument(
            "--num-layers",
            default=2,
            type=int,
            help="""number of layers of the predictor """,
        )

        pred_parser.add_argument(
            "--hid-feats",
            default=512,
            type=int,
            help="""hidden features of the predictor""",
        )
        pred_parser.add_argument(
            "--out-feats",
            default=512,
            type=int,
            help="""output features of the predictor""",
        )
        pred_parser.add_argument(
            "--context-size",
            default=2,
            type=int,
            help="""context length of the convolutional 
                                 predictor, 1->bigram, 2-> trigram,...""",
        )

        parser.add_argument("--predictor", action=ActionParser(parser=pred_parser))

    @staticmethod
    def add_joiner_args(parser: ArgumentParser) -> None:
        """Add joiner configuration arguments to a parser.

        Args:
          parser: Parser to extend.
        """
        pred_parser = ArgumentParser(prog="")
        pred_parser.add_argument(
            "--joiner-type",
            default="basic",
            choices=["basic"],
            help="""type of joiner network, there is only basic joiner for now""",
        )
        pred_parser.add_argument(
            "--hid-feats",
            default=512,
            type=int,
            help="""hidden features of the joiner""",
        )
        parser.add_argument("--joiner", action=ActionParser(parser=pred_parser))

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Add decoder construction arguments to a parser.

        Args:
          parser: Parser to extend.
          prefix: Optional prefix for nested argument groups.
          skip: Optional set of argument names to omit.
        """
        if skip is None:
            skip = {"in_feats", "blank_id", "vocab_size"}
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats", type=int, required=True, help=("input feature dimension")
            )
        if "blank_id" not in skip:
            parser.add_argument(
                "--blank-id",
                type=int,
                default=0,
                help=("blank id from tokenizer model"),
            )
        if "vocab_size" not in skip:
            parser.add_argument(
                "--vocab-size",
                type=int,
                required=True,
                help=("output prediction dimension"),
            )

        RNNTransducerDecoder.add_pred_args(parser)
        RNNTransducerDecoder.add_joiner_args(parser)
        parser.add_argument(
            "--rnnt-loss",
            default="k2_pruned",
            choices=["torchaudio", "k2", "k2_pruned"],
            help="""type of rnn-t loss between torchaudio, k2 or k2_pruned.""",
        )
        parser.add_argument(
            "--rnnt-type",
            default="regular",
            choices=["regular", "modified", "constrained"],
            help="""type of rnn-t loss between regular, modified or constrained.""",
        )
        parser.add_argument(
            "--delay-penalty",
            default=0.0,
            type=float,
            help="""penalize symbol delay, which is used to make symbol emit earlier
            for streaming models.""",
        )
        parser.add_argument(
            "--reduction",
            default="sum",
            choices=["sum", "mean"],
            help="""type of reduction for rnn-t loss between sum or mean""",
        )
        parser.add_argument(
            "--prune-range",
            default=5,
            type=int,
            help="""how many symbols to keep for each frame in k2 rnn-t 
            pruned loss.""",
        )
        parser.add_argument(
            "--lm-scale",
            default=0.25,
            type=float,
            help="""language model scale in rnn-t smoothed loss""",
        )
        parser.add_argument(
            "--am-scale",
            default=0.0,
            type=float,
            help="""acoustic model scale in rnn-t smoothed loss""",
        )
        parser.add_argument(
            "--simple-loss-scale",
            default=0.5,
            type=float,
            help="""weight of rnn-t simple loss when using k2 pruned loss""",
        )
        parser.add_argument(
            "--pruned-warmup-steps",
            default=2000,
            type=int,
            help="""number of steps to warm up the k2 rnn-t pruned loss 
            from 0.1 to 1""",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Add fine-tuning arguments to a parser.

        Args:
          parser: Parser to extend.
          prefix: Optional prefix for nested argument groups.
          skip: Currently unused compatibility parameter.
        """
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--override-dropouts",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to use the dropout probabilities passed in the "
                "arguments instead of the defaults in the pretrained model."
            ),
        )
        parser.add_argument(
            "--embed-dropout-rate",
            default=0.0,
            type=float,
            help=("dropout prob for decoder input embeddings"),
        )
        parser.add_argument(
            "--rnn-dropout-rate",
            default=0.0,
            type=float,
            help=("dropout prob for decoder RNN "),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
