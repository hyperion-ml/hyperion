"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....np.preprocessing import ResamplerToTargetFreq
from ....utils.misc import PathLike

SAMPLING_RATE = 16000
DNSMOS_INPUT_LENGTH = 9.01
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hyperion" / "dnsmos"
PRIMARY_MODEL_FILENAME = "sig_bak_ovr.onnx"
PERSONALIZED_PRIMARY_MODEL_FILENAME = "pdnsmos_sig_bak_ovr.onnx"
PRIMARY_MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
PERSONALIZED_PRIMARY_MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
P808_MODEL_FILENAME = "nonpersonalized_mos_predictor.onnx"
P808_MODEL_URL = (
    "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx"
)


class DNSMOS:
    """Wrapper around the DNSMOS ONNX checkpoints (P.835 + optional P.808 regressor)."""

    def __init__(
        self,
        primary_model_path: Optional[PathLike] = None,
        p808_model_path: Optional[PathLike] = None,
        device: str = "cpu",
        is_personalized_mos: bool = False,
        cache_dir: PathLike = DEFAULT_CACHE_DIR,
        enable_p808: bool = True,
    ):
        """
        Args:
          primary_model_path: Path to sig_bak_ovr.onnx (P.835 DNSMOS model). When None the
            checkpoint is downloaded into cache_dir.
          p808_model_path: Optional path to the P.808 regressor ONNX checkpoint. When enable_p808
            is True and this is None, the checkpoint is downloaded into cache_dir.
          device: Either 'cpu' or 'cuda'. CUDA requires the corresponding runtime build.
          is_personalized_mos: Whether to apply the personalized MOS calibration curves.
          cache_dir: Directory where downloaded checkpoints are stored.
          enable_p808: Whether to run the optional P.808 regressor.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        primary_model_filename = (
            PERSONALIZED_PRIMARY_MODEL_FILENAME
            if is_personalized_mos
            else PRIMARY_MODEL_FILENAME
        )
        primary_model_url = (
            PERSONALIZED_PRIMARY_MODEL_URL if is_personalized_mos else PRIMARY_MODEL_URL
        )
        primary_description = (
            "DNSMOS primary (personalized)" if is_personalized_mos else "DNSMOS primary"
        )

        primary_model_path = self._ensure_model_file(
            primary_model_path,
            primary_model_filename,
            primary_model_url,
            primary_description,
        )

        providers = None
        if device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device.lower() != "cpu":
            raise ValueError(f"Unsupported device '{device}' for DNSMOS ONNX runtime")

        session_kwargs = {}
        if providers is not None:
            session_kwargs["providers"] = providers

        self.onnx_sess = ort.InferenceSession(str(primary_model_path), **session_kwargs)

        enable_p808 = enable_p808 or (p808_model_path is not None)
        self.p808_onnx_sess = None
        if enable_p808:
            try:
                p808_model_path = self._ensure_model_file(
                    p808_model_path,
                    P808_MODEL_FILENAME,
                    P808_MODEL_URL,
                    "DNSMOS P.808",
                )
                self.p808_onnx_sess = ort.InferenceSession(
                    str(p808_model_path), **session_kwargs
                )
            except (RuntimeError, FileNotFoundError) as exc:
                logging.warning(
                    "Could not load DNSMOS P.808 regressor (%s). "
                    "Continuing without P.808 MOS. "
                    "Provide a local p808_model_path or set enable_p808=False.",
                    exc,
                )

        self.is_personalized_mos = is_personalized_mos
        self.resampler = ResamplerToTargetFreq(float(SAMPLING_RATE))

    @staticmethod
    def audio_melspec(
        audio: np.ndarray,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = SAMPLING_RATE,
        to_db: bool = True,
    ) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    @staticmethod
    def get_polyfit_val(
        sig: float, bak: float, ovr: float, is_personalized_mos: bool
    ) -> Tuple[float, float, float]:
        if is_personalized_mos:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def _prepare_audio(
        self, audio: Union[np.ndarray, torch.Tensor], fs: float
    ) -> Tuple[np.ndarray, float]:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio, fs = self.resampler(audio, fs)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"DNSMOS expects mono audio. Received shape {audio.shape}")
        return audio, fs

    def _score_single(
        self, audio: np.ndarray, fs: float, audio_id: str
    ) -> Dict[str, float]:
        fs_int = int(fs)
        if fs_int != SAMPLING_RATE:
            logging.warning(
                "Resampling mismatch detected (fs=%s). DNSMOS expects %d Hz.",
                fs,
                SAMPLING_RATE,
            )

        actual_audio_len = len(audio)
        len_samples = int(DNSMOS_INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - DNSMOS_INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        predicted_mos_sig_seg_raw: List[float] = []
        predicted_mos_bak_seg_raw: List[float] = []
        predicted_mos_ovr_seg_raw: List[float] = []
        predicted_mos_sig_seg: List[float] = []
        predicted_mos_bak_seg: List[float] = []
        predicted_mos_ovr_seg: List[float] = []
        predicted_p808_mos: List[float] = []

        for idx in range(max(num_hops, 0)):
            start = int(idx * hop_len_samples)
            end = int((idx + DNSMOS_INPUT_LENGTH) * hop_len_samples)
            audio_seg = audio[start:end]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg, dtype=np.float32, copy=False)[
                np.newaxis, :
            ]
            oi = {"input_1": input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, self.is_personalized_mos
            )

            predicted_mos_sig_seg_raw.append(float(mos_sig_raw))
            predicted_mos_bak_seg_raw.append(float(mos_bak_raw))
            predicted_mos_ovr_seg_raw.append(float(mos_ovr_raw))
            predicted_mos_sig_seg.append(float(mos_sig))
            predicted_mos_bak_seg.append(float(mos_bak))
            predicted_mos_ovr_seg.append(float(mos_ovr))

            if self.p808_onnx_sess is not None:
                # The official implementation excludes the last frame (160 samples) before computing the mel features.
                p808_audio_seg = audio_seg[:-160] if len(audio_seg) > 160 else audio_seg
                p808_input = self.audio_melspec(audio=p808_audio_seg).astype(
                    np.float32, copy=False
                )[np.newaxis, :, :]
                p808_oi = {"input_1": p808_input}
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
                predicted_p808_mos.append(float(p808_mos))

        clip_dict: Dict[str, float] = {
            "id": audio_id,
            "p835_ovrl_raw": (
                float(np.mean(predicted_mos_ovr_seg_raw))
                if predicted_mos_ovr_seg_raw
                else np.nan
            ),
            "p835_sig_raw": (
                float(np.mean(predicted_mos_sig_seg_raw))
                if predicted_mos_sig_seg_raw
                else np.nan
            ),
            "p835_bak_raw": (
                float(np.mean(predicted_mos_bak_seg_raw))
                if predicted_mos_bak_seg_raw
                else np.nan
            ),
            "p835_ovrl": (
                float(np.mean(predicted_mos_ovr_seg))
                if predicted_mos_ovr_seg
                else np.nan
            ),
            "p835_sig": (
                float(np.mean(predicted_mos_sig_seg))
                if predicted_mos_sig_seg
                else np.nan
            ),
            "p835_bak": (
                float(np.mean(predicted_mos_bak_seg))
                if predicted_mos_bak_seg
                else np.nan
            ),
        }

        if self.p808_onnx_sess is not None:
            clip_dict["p808_mos"] = (
                float(np.mean(predicted_p808_mos)) if predicted_p808_mos else np.nan
            )

        return clip_dict

    def _ensure_model_file(
        self,
        model_path: Optional[PathLike],
        default_filename: str,
        url: str,
        description: str,
    ) -> Path:
        if model_path is not None:
            resolved = Path(model_path)
            if resolved.is_dir():
                resolved = resolved / default_filename
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(
                f"{description} checkpoint expected at {resolved} but was not found"
            )

        cache_path = self.cache_dir / default_filename
        if not cache_path.is_file():
            logging.info(
                "Downloading %s checkpoint from %s to %s",
                description,
                url,
                cache_path,
            )
            self._download_model(url, cache_path)
        return cache_path

    @staticmethod
    def _download_model(url: str, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
        try:
            with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as fh:
                shutil.copyfileobj(response, fh)
            tmp_path.replace(dst_path)
        except (urllib.error.URLError, OSError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to download DNSMOS model from {url}") from exc
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def compute_scores(
        self,
        audios: Iterable[Union[np.ndarray, torch.Tensor]],
        audio_fs: Iterable[float],
        audio_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Computes DNSMOS statistics for a collection of audios.

        Returns:
          pandas.DataFrame indexed by audio ids with the MOS fields.
        """
        audios = list(audios)
        audio_fs = list(audio_fs)
        if audio_ids is None:
            audio_ids = [str(i) for i in range(len(audios))]
        else:
            audio_ids = list(audio_ids)

        if not (len(audios) == len(audio_fs) == len(audio_ids)):
            raise ValueError(
                "Mismatch between number of audios, sample rates, and audio ids"
            )

        results: List[Dict[str, float]] = []
        for audio, fs, audio_id in zip(audios, audio_fs, audio_ids):
            audio_np, fs_out = self._prepare_audio(audio, fs)
            clip_result = self._score_single(audio_np, fs_out, audio_id)
            results.append(clip_result)

        df = pd.DataFrame(results)
        df.set_index("id", inplace=True)
        return df

    def __call__(
        self,
        audios: Iterable[Union[np.ndarray, torch.Tensor]],
        audio_fs: Iterable[float],
        audio_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        return self.compute_scores(audios, audio_fs, audio_ids)

    @staticmethod
    def add_class_args(
        parser, prefix: Optional[str] = None, skip: Optional[set] = None
    ):
        """Register DNSMOS CLI arguments."""
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "primary_model_path" not in skip:
            parser.add_argument(
                "--primary-model-path",
                type=str,
                default=None,
                help="Path to sig_bak_ovr.onnx. If omitted, it is downloaded to the cache directory.",
            )

        if "p808_model_path" not in skip:
            parser.add_argument(
                "--p808-model-path",
                type=str,
                default=None,
                help="Path to nonpersonalized_mos_predictor.onnx. If omitted, download is attempted when enabled.",
            )

        if "device" not in skip:
            parser.add_argument(
                "--device",
                type=str,
                default="cpu",
                choices=["cpu", "cuda"],
                help="Execution device for ONNX Runtime.",
            )

        if "is_personalized_mos" not in skip:
            parser.add_argument(
                "--is-personalized-mos",
                default=False,
                action=ActionYesNo,
                help="Apply the personalized MOS calibration curves.",
            )

        if "cache_dir" not in skip:
            parser.add_argument(
                "--cache-dir",
                type=str,
                default=str(DEFAULT_CACHE_DIR),
                help="Directory where DNSMOS checkpoints are cached.",
            )

        if "enable_p808" not in skip:
            parser.add_argument(
                "--enable-p808",
                default=True,
                action=ActionYesNo,
                help="Enable the P.808 regressor (requires nonpersonalized_mos_predictor.onnx).",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
