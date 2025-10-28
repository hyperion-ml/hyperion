"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import csv
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.langcodes import dialect_to_alpha3, language_to_alpha2, language_to_alpha3
from ..utils.misc import PathLike
from .data_prep import DataPrep


class CommonVoiceDataPrep(DataPrep):
    """
    Prepares Mozilla Common Voice datasets into structured tables.

    Supports single or multi-language preparation and allows selecting subsets such as
    'validated', 'train', 'test', etc. Builds `RecordingSet`, `SegmentSet`, and
    class info tables for speaker and language.

    Attributes:
        corpus_dir (PathLike): Root Common Voice directory containing language subfolders.
        language (str): Language code (e.g., 'en') or 'all' to process every language folder.
        subset (str): Which subset file to process (e.g., 'validated', 'train', 'test').
        output_dir (PathLike): Directory to save prepared outputs.
        use_kaldi_ids (bool): Whether to prepend speaker ID to each segment ID.
        target_sample_freq (Optional[int]): Optional target sample rate (Hz).
        num_threads (int): Number of threads for parallel audio processing.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        language: str,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        """
        Initializes the CommonVoice preparation logic.

        Args:
            corpus_dir (PathLike): Path to the dataset root.
            language (str): Language code or 'all'.
            subset (str): Subset file to process ('validated', 'train', 'test', etc.).
            output_dir (PathLike): Where to save processed data.
            use_kaldi_ids (bool): Whether to format IDs with speaker prefix.
            target_sample_freq (Optional[int]): If set, resample audio to this frequency.
            num_threads (int): Number of parallel threads for duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        if language != "all":
            self.language_alpha2 = language_to_alpha2(language)
        else:
            self.language_alpha2 = "all"
        self.subset = subset.lower()

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name identifier."""
        return "commonvoice"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds CLI arguments specific to Common Voice.

        Args:
            parser: Argument parser object.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--language",
            required=True,
            help="Language code (e.g., 'en') or 'all' to prepare all available languages.",
        )
        parser.add_argument(
            "--subset",
            required=True,
            choices=["validated", "test", "train", "dev", "other"],
            help="Which Common Voice TSV subset to prepare (e.g., 'validated', 'test', 'train').",
        )

    def _load_precomputed_durations(
        self, lang_dir: Path
    ) -> Optional[Tuple[pd.Series, Path]]:
        """
        Locate and load Common Voice's precomputed clip durations if available.

        Args:
            lang_dir: Directory for the specific language within the corpus.

        Returns:
            A tuple containing a pandas Series indexed by clip filename with durations
            in seconds, and the Path to the file used. Returns None if no suitable file
            is found.
        """
        # Common Voice releases provide clip durations in clip_durations.tsv.
        path = lang_dir / "clip_durations.tsv"
        if path.exists():
            durations = self._read_duration_file(path)
            if durations is not None:
                return durations, path

        return None

    @staticmethod
    def _read_duration_file(path: Path) -> Optional[pd.Series]:
        """
        Parse a duration file into a Series keyed by clip filename.

        Args:
            path: Path to the duration file.

        Returns:
            Series with durations in seconds keyed by clip filenames, or None on failure.
        """
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:  # pragma: no cover - file format errors
            logging.warning("Failed to read durations from %s: %s", path, exc)
            return None

        if df.empty:
            logging.warning("Duration file %s is empty", path)
            return None

        column_map = {col.lower(): col for col in df.columns}
        clip_col = next(
            (
                column_map[key]
                for key in column_map
                if any(token in key for token in ("clip", "path", "file", "filename"))
            ),
            None,
        )
        duration_col = next(
            (column_map[key] for key in column_map if "duration" in key),
            None,
        )

        if clip_col is None or duration_col is None:
            logging.warning(
                "Duration file %s missing required columns (clip, duration)", path
            )
            return None

        duration_df = df[[clip_col, duration_col]].copy()
        duration_df[clip_col] = duration_df[clip_col].astype(str).str.strip()
        duration_df[clip_col] = duration_df[clip_col].map(
            lambda value: Path(value).name
        )

        durations = pd.to_numeric(duration_df[duration_col], errors="coerce")
        duration_df["duration"] = durations
        duration_df = duration_df.dropna(subset=[clip_col, "duration"])
        if duration_df.empty:
            logging.warning(
                "Duration file %s does not contain valid numeric values", path
            )
            return None

        series = duration_df.set_index(clip_col)["duration"]
        series = series[~series.index.duplicated(keep="first")]
        max_value = series.max()
        if max_value is None:
            return None

        if "ms" in duration_col.lower() or max_value > 1000:
            series = series / 1000.0

        return series.astype(float)

    def prepare(self) -> None:
        """
        Executes the preparation for one or all languages.

        Loads metadata, aligns it with audio, extracts durations, and saves a HypDataset
        for each selected language.
        """
        if self.language_alpha2 == "all":
            langs = [
                d.name
                for d in self.corpus_dir.iterdir()
                if (d / f"{self.subset}.tsv").is_file()
            ]
        else:
            langs = [self.language_alpha2]

        for lang in langs:
            logging.info(f"Preparing Common Voice {lang} subset={self.subset}")
            self._prepare_language(lang)

    def _prepare_language(self, lang: str) -> None:
        """
        Prepares a single language subset.

        Args:
            lang (str): Language code to process.
        """
        lang_dir = self.corpus_dir / lang
        tsv_path = lang_dir / f"{self.subset}.tsv"
        clips_dir = (lang_dir / "clips").resolve()
        lang_alpha3 = language_to_alpha3(lang)

        assert tsv_path.exists(), f"Missing {self.subset}.tsv in {lang_dir}"
        assert clips_dir.is_dir(), f"Missing 'clips/' directory in {lang_dir}"

        df = pd.read_csv(
            tsv_path,
            sep="\t",
            engine="python",
            quoting=csv.QUOTE_NONE,
            dtype={
                "client_id": "string",
                "sentence_domain": "string",
                "gender": "string",
                "age": "string",
                "accents": "string",
                "up_votes": "Int64",
                "down_votes": "Int64",
            },
        )
        if "sentence" in df.columns:
            df = df.rename(columns={"sentence": "transcript"})
        if "transcript" in df.columns:
            df["transcript"] = (
                df["transcript"].astype("string").map(self._clean_transcript_text)
            )
        if "gender" in df.columns:
            df = df.rename(columns={"gender": "gender_extended"})
            df["gender"] = df["gender_extended"].map(
                {"male_masculine": "m", "female_feminine": "f"}
            )
        if "age" in df.columns:
            df = df.rename(columns={"age": "age_decade"})
        if "accents" in df.columns:
            accents_series = (
                df["accents"]
                .astype(str)
                .str.split(",", expand=False)
                .map(lambda values: [v.strip() for v in values if v.strip()])
            )
            df["accent"] = accents_series.map(
                lambda values: values[0] if len(values) == 1 else pd.NA
            )
        df["speaker"] = df["client_id"].apply(lambda x: f"cv-{x}")
        df["id"] = df["path"].apply(lambda x: f"cv-{Path(x).with_suffix('').name}")
        df["storage_path"] = df["path"].apply(lambda x: str(clips_dir / x))
        df["language"] = lang_alpha3
        df["source_type"] = "read_speech"
        df["clip_key"] = df["path"].apply(lambda x: Path(str(x)).name)

        if self.use_kaldi_ids:
            df["id"] = df.apply(lambda row: f"{row['speaker']}-{row['id']}", axis=1)

        logging.info(f"Creating RecordingSet for {lang_alpha3}")
        recs = RecordingSet(
            pd.DataFrame({"id": df["id"], "storage_path": df["storage_path"]})
        )

        recs.get_durations(self.num_threads, report_every=500)
        recs_indexed = recs.df.set_index("id")
        df["duration"] = df["id"].map(recs_indexed["duration"])
        df["sample_freq"] = df["id"].map(recs_indexed["sample_freq"])

        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        df.drop(columns=["clip_key"], inplace=True, errors="ignore")

        for column in [
            "accent",
            "accents",
            "age_decade",
            "gender",
            "gender_extended",
        ]:
            if column not in df.columns:
                df[column] = pd.NA

        for column in ["accents", "accent"]:
            if column in df.columns:
                df[column] = df[column].astype("string").str.lower()

        def _map_dialect(value) -> Optional[str]:
            if pd.isna(value):
                return pd.NA
            text = str(value).strip()
            if not text:
                return pd.NA
            try:
                return dialect_to_alpha3(lang_alpha3, text)
            except ValueError:
                return pd.NA

        df["dialect"] = df["accent"].map(_map_dialect)

        logging.info(f"Creating SegmentsSet for {lang_alpha3}")
        segments = SegmentSet(
            df[
                [
                    "id",
                    "speaker",
                    "transcript",
                    "duration",
                    "language",
                    "age_decade",
                    "gender",
                    "gender_extended",
                    "sentence_domain",
                    "up_votes",
                    "down_votes",
                    "dialect",
                    "accents",
                ]
            ].copy()
        )
        segments["original_bandwidth"] = (
            segments["id"].map(recs_indexed["sample_freq"]).astype(float) / 2.0
        )
        segments.sort()

        logging.info(f"Creating ClassInfo tables for {lang_alpha3}")
        speakers = ClassInfo(pd.DataFrame({"id": np.unique(df["speaker"])}))
        languages = ClassInfo(pd.DataFrame({"id": [lang_alpha3]}))
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        class_infos = {
            "speaker": speakers,
            "language": languages,
            "gender": genders,
        }
        for class_name in [
            "age_decade",
            "sentence_domain",
            "dialect",
        ]:
            values = df[class_name].dropna()
            if not values.empty:
                if class_name == "age_decade":
                    age_order = {
                        label: idx
                        for idx, label in enumerate(
                            [
                                "teens",
                                "twenties",
                                "thirties",
                                "forties",
                                "fifties",
                                "sixties",
                                "seventies",
                                "eighties",
                                "nineties",
                            ]
                        )
                    }
                    unique_values = pd.Series(values.unique()).sort_values(
                        key=lambda s: s.map(age_order).fillna(len(age_order)),
                        ignore_index=True,
                    )
                else:
                    unique_values = pd.Series(values.unique()).sort_values(
                        ignore_index=True
                    )
                class_infos[class_name] = ClassInfo(pd.DataFrame({"id": unique_values}))

        output_path = (
            self.output_dir / lang_alpha3
            if self.language_alpha2 == "all"
            else self.output_dir
        )
        output_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving dataset for {lang_alpha3} to {output_path}")
        dataset = HypDataset(
            segments=segments,
            recordings=recs,
            classes=class_infos,
        )
        dataset.save(output_path)
        dataset.describe()
        logging.info(
            "Language %s: %d segments, %d speakers",
            lang_alpha3,
            len(segments),
            len(speakers),
        )

    @staticmethod
    def _clean_transcript_text(value: Optional[str]) -> Optional[str]:
        if value is None or pd.isna(value):
            return pd.NA

        text = str(value)
        if not text:
            return pd.NA

        # Remove unbalanced leading/trailing quotes; if still unbalanced, strip all.
        if text.startswith('"') and not text.endswith('"'):
            text = text.lstrip('"')
        if text.endswith('"') and not text.startswith('"'):
            text = text.rstrip('"')
        if text.count('"') % 2 == 1:
            text = text.replace('"', "")

        return text

    def _report_segment_recording_mismatch(
        self,
        metadata_df: pd.DataFrame,
        recordings: RecordingSet,
        lang_alpha3: str,
    ) -> None:
        """
        Warn when segments reference recordings that were not materialised.
        """
        segment_ids = metadata_df["id"].astype(str)
        recording_ids = recordings.df["id"].astype(str)
        if len(recording_ids) != len(metadata_df):
            logging.warning(
                "Language %s: %d segments but %d recordings detected.",
                lang_alpha3,
                len(segment_ids),
                len(recordings),
            )

        missing_mask = ~segment_ids.isin(recording_ids)
        if not missing_mask.any():
            return

        missing_segments = metadata_df.loc[
            missing_mask,
            ["id", "speaker", "storage_path", "path"],
        ].copy()
        missing_segments["audio_exists"] = missing_segments["storage_path"].map(
            lambda value: (
                Path(value).is_file()
                if isinstance(value, (str, Path)) and value
                else False
            )
        )

        logging.warning(
            "Language %s: %d segment(s) do not have a matching recording entry. "
            "First missing IDs: %s",
            lang_alpha3,
            len(missing_segments),
            missing_segments["id"].head(10).tolist(),
        )
        logging.debug(
            "Language %s: missing segment details (up to 20 rows): %s",
            lang_alpha3,
            missing_segments.head(20).to_dict("records"),
        )
