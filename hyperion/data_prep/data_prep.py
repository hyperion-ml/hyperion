"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ActionYesNo
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ..utils import PathLike


class DataPrep:
    """Base class for data preparation classes.

    Attributes:
      corpus_dir: input data directory
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
      num_threads: number of parallel threads
    """

    registry = {}

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.use_kaldi_ids = use_kaldi_ids
        self.target_sample_freq = target_sample_freq
        self.num_threads = num_threads

        self.output_dir.mkdir(exist_ok=True, parents=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.dataset_name()] = cls

    @staticmethod
    def dataset_name():
        raise NotImplementedError()

    @staticmethod
    def _get_recording_duration(scp, i, n):
        from ..io import SequentialAudioReader as AR

        durations = []
        fss = []
        with AR(scp, part_idx=i, num_parts=n) as reader:
            for data in reader:
                key, x, fs = data
                duration = x.shape[0] / fs
                fss.append(fs)
                durations.append(duration)

        return fss, durations

    def get_recording_duration(self, recording_set):

        from ..utils import SCPList
        import itertools

        scp = SCPList(recording_set["id"].values, recording_set["storage_path"].values)
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            for i in range(self.num_threads):
                future = pool.submit(
                    DataPrep._get_recording_duration, scp, i, self.num_threads
                )
                futures.append(future)

        res = [f.result() for f in tqdm(futures)]
        fss = list(itertools.chain(*[r[0] for r in res]))
        durations = list(itertools.chain(*[r[0] for r in res]))

        recording_set["duration"] = durations
        recording_set["sample_freq"] = fss

    @staticmethod
    def add_class_args(parser):
        parser.add_argument(
            "--corpus-dir", required=True, help="""input data directory""",
        )
        parser.add_argument(
            "--output-dir", required=True, help="""output data directory""",
        )
        parser.add_argument(
            "--use-kaldi-ids",
            default=False,
            action=ActionYesNo,
            help="""put speaker-id in front of segment id like kaldi""",
        )
        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help="""target sampling frequency to convert the audios to""",
        )

        parser.add_argument(
            "--num-threads",
            default=10,
            type=int,
            help="""number of parallel threads""",
        )
