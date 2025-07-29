"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Pattern, Union


class ParallelFileFinder:
    def __init__(self, root: str, pattern: Union[str, Pattern], num_threads: int = 16):
        self.root = root
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.num_threads = num_threads

    def _find_files_in_dir(self, path: str) -> List[str]:
        matched = []
        for dirpath, _, filenames in os.walk(path, followlinks=False):
            for fname in filenames:
                if self.pattern.search(fname):
                    matched.append(os.path.join(dirpath, fname))
        return matched

    def __call__(self) -> List[str]:
        root_dirs = [
            entry.path
            for entry in os.scandir(self.root)
            if entry.is_dir(follow_symlinks=True)
        ]

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            results = pool.map(self._find_files_in_dir, root_dirs)

        return [Path(f) for sublist in results for f in sublist]
