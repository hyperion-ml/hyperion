"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import typing

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Pattern, Union


class ParallelFileFinder:
    """Find files matching a regex under first-level subdirectories in parallel.

    Attributes:
        root: Root directory whose direct child directories are searched.
        pattern: Compiled regex used to match file names.
        num_threads: Number of worker threads used during traversal.

    Example:
        >>> finder = ParallelFileFinder("/data", r"\\.wav$", num_threads=8)
        >>> wav_files = finder()
        >>> print(len(wav_files))
    """

    def __init__(self, root: str, pattern: Union[str, Pattern], num_threads: int = 16) -> None:
        """Initialize the parallel file finder.

        Args:
            root: Root directory whose direct child directories are searched.
            pattern: Regular expression (string or compiled pattern) matched
                against file names.
            num_threads: Number of worker threads used to traverse directories.
        """
        self.root = root
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.num_threads = num_threads

    def _find_files_in_dir(self, path: str) -> List[str]:
        """Recursively collect matching file paths under one directory.

        Args:
            path: Directory path to traverse.

        Returns:
            A list of matching file paths as strings.
        """
        matched = []
        for dirpath, _, filenames in os.walk(path, followlinks=False):
            for fname in filenames:
                if self.pattern.search(fname):
                    matched.append(os.path.join(dirpath, fname))
        return matched

    def __call__(self) -> List[Path]:
        """Run the search and return all matching files.

        Returns:
            A flat list of matching paths as ``pathlib.Path`` objects.
        """
        root_dirs = [
            entry.path
            for entry in os.scandir(self.root)
            if entry.is_dir(follow_symlinks=True)
        ]

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            results = pool.map(self._find_files_in_dir, root_dirs)

        return [Path(f) for sublist in results for f in sublist]
