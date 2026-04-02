"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import threading
from typing import Callable, Iterator, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class ThreadSafeIter(Iterator[T]):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it: Iterator[T]) -> None:
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self) -> "ThreadSafeIter[T]":
        return self

    def __next__(self) -> T:
        with self.lock:
            return self.it.__next__()

    def next(self) -> T:
        with self.lock:
            return self.it.next()


def threadsafe_generator(
    f: Callable[P, Iterator[T]],
) -> Callable[P, ThreadSafeIter[T]]:
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def generator(*args: P.args, **kwargs: P.kwargs) -> ThreadSafeIter[T]:
        return ThreadSafeIter(f(*args, **kwargs))

    return generator
