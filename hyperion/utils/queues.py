"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import warnings
import copy
import time
import numpy as np
import multiprocessing
import threading
import six
from abc import abstractmethod


try:
    import queue
except ImportError:
    import Queue as queue


class SequenceQueue(object):
    """Base class to enqueue inputs.

    The task of an Queue is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    # Examples

    ```python
    enqueuer = SequenceQueue(...)
    enqueuer.start()
    datas = enqueuer.get()
    for data in datas:
        # Use the inputs; training, evaluating, predicting.
        # ... stop sometime.
    enqueuer.close()
    ```

    The `enqueuer.get()` should be an infinite stream of datas.

    """

    @abstractmethod
    def is_running(self):
        raise NotImplemented

    @abstractmethod
    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`).
        """
        raise NotImplemented

    @abstractmethod
    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        raise NotImplemented

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplemented


class OrderedQueue(SequenceQueue):
    """Builds a Queue from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        scheduling: Sequential querying of datas if 'sequential', random otherwise.
    """

    def __init__(self, sequence, use_multiprocessing=False, scheduling="sequential"):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing
        self.scheduling = scheduling
        self.workers = 0
        self.executor = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor = multiprocessing.Pool(workers)
        else:
            self.executor = ThreadPool(workers)
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _run(self):
        """Function to submit request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        while True:
            if self.scheduling is not "sequential":
                random.shuffle(sequence)
            for i in sequence:
                if self.stop_signal.is_set():
                    return
                self.queue.put(
                    self.executor.apply_async(get_index, (self.sequence, i)), block=True
                )

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            raise StopIteration(e)

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.executor.close()
        self.executor.join()
        self.run_thread.join(timeout)


class GeneratorQueue(SequenceQueue):
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(
        self, generator, use_multiprocessing=False, wait_time=0.05, random_seed=None
    ):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
