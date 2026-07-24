Input and Output API
====================

``hyperion.io`` reads and writes waveform audio, keyed feature matrices, and
voice-activity metadata. It is the boundary between files on disk and the
NumPy/PyTorch workflows documented elsewhere.

CSV-indexed archives are the maintained interchange format. Use a paired Ark
or HDF5 archive plus CSV index, for example
``ark,csv:embeddings.ark,embeddings.csv``. Kaldi ``.scp`` inputs remain
available only through legacy compatibility paths; new package workflows use
CSV indexes.

Core contracts
--------------

Feature readers and writers use string keys. Writers associate each key with
one matrix/vector and optional metadata; readers return data in key order.
Keep keys identical across embedding archives, manifests, enrollment maps, and
trial tables. Missing or reordered keys are alignment errors.

Use the factories rather than selecting concrete Ark/HDF5 implementations in
application code. They parse the archive/index specifier and return the
appropriate implementation.

.. autoclass:: hyperion.io.DataWriterFactory
   :no-index:
   :members: create, filter_args

.. autoclass:: hyperion.io.SequentialDataReaderFactory
   :no-index:
   :members: create, filter_args

.. autoclass:: hyperion.io.RandomAccessDataReaderFactory
   :no-index:
   :members: create, filter_args

Archive specifications
----------------------

Common write specifications:

.. code-block:: text

   ark,csv:exp/xvectors.ark,exp/xvectors.csv
   h5,csv:exp/xvectors.h5,exp/xvectors.csv

Common read specifications:

.. code-block:: text

   csv:exp/xvectors.csv
   ark:exp/xvectors.ark
   h5:exp/xvectors.h5

The CSV index records keys and archive locations and can hold declared metadata
columns. It is preferred because it is inspectable, extensible, and consistent
with the manifest/table layer.

Feature API
-----------

.. autoclass:: hyperion.io.data_reader.DataReader
   :no-index:
   :members: close

.. autoclass:: hyperion.io.data_writer.DataWriter
   :no-index:
   :members: standardize_write_args

.. autoclass:: hyperion.io.ArkDataWriter
   :no-index:

.. autoclass:: hyperion.io.H5DataWriter
   :no-index:

For random lookup of embeddings by segment id, use
``RandomAccessDataReaderFactory``. For sequential bulk processing, use
``SequentialDataReaderFactory``. The scoring commands use random access
because enrollment and trial tables select vectors by id.

Audio API
---------

.. autoclass:: hyperion.io.AudioReader
   :no-index:

.. autoclass:: hyperion.io.SequentialAudioReader
   :no-index:

.. autoclass:: hyperion.io.RandomAccessAudioReader
   :no-index:

.. autoclass:: hyperion.io.AudioWriter
   :no-index:

Audio readers can consume a ``HyperDataset`` manifest or recording/segment
inputs. Waveform extractors resample to the loaded model's expected sample
frequency; see :doc:`how-to/train-waveform-xvector`.

VAD API
-------

.. autoclass:: hyperion.io.VADReaderFactory
   :no-index:
   :members: create, filter_args

.. autoclass:: hyperion.io.BinVADReader
   :no-index:

.. autoclass:: hyperion.io.TableVADReader
   :no-index:

Binary VAD needs its frame length, frame shift, and ``snip_edges`` convention
to be interpreted correctly. Table VAD represents time marks. See
:doc:`how-to/prepare-data-and-vad` for conversion and validation guidance.

Compatibility interfaces
------------------------

``HypDataReader`` and ``HypDataWriter`` remain available for legacy callers,
but new code should use the factory-based data reader/writer interfaces above.

.. autoclass:: hyperion.io.HypDataReader
   :no-index:

.. autoclass:: hyperion.io.HypDataWriter
   :no-index:

See also
--------

* :doc:`how-to/extract-score-xvectors`
* :doc:`utils`
* :doc:`data-model`
