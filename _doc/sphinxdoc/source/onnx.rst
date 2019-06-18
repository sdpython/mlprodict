
.. _l-onnx-pyrun:

ONNX
====

*mlprodict* implements two runtimes.
The first uses :epkg:`numpy` and implements
mathematical functions defined by :epkg:`ONNX`.
The second one leverages :epkg:`onnxruntime` to
compute the output of every node using
:epkg:`onnxruntime` but :epkg:`python` stills handles the graph
logic. The third one uses :epkg:`onnxruntime` to compute
everything.

.. toctree::
    :maxdepth: 1

    onnx_python
    onnx_onnxrt
    onnx_onnxrt_whole
    skl_converters/index

All results were obtained using out the following versions
of module below:

.. runpython::
    :showcode:

    import numpy
    import scipy
    import pandas
    import onnx
    import onnxruntime
    import sklearn
    import onnxconverter_common
    import skl2onnx
    import mlprodict
    for mod in [numpy, scipy, pandas, onnx, onnxruntime, sklearn,
                onnxconverter_common, skl2onnx, mlprodict]:
        print(mod.__name__, mod.__version__)
