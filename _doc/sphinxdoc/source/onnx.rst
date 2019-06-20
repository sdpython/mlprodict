
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
    onnx_bench

All results were obtained using out the following versions
of modules below:

.. runpython::
    :showcode:
    :rst:

    from mlprodict.onnxrt.validate_helper import modules_list
    from pyquickhelper.pandashelper import df2rst
    from pandas import DataFrame
    print(df2rst(DataFrame(modules_list())))
