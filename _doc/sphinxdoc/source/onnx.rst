
.. _l-onnx-pyrun:

ONNX, Runtime, Backends
=======================

*mlprodict* implements two runtimes.
The first uses :epkg:`numpy` and implements
mathematical functions defined by :epkg:`ONNX`.
The second one leverages :epkg:`onnxruntime` to
compute the output of every node using
:epkg:`onnxruntime` but :epkg:`python` stills handles the graph
logic. A last one just wraps :epkg:`onnxruntime` to compute
predictions, it handles the graph and operators runtimes.

.. toctree::
    :maxdepth: 1

    onnx_runtime
    backends/index

All results were obtained using out the following versions
of modules below:

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :rst:

    from mlprodict.onnxrt.validate.validate_helper import modules_list
    from pyquickhelper.pandashelper import df2rst
    from pandas import DataFrame
    print(df2rst(DataFrame(modules_list())))
