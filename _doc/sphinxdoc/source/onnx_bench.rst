
.. _l-onnx-availability:

Availability and Benchmarks
===========================

:epkg:`sklearn-onnx` converts many :epkg:`scikit-learn`
models into :epkg:`ONNX`. Every of them is tested against
a couple of runtimes. The following pages shows
which models are correctly converted and compares
the predictions obtained by every runtime. It also
displays some benchmark.

.. toctree::
    :maxdepth: 1

    skl_converters/bench_python
    skl_converters/bench_onnxrt
    skl_converters/bench_onnxrt_whole

All results were obtained using out the following versions
of modules below:

.. runpython::
    :showcode:
    :rst:

    from mlprodict.onnxrt.validate_helper import modules_list
    from pyquickhelper.pandashelper import df2rst
    from pandas import DataFrame
    print(df2rst(DataFrame(modules_list())))
