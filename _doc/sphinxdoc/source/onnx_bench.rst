
.. _l-onnx-availability:

Availability and Benchmarks
===========================

:epkg:`sklearn-onnx` converts many :epkg:`scikit-learn`
models into :epkg:`ONNX`. Every of them is tested against
a couple of runtimes. The following pages shows
which models are correctly converted and compares
the predictions obtained by every runtime. It also
displays some benchmark. The benchmark evaluates
every model on a dataset inspired from the :epkg:`Iris`
dataset, so with four features, and different number of
observations *N= 1, 10, 100, 1000, 100.00, 100.000*.
The measures for high values of *N* may be missing
because  the first one took too long.

.. toctree::
    :maxdepth: 1

    skl_converters/bench_python
    skl_converters/bench_onnxrt1
    skl_converters/bench_onnxrt2

All results were obtained using out the following versions
of modules below:

.. runpython::
    :showcode:
    :rst:

    from mlprodict.onnxrt.validate_helper import modules_list
    from pyquickhelper.pandashelper import df2rst
    from pandas import DataFrame
    print(df2rst(DataFrame(modules_list())))
