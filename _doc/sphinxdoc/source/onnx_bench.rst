
.. _l-onnx-availability:

.. _l-model-problem-list:

ONNX Converters Coverage and Benchmarks
=======================================

:epkg:`sklearn-onnx` converts many :epkg:`scikit-learn`
models into :epkg:`ONNX`. Every of them is tested against
a couple of runtimes. The following pages shows
which models are correctly converted and compares
the predictions obtained by every runtime
(see :ref:`l-onnx-runtimes`). It also
displays some figures on how the runtime behave
compare to :epkg:`scikit-learn` in term of speed processing.
The benchmark evaluates every model on a dataset
inspired from the :epkg:`Iris` dataset,
so with four features, and different number of
observations *N= 1, 10, 100, 1000, 100.00, 100.000*.
The measures for high values of *N* may be missing
because the first one took too long.

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

Every model is tested through a defined list of standard
problems created from the :epkg:`Iris` dataset. Function
:func:`find_suitable_problem
<mlprodict.onnxrt.validate_problems.find_suitable_problem>`
describes the list of considered problems.

.. runpython::
    :showcode:
    :rst:

    from mlprodict.onnxrt.validate import sklearn_operators, find_suitable_problem
    from pyquickhelper.pandashelper import df2rst
    from pandas import DataFrame
    res = sklearn_operators(extended=True)
    rows = []
    for model in res:
        name = model['name']
        row = dict(name=name)
        try:
            prob = find_suitable_problem(model['cl'])
            for p in prob:
                row[p] = 'X'
        except RuntimeError:
            pass
        rows.append(row)
    df = DataFrame(rows).set_index('name')
    df = df.sort_index()
    cols = list(sorted(df.columns))
    df = df[cols]
    print(df2rst(df, index=True))
