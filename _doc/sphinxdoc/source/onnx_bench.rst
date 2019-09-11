
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
observations *N= 1, 10, 100, 1.000, 10.000, 100.000*.
The measures for high values of *N* may be missing
because the first one took too long.

.. contents::
    :local:

Benchmarks
++++++++++

.. toctree::
    :maxdepth: 1

    skl_converters/bench_python
    skl_converters/bench_onnxrt1
    skl_converters/bench_onnxrt2

Versions
++++++++

All results were obtained using out the following versions
of modules below:

.. runpython::
    :showcode:
    :rst:

    from mlprodict.onnxrt.validate.validate_helper import modules_list
    from pyquickhelper.pandashelper import df2rst
    from pandas import DataFrame
    print(df2rst(DataFrame(modules_list())))

On:

.. runpython
    :showcode:

    import datetime
    print(datetime.datetime.now())

Supported models
++++++++++++++++

Every model is tested through a defined list of standard
problems created from the :epkg:`Iris` dataset. Function
:func:`find_suitable_problem
<mlprodict.onnxrt.validate.validate_problems.find_suitable_problem>`
describes the list of considered problems.

.. runpython::
    :showcode:
    :rst:

    from mlprodict.onnxrt.validate.validate import sklearn_operators, find_suitable_problem
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

Summary graph
+++++++++++++

The following graph summarizes the performance for every
supported models and compares *python runtime* and *onnxruntime*
to *scikit-learn* in the same condition. It displays a ratio *r*.
Above 1, it is *r* times slower than *scikit-learn*. Below 1,
it is *1/r* faster than *scikit-learn*.

.. plot::

    import pandas
    import matplotlib.pyplot as plt
    import numpy

    df1 = pandas.read_excel("bench_sum_python.xlsx")
    df2 = pandas.read_excel("bench_sum_onnxruntime1.xlsx")
    if 'n_features' not in df1.columns:
        df1["n_features"] = 4
    if 'n_features' not in df2.columns:
        df2["n_features"] = 4
    fmt = "{} [{}-{}] D{}"
    df1["label"] = df1.apply(lambda row: fmt.format(
                             row["name"], row["problem"], row["scenario"], row["n_features"]), axis=1)
    df2["label"] = df2.apply(lambda row: fmt.format(
                             row["name"], row["problem"], row["scenario"], row["n_features"]), axis=1)
    indices = ['label']
    values = ['RT/SKL-N=1', 'N=10', 'N=100', 'N=1000', 'N=10000', 'N=100000']
    df1 = df1[indices + values]
    df2 = df2[indices + values]
    df = df1.merge(df2, on="label", suffixes=("__pyrt", "__ort"))

    na = df["RT/SKL-N=1__pyrt"].isnull() & df["RT/SKL-N=1__ort"].isnull()
    dfp = df[~na].sort_values("label", ascending=False)

    total = dfp.shape[0] * 0.45
    fig, ax = plt.subplots(1, (dfp.shape[1]-1) // 2, figsize=(14,total), sharex=False, sharey=True)
    x = numpy.arange(dfp.shape[0])
    height = total / dfp.shape[0] * 0.65
    for c in df.columns[1:]:
        place, runtime = c.split('__')
        dec = {'pyrt': 1, 'ort': -1}
        index = values.index(place)
        yl = dfp.loc[:, c]
        xl = x + dec[runtime] * height / 2
        ax[index].barh(xl, yl, label=runtime, height=height)
        ax[index].set_title(place)
    for i in range(len(ax)):
        ax[i].plot([1, 1], [min(x), max(x)], 'g-')
        ax[i].plot([2, 2], [min(x), max(x)], 'r--')
        ax[i].legend()
        ax[i].set_xscale('log')

    ax[0].set_yticks(x)
    ax[0].set_yticklabels(dfp['label'])
    fig.subplots_adjust(left=0.25)

    plt.show()
