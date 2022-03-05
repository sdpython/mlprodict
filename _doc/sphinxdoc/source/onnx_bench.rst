
.. _l-onnx-availability:

.. _l-model-problem-list:

scikit-learn Converters and Benchmarks
======================================

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

Another benchmark based on :epkg:`asv` is available and shows
similar results but also measure the memory peaks :
`ASV Benchmark <http://www.xavierdupre.fr/app/mlprodict_bench/helpsphinx/index.html>`_.

Visual Representations
++++++++++++++++++++++

:epkg:`sklearn-onnx` converts many :epkg:`scikit-learn` models
to :epkg:`ONNX`, it rewrites the prediction
function using :epkg:`ONNX Operators` and :epkg:`ONNX ML Operators`.
The current package *mlprodict* implements a
:ref:`l-onnx-runtime-operators`.

.. toctree::
    :maxdepth: 2

    onnx_conv
    skl_converters/index

Benchmarks
++++++++++

.. toctree::
    :maxdepth: 2

    skl_converters/bench_python
    skl_converters/bench_onnxrt1
    skl_converters/bench_onnxrt2

Versions
++++++++

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

On:

.. runpython
    :showcode:

    import datetime
    print(datetime.datetime.now())

:epkg:`onnxruntime` is compiled with the following options:

.. runpython
    :showcode:

    import onnxruntime
    print(onnxruntime.get_device())

Supported models
++++++++++++++++

Every model is tested through a defined list of standard
problems created from the :epkg:`Iris` dataset. Function
:func:`find_suitable_problem
<mlprodict.onnxrt.validate.validate_problems.find_suitable_problem>`
describes the list of considered problems.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
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
            if prob is None:
                continue
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
    from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
    from mlprodict.plotting.plotting_validate_graph import _model_name

    df1 = pandas.read_excel("bench_sum_python_compiled.xlsx")
    df2 = pandas.read_excel("bench_sum_onnxruntime1.xlsx")

    if 'n_features' not in df1.columns:
        df1["n_features"] = 4
    if 'n_features' not in df2.columns:
        df2["n_features"] = 4
    df1['optim'] = df1['optim'].fillna('')
    df2['optim'] = df2['optim'].fillna('')

    last_opset = max(int(_[5:]) for _ in list(df1.columns) if _.startswith("opset"))
    opset_col = 'opset%d' % last_opset

    df1['opset'] = df1[opset_col].fillna('')
    df2['opset'] = df2[opset_col].fillna('')

    df1['opset'] = df1['opset'].apply(lambda x: str(last_opset) if "OK %d" % last_opset in x else "")
    df2['opset'] = df2['opset'].apply(lambda x: str(last_opset) if "OK %d" % last_opset in x else "")
    sops = str(get_opset_number_from_onnx())
    oksops = "OK " + str(get_opset_number_from_onnx())
    df1['opset'] = df1['opset'].apply(lambda x: sops if oksops in x else "")
    df2['opset'] = df2['opset'].apply(lambda x: sops if oksops in x else "")

    fmt = "{} [{}-{}|{}] D{}-o{}"
    df1["label"] = df1.apply(
        lambda row: fmt.format(
            row["name"], row["problem"], row["scenario"], row["optim"],
            row["n_features"], row["opset"]).replace("-default|", "-*]"), axis=1)
    df2["label"] = df2.apply(
        lambda row: fmt.format(
            row["name"], row["problem"], row["scenario"], row["optim"],
            row["n_features"], row["opset"]).replace("-default|", "-*]"), axis=1)
    indices = ['label']
    values = ['RT/SKL-N=1', 'N=10', 'N=100', 'N=1000', 'N=10000']
    df1 = df1[indices + values]
    df2 = df2[indices + values]
    df = df1.merge(df2, on="label", suffixes=("__pyrtc", "__ort"), how='outer')

    na = df["RT/SKL-N=1__pyrtc"].isnull() & df["RT/SKL-N=1__ort"].isnull()
    dfp = df[~na].sort_values("label", ascending=False).reset_index(drop=True)

    # dfp = dfp[-10:]

    # We add the runtime name as model.
    ncol = (dfp.shape[1] - 1) // 2
    dfp_legend = dfp.iloc[:3, :].copy()
    dfp_legend.iloc[:, 1:] = numpy.nan
    dfp_legend.iloc[1, 1:1+ncol] = dfp.iloc[:, 1:1+ncol].mean()
    dfp_legend.iloc[2, 1+ncol:] = dfp.iloc[:, 1+ncol:].mean()
    dfp_legend.iloc[1, 0] = "avg_" + dfp_legend.columns[1].split('__')[-1]
    dfp_legend.iloc[2, 0] = "avg_" + dfp_legend.columns[1+ncol].split('__')[-1]
    dfp_legend.iloc[0, 0] = "------"

    rleg = dfp_legend.iloc[::-1, :].copy()
    rleg.iloc[1, 1:1+ncol] = dfp.iloc[:, 1:1+ncol].median()
    rleg.iloc[0, 1+ncol:] = dfp.iloc[:, 1+ncol:].median()
    rleg.iloc[1, 0] = "med_" + dfp_legend.columns[1].split('__')[-1]
    rleg.iloc[0, 0] = "med_" + dfp_legend.columns[1+ncol].split('__')[-1]

    # draw lines between models
    dfp = dfp.sort_values('label', ascending=False).copy()
    vals = dfp.iloc[:, 1:].values.ravel()
    xlim = [max(1e-3, min(0.5, min(vals))), min(1000, max(2, max(vals)))]
    i = 0
    while i < dfp.shape[0] - 1:
        i += 1
        label = dfp.iloc[i, 0]
        if '[' not in label:
            continue
        prev = dfp.iloc[i-1, 0]
        if '[' not in label:
            continue
        label = label.split()[0]
        prev = prev.split()[0]
        if _model_name(label) == _model_name(prev):
            continue

        blank = dfp.iloc[:1,:].copy()
        blank.iloc[0, 0] = '------'
        blank.iloc[0, 1:] = xlim[0]
        dfp = pandas.concat([dfp[:i], blank, dfp[i:]])
        i += 1
    dfp = dfp.reset_index(drop=True).copy()

    # add exhaustive statistics
    dfp = pandas.concat([rleg, dfp, dfp_legend]).reset_index(drop=True)
    dfp["x"] = numpy.arange(0, dfp.shape[0])

    # plot
    total = dfp.shape[0] * 0.5
    fig = plt.figure(figsize=(14, total))

    ax = list(None for c in range((dfp.shape[1]-1) // 2))
    p = 1.2
    b = 0.35
    for i in range(len(ax)):
        x1 = i * 1. / len(ax)
        x2 = (i + 0.95) * 1. / len(ax)
        x1 = x1 ** p
        x2 = x2 ** p
        x1 = b + (0.99 - b) * x1
        x2 = b + (0.99 - b) * x2
        bo = [x1, 0.1, x2 - x1, 0.8]
        if True or i == 0:
            ax[i] = fig.add_axes(bo)
        else:
            # Does not work because all graph shows the same
            # labels.
            ax[i] = fig.add_axes(bo, sharey=ax[i-1])

    # fig, ax = plt.subplots(1, (dfp.shape[1]-1) // 2, figsize=(14, total),
    #                        sharex=False, sharey=True)
    x = dfp['x']
    height = total / dfp.shape[0] * 0.65
    for c in df.columns[1:]:
        place, runtime = c.split('__')
        dec = {'pyrtc': 1, 'ort': -1}
        index = values.index(place)
        yl = dfp.loc[:, c].fillna(0)
        xl = xl = x + dec[runtime] * height / 2
        ax[index].barh(xl, yl, label=runtime, height=height)
        ax[index].set_title(place)
    for i in range(len(ax)):
        ax[i].plot([1, 1], [min(x), max(x)], 'g-')
        ax[i].plot([2, 2], [min(x), max(x)], 'r--')
        ax[i].plot([5, 5], [min(x), max(x)], 'r--', lw=3)
        ax[i].set_xscale('log')
        ax[i].set_xlim(xlim)
        ax[i].set_ylim([min(x) - 2, max(x) + 1])

    for i in range(1, len(ax)):
        ax[i].set_yticklabels([])

    ax[0].set_yticks(x)
    ax[0].set_yticklabels(dfp['label'])
    fig.subplots_adjust(left=0.35)

    plt.show()
