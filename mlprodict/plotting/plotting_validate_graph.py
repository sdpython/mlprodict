"""
@file
@brief Functions to help visualizing performances.
"""
import numpy
import pandas


def _model_name(name):
    """
    Extracts the main component of a model, removes
    suffixes such ``Classifier``, ``Regressor``, ``CV``.

    @param      name        string
    @return                 shorter string
    """
    if name.startswith("Select"):
        return "Select"
    if name.startswith("Nu"):
        return "Nu"
    modif = 1
    while modif > 0:
        modif = 0
        for suf in ['Classifier', 'Regressor', 'CV', 'IC',
                    'Transformer']:
            if name.endswith(suf):
                name = name[:-len(suf)]
                modif += 1
    return name


def plot_validate_benchmark(df):
    """
    Plots a graph which summarizes the performances of a benchmark
    validating a runtime for :epkg:`ONNX`.

    @param      df      output of function @see fn summary_report
    @return             fig, ax

    .. plot::

        from logging import getLogger
        from pandas import DataFrame
        import matplotlib.pyplot as plt
        from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
        from mlprodict.tools.plotting import plot_validate_benchmark

        logger = getLogger('skl2onnx')
        logger.disabled = True

        rows = list(enumerate_validated_operator_opsets(
            verbose=0, models={"LinearRegression"}, opset_min=11,
            runtime=['python', 'onnxruntime1'], debug=False,
            benchmark=True, n_features=[None, 10]))

        df = DataFrame(rows)
        piv = summary_report(df)
        fig, ax = plot_validate_benchmark(piv)
        plt.show()
    """
    import matplotlib.pyplot as plt

    if 'n_features' not in df.columns:
        df["n_features"] = numpy.nan  # pragma: no cover
    if 'runtime' not in df.columns:
        df['runtime'] = '?'  # pragma: no cover

    fmt = "{} [{}-{}|{}] D{}"
    df["label"] = df.apply(
        lambda row: fmt.format(
            row["name"], row["problem"], row["scenario"],
            row['optim'], row["n_features"]).replace("-default|", "-**]"), axis=1)
    df = df.sort_values(["name", "problem", "scenario", "optim",
                         "n_features", "runtime"],
                        ascending=False).reset_index(drop=True).copy()
    indices = ['label', 'runtime']
    values = [c for c in df.columns
              if 'N=' in c and '-min' not in c and '-max' not in c]
    try:
        df = df[indices + values]
    except KeyError as e:  # pragma: no cover
        raise RuntimeError(
            "Unable to find the following columns {}\nin {}".format(
                indices + values, df.columns)) from e

    if 'RT/SKL-N=1' not in df.columns:
        raise RuntimeError(  # pragma: no cover
            "Column 'RT/SKL-N=1' is missing, benchmark was probably not run.")
    na = df["RT/SKL-N=1"].isnull()
    dfp = df[~na]
    runtimes = list(sorted(set(dfp['runtime'])))
    final = None
    for rt in runtimes:
        sub = dfp[dfp.runtime == rt].drop('runtime', axis=1).copy()
        col = list(sub.columns)
        for i in range(1, len(col)):
            col[i] += "__" + rt
        sub.columns = col

        if final is None:
            final = sub
        else:
            final = final.merge(sub, on='label', how='outer')

    # let's add average and median
    ncol = (final.shape[1] - 1) // len(runtimes)
    if len(runtimes) + 1 > final.shape[0]:
        dfp_legend = final.iloc[:len(runtimes) + 1, :].copy()
        while dfp_legend.shape[0] < len(runtimes) + 1:
            dfp_legend = pandas.concat([dfp_legend, dfp_legend[:1]])
    else:
        dfp_legend = final.iloc[:len(runtimes) + 1, :].copy()
    rleg = dfp_legend.copy()
    dfp_legend.iloc[:, 1:] = numpy.nan
    rleg.iloc[:, 1:] = numpy.nan

    for r, runt in enumerate(runtimes):
        sli = slice(1 + ncol * r, 1 + ncol * r + ncol)
        cm = final.iloc[:, sli].mean().values
        dfp_legend.iloc[r + 1, sli] = cm
        rleg.iloc[r, sli] = final.iloc[:, sli].median()
        dfp_legend.iloc[r + 1, 0] = "avg_" + runt
        rleg.iloc[r, 0] = "med_" + runt
    dfp_legend.iloc[0, 0] = "------"
    rleg.iloc[-1, 0] = "------"

    # sort
    final = final.sort_values('label', ascending=False).copy()

    # add global statistics
    final = pandas.concat([rleg, final, dfp_legend]).reset_index(drop=True)

    # graph beginning
    total = final.shape[0] * 0.45
    fig, ax = plt.subplots(1, len(values), figsize=(14, total),
                           sharex=False, sharey=True)
    x = numpy.arange(final.shape[0])
    subh = 1.0 / len(runtimes)
    height = total / final.shape[0] * (subh + 0.1)
    decrt = {rt: height * i for i, rt in enumerate(runtimes)}
    colors = {rt: c for rt, c in zip(
        runtimes, ['blue', 'orange', 'cyan', 'yellow'])}

    # draw lines between models
    vals = final.iloc[:, 1:].values.ravel()
    xlim = [min(0.5, min(vals)), max(2, max(vals))]
    while i < final.shape[0] - 1:
        i += 1
        label = final.iloc[i, 0]
        if '[' not in label:
            continue
        prev = final.iloc[i - 1, 0]
        if '[' not in label:
            continue  # pragma: no cover
        label = label.split()[0]
        prev = prev.split()[0]
        if _model_name(label) == _model_name(prev):
            continue

        blank = final.iloc[:1, :].copy()
        blank.iloc[0, 0] = '------'
        blank.iloc[0, 1:] = xlim[0]
        final = pandas.concat([final[:i], blank, final[i:]])
        i += 1

    final = final.reset_index(drop=True).copy()
    x = numpy.arange(final.shape[0])

    done = set()
    for c in final.columns[1:]:
        place, runtime = c.split('__')
        if hasattr(ax, 'shape'):
            index = values.index(place)
            if (index, runtime) in done:
                raise RuntimeError(  # pragma: no cover
                    "Issue with column '{}'\nlabels={}\nruntimes={}\ncolumns="
                    "{}\nvalues={}\n{}".format(
                        c, list(final.label), runtimes, final.columns, values, final))
            axi = ax[index]
            done.add((index, runtime))
        else:
            if (0, runtime) in done:  # pragma: no cover
                raise RuntimeError(
                    "Issue with column '{}'\nlabels={}\nruntimes={}\ncolumns="
                    "{}\nvalues={}\n{}".format(
                        c, final.label, runtimes, final.columns, values, final))
            done.add((0, runtime))  # pragma: no cover
            axi = ax  # pragma: no cover
        if c in final.columns:
            yl = final.loc[:, c]
            xl = x + decrt[runtime] / 2
            axi.barh(xl, yl, label=runtime, height=height,
                     color=colors[runtime])
            axi.set_title(place)

    def _plot_axis(axi, x, xlim):
        axi.plot([1, 1], [0, max(x)], 'g-')
        axi.plot([2, 2], [0, max(x)], 'r--')
        axi.set_xlim(xlim)
        axi.set_xscale('log')
        axi.set_ylim([min(x) - 2, max(x) + 1])

    def _plot_final(axi, x, final):
        axi.set_yticks(x)
        axi.set_yticklabels(final['label'])

    if hasattr(ax, 'shape'):
        for i in range(len(ax)):  # pylint: disable=C0200
            _plot_axis(ax[i], x, xlim)

        ax[min(ax.shape[0] - 1, 2)].legend()
        _plot_final(ax[0], x, final)
    else:  # pragma: no cover
        _plot_axis(ax, x, xlim)
        _plot_final(ax, x, final)
        ax.legend()

    fig.subplots_adjust(left=0.25)
    return fig, ax
