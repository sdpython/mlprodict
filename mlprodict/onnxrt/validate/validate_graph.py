"""
@file
@brief Functions to help visualizing performances.
"""
import numpy


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
        from mlprodict.onnxrt.validate.validate import enumerate_validated_operator_opsets, summary_report
        from mlprodict.onnxrt.validate.validate_graph import plot_validate_benchmark

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
        df["n_features"] = 4
    if 'runtime' not in df.columns:
        df['runtime'] = '?'

    fmt = "{} [{}-{}] D{}"
    df["label"] = df.apply(lambda row: fmt.format(
                           row["name"], row["problem"], row["scenario"],
                           row["n_features"]).replace("-default]", "]"), axis=1)
    df = df.sort_values(["name", "problem", "scenario", "n_features", "runtime"],
                        ascending=False).reset_index(drop=True).copy()
    indices = ['label', 'runtime']
    values = [c for c in df.columns
              if 'N=' in c and '-min' not in c and '-max' not in c]
    try:
        df = df[indices + values]
    except KeyError as e:
        raise RuntimeError("Unable to find the following columns {}\nin {}".format(
            indices + values, df.columns)) from e

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

    total = final.shape[0] * 0.45
    fig, ax = plt.subplots(1, len(values), figsize=(14, total),
                           sharex=False, sharey=True)
    x = numpy.arange(final.shape[0])
    subh = 1.0 / len(runtimes)
    height = total / final.shape[0] * (subh + 0.1)
    decrt = {rt: height * i for i, rt in enumerate(runtimes)}
    colors = {rt: c for rt, c in zip(
        runtimes, ['blue', 'orange', 'cyan', 'yellow'])}

    done = set()
    for c in final.columns[1:]:
        place, runtime = c.split('__')
        if hasattr(ax, 'shape'):
            index = values.index(place)
            if (index, runtime) in done:
                raise RuntimeError(
                    "Issue with column '{}'\nlabels={}\nruntimes={}\ncolumns={}\nvalues={}\n{}".format(
                        c, list(final.label), runtimes, final.columns, values, final))
            axi = ax[index]
            done.add((index, runtime))
        else:
            if (0, runtime) in done:
                raise RuntimeError(
                    "Issue with column '{}'\nlabels={}\nruntimes={}\ncolumns={}\nvalues={}\n{}".format(
                        c, final.label, runtimes, final.columns, values, final))
            done.add((0, runtime))
            axi = ax
        if c in final.columns:
            yl = final.loc[:, c]
            xl = x + decrt[runtime] / 2
            axi.barh(xl, yl, label=runtime, height=height,
                     color=colors[runtime])
            axi.set_title(place)

    if hasattr(ax, 'shape'):
        for i in range(len(ax)):  # pylint: disable=C0200
            ax[i].plot([1, 1], [0, max(x)], 'g-')
            ax[i].plot([2, 2], [0, max(x)], 'r--')
            ax[i].set_xscale('log')
        ax[min(ax.shape[0] - 1, 2)].legend()
        ax[0].set_yticks(x)
        ax[0].set_yticklabels(final['label'])
    else:
        ax.plot([1, 1], [0, max(x)], 'g-')
        ax.plot([2, 2], [0, max(x)], 'r--')
        ax.set_xscale('log')
        ax.legend()
        ax.set_yticks(x)
        ax.set_yticklabels(final['label'])

    fig.subplots_adjust(left=0.25)
    return fig, ax
