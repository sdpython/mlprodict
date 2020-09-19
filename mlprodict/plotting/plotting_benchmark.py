"""
@file
@brief Useful plots.
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel=None, **kwargs):
    """
    Creates a heatmap from a numpy array and two lists of labels.
    See @see fn plot_benchmark_metrics for an example.

    @param  data        a 2D numpy array of shape (N, M).
    @param  row_labels  a list or array of length N with the labels for the rows.
    @param  col_labels  a list or array of length M with the labels for the columns.
    @param  ax          a `matplotlib.axes.Axes` instance to which the heatmap is plotted,
                        if not provided, use current axes or create a new one. Optional.
    @param  cbar_kw     a dictionary with arguments to `matplotlib.Figure.colorbar
                        <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.colorbar.html>`_.
                        Optional.
    @param  cbarlabel   the label for the colorbar. Optional.
    @param   kwargs     all other arguments are forwarded to `imshow
                        <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_
    @return             ax, image, color bar
    """

    if not ax:
        ax = plt.gca()  # pragma: no cover

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbar_kw is None:
        cbar_kw = {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    if cbarlabel is not None:
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return ax, im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "black"),
                     threshold=None, **textkw):
    """
    Annotates a heatmap.
    See @see fn plot_benchmark_metrics for an example.

    @param  im          the *AxesImage* to be labeled.
    @param  data        data used to annotate. If None, the image's data is used. Optional.
    @param  valfmt      the format of the annotations inside the heatmap. This should either
                        use the string format method, e.g. `"$ {x:.2f}"`, or be a
                        `matplotlib.ticker.Formatter
                        <https://matplotlib.org/api/ticker_api.html>`_. Optional.
    @param  textcolors  a list or array of two color specifications.  The first is used for
                        values below a threshold, the second for those above. Optional.
    @param  threshold   value in data units according to which the colors from textcolors are
                        applied. If None (the default) uses the middle of the colormap as
                        separation. Optional.
    @param  textkw      all other arguments are forwarded to each call to `text` used to create
                        the text labels.
    @return             annotated objects
    """
    if not isinstance(data, (list, numpy.ndarray)):
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)  # pragma: no cover
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_benchmark_metrics(metric, xlabel=None, ylabel=None,
                           middle=1., transpose=False, ax=None,
                           cbar_kw=None, cbarlabel=None,
                           valfmt="{x:.2f}x"):
    """
    Plots a heatmap which represents a benchmark.
    See example below.

    @param      metric      dictionary ``{ (x,y): value }``
    @param      xlabel      x label
    @param      ylabel      y label
    @param      middle      force the white color to be this value
    @param      transpose   switches *x* and *y*
    @param      ax          axis to borrow
    @param      cbar_kw     a dictionary with arguments to `matplotlib.Figure.colorbar
                            <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.colorbar.html>`_.
                            Optional.
    @param      cbarlabel   the label for the colorbar. Optional.
    @param      valfmt      format for the annotations
    @return                 ax, colorbar

    .. exref::
        :title: Plot benchmark improvments
        :lid: plot-2d-benchmark-metric

        .. plot::

            import matplotlib.pyplot as plt
            from mlprodict.tools.plotting import plot_benchmark_metrics

            data = {(1, 1): 0.1, (10, 1): 1, (1, 10): 2,
                    (10, 10): 100, (100, 1): 100, (100, 10): 1000}

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            plot_benchmark_metrics(data, ax=ax[0], cbar_kw={'shrink': 0.6})
            plot_benchmark_metrics(data, ax=ax[1], transpose=True,
                                   xlabel='X', ylabel='Y',
                                   cbarlabel="ratio")
            plt.show()
    """
    if transpose:
        metric = {(k[1], k[0]): v for k, v in metric.items()}
        return plot_benchmark_metrics(metric, ax=ax, xlabel=ylabel, ylabel=xlabel,
                                      middle=middle, transpose=False,
                                      cbar_kw=cbar_kw, cbarlabel=cbarlabel)

    x = numpy.array(list(sorted(set(k[0] for k in metric))))
    y = numpy.array(list(sorted(set(k[1] for k in metric))))
    rx = {v: i for i, v in enumerate(x)}
    ry = {v: i for i, v in enumerate(y)}

    X, _ = numpy.meshgrid(x, y)
    zm = numpy.zeros(X.shape, dtype=numpy.float64)
    for k, v in metric.items():
        zm[ry[k[1]], rx[k[0]]] = v

    xs = [str(_) for _ in x]
    ys = [str(_) for _ in y]
    vmin = min(metric.values())
    vmax = max(metric.values())
    if middle is not None:
        v1 = middle / vmin
        v2 = middle / vmax
        vmin = min(vmin, v2)
        vmax = max(vmax, v1)
    ax, im, cbar = heatmap(zm, ys, xs, ax=ax, cmap="bwr",
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           cbarlabel=cbarlabel, cbar_kw=cbar_kw)
    annotate_heatmap(im, valfmt=valfmt)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax, cbar
