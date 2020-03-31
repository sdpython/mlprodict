"""
@file
@brief Command line about validation of prediction runtime.
"""
from pandas import DataFrame


def benchmark_replay(folder, runtime='python', time_kwargs=None,
                     skip_long_test=True, time_kwargs_fact=None,
                     time_limit=4, out=None, verbose=1, fLOG=print):
    """
    The command rerun a benchmark if models were stored by
    command line `vaidate_runtime`.

    :param folder: where to find pickled files
    :param runtime: runtimes, comma separated list
    :param verbose: integer from 0 (None) to 2 (full verbose)
    :param out: output raw results into this file (excel format)
    :param time_kwargs: a dictionary which defines the number of rows and
        the parameter *number* and *repeat* when benchmarking a model,
        the value must follow :epkg:`json` format
    :param skip_long_test: skips tests for high values of N if
        they seem too long
    :param time_kwargs_fact: to multiply number and repeat in
        *time_kwargs* depending on the model
        (see :func:`_multiply_time_kwargs <mlprodict.onnxrt.validate.validate_helper._multiply_time_kwargs>`)
    :param time_limit: to stop benchmarking after this limit of time
    :param fLOG: logging function

    .. cmdref::
        :title: Replay a benchmark of stored converted models by validate_runtime
        :cmd: -m mlprodict benchmark_replay --help
        :lid: l-cmd-benchmark_replay

        The command rerun a benchmark if models were stored by
        command line `vaidate_runtime`.

        Example::

            python -m mlprodict benchmark_replay --folder dumped --out bench_results.xlsx

        Parameter ``--time_kwargs`` may be used to reduce or increase
        bencharmak precisions. The following value tells the function
        to run a benchmarks with datasets of 1 or 10 number, to repeat
        a given number of time *number* predictions in one row.
        The total time is divided by :math:`number \\times repeat``.
        Parameter ``--time_kwargs_fact`` may be used to increase these
        number for some specific models. ``'lin'`` multiplies
        by 10 number when the model is linear.

        ::

            -t "{\\"1\\":{\\"number\\":10,\\"repeat\\":10},\\"10\\":{\\"number\\":5,\\"repeat\\":5}}"
    """
    from ..onnxrt.validate.validate_benchmark_replay import enumerate_benchmark_replay  # pylint: disable=E0402

    rows = list(enumerate_benchmark_replay(
        folder=folder, runtime=runtime, time_kwargs=time_kwargs,
        skip_long_test=skip_long_test, time_kwargs_fact=time_kwargs_fact,
        time_limit=time_limit, verbose=verbose, fLOG=fLOG))
    if out is not None:
        df = DataFrame(rows)
        df.to_excel(out, index=False)
    return rows
