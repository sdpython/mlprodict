"""
@file
@brief Measures time processing for ONNX models.
"""
import numpy
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from ... import __version__ as ort_version
from .validate_helper import default_time_kwargs, measure_time


def benchmark_fct(fct, X, time_limit=4, obs=None, node_time=False,
                  time_kwargs=None, skip_long_test=True):
    """
    Benchmarks a function which takes an array
    as an input and changes the number of rows.

    @param      fct             function to benchmark, signature
                                is fct(xo)
    @param      X               array
    @param      time_limit      above this time, measurement as stopped
    @param      obs             all information available in a dictionary
    @param      node_time       measure time execution for each node in the graph
    @param      time_kwargs     to define a more precise way to measure a model
    @param      skip_long_test  skips tests for high values of N if they seem too long
    @return                     dictionary with the results

    The function uses *obs* to reduce the number of tries it does.
    :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`
    produces huge *NxN* if predict method is called
    with ``return_cov=True``.
    The default for *time_kwargs* is the following:

    .. runpython::
        :showcode:

        from mlprodict.onnxrt.validate.validate_helper import default_time_kwargs
        import pprint
        pprint.pprint(default_time_kwargs())

    See also notebook :ref:`onnxnodetimerst` to see how this function
    can be used to measure time spent in each node.
    """
    if time_kwargs is None:
        time_kwargs = default_time_kwargs()

    def make(x, n):
        if n < x.shape[0]:
            return x[:n].copy()
        elif len(x.shape) < 2:
            r = numpy.empty((n, ), dtype=x.dtype)
            for i in range(0, n, x.shape[0]):
                end = min(i + x.shape[0], n)
                r[i: end] = x[0: end - i]
        else:
            r = numpy.empty((n, x.shape[1]), dtype=x.dtype)
            for i in range(0, n, x.shape[0]):
                end = min(i + x.shape[0], n)
                r[i: end, :] = x[0: end - i, :]
        return r

    def allow(N, obs):
        if obs is None:
            return True
        prob = obs['problem']
        if "-cov" in prob and N > 1000:
            return False
        return True

    Ns = list(sorted(time_kwargs))
    res = {}
    for N in Ns:
        if not isinstance(N, int):
            raise RuntimeError("time_kwargs ({}) is wrong:\n{}".format(
                type(time_kwargs), time_kwargs))
        if not allow(N, obs):
            continue
        x = make(X, N)
        number = time_kwargs[N]['number']
        repeat = time_kwargs[N]['repeat']
        if node_time:
            fct(x)
            main = None
            for __ in range(repeat):
                agg = None
                for _ in range(number):
                    ms = fct(x)[1]
                    if agg is None:
                        agg = ms
                        for row in agg:
                            row['N'] = N
                    else:
                        if len(agg) != len(ms):
                            raise RuntimeError(
                                "Not the same number of nodes {} != {}.".format(len(agg), len(ms)))
                        for a, b in zip(agg, ms):
                            a['time'] += b['time']
                if main is None:
                    main = agg
                else:
                    if len(agg) != len(main):
                        raise RuntimeError(
                            "Not the same number of nodes {} != {}.".format(len(agg), len(main)))
                    for a, b in zip(main, agg):
                        a['time'] += b['time']
                        a['max_time'] = max(
                            a.get('max_time', b['time']), b['time'])
                        a['min_time'] = min(
                            a.get('min_time', b['time']), b['time'])
            for row in main:
                row['repeat'] = repeat
                row['number'] = number
                row['time'] /= repeat * number
                if 'max_time' in row:
                    row['max_time'] /= number
                    row['min_time'] /= number
                else:
                    row['max_time'] = row['time']
                    row['min_time'] = row['time']
            res[N] = main
        else:
            res[N] = measure_time(fct, x, repeat=repeat,
                                  number=number, div_by_number=True)
        if (skip_long_test and not node_time and
                res[N] is not None and
                res[N].get('total', time_limit) >= time_limit):
            # too long
            break
    if node_time:
        rows = []
        for _, v in res.items():
            rows.extend(v)
        return rows
    else:
        return res
