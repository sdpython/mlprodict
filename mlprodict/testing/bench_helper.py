"""
@file
@brief Helpers for benchmarks.
"""
from timeit import Timer
import numpy


def measure_time(stmt, *x, repeat=5, number=5, div_by_number=True, first_run=True):
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string
    :param *x: inputs
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param div_by_number: divide by the number of executions
    :param first_run: if True, runs the function once before measuring
    :return: dictionary

    See `Timer.repeat
    <https://docs.python.org/3/library/timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    try:
        stmt(*x)
    except RuntimeError as e:  # pragma: no cover
        raise RuntimeError("{}-{}".format(type(x), x.dtype)) from e

    def fct():
        stmt(*x)

    if first_run:
        fct()
    tim = Timer(fct)
    res = numpy.array(tim.repeat(repeat=repeat, number=number))
    total = numpy.sum(res)
    if div_by_number:
        res /= number
    mean = numpy.mean(res)
    dev = numpy.mean(res ** 2)
    dev = max(0, (dev - mean**2)) ** 0.5
    mes = dict(average=mean, deviation=dev, min_exec=numpy.min(res),
               max_exec=numpy.max(res), repeat=repeat, number=number,
               total=total)
    return mes
