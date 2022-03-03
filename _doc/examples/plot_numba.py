"""
.. _l-b-numpy-numba-ort:

Compares numba, numpy, onnxruntime for simple functions
=======================================================

The following benchmark is inspired from `bench_arrayexprs.py
<https://github.com/numba/numba-benchmark/blob/master/benchmarks/bench_arrayexprs.py>`_.
It compares :epkg:`numba`, :epkg:`numpy` and :epkg:`onnxruntime`
for simple functions. As expected, :epkg:`numba` is better than the other options.

.. contents::
    :local:

The functions
+++++++++++++
"""

import numpy
import pandas
import matplotlib.pyplot as plt
from numba import jit
from typing import Any
import numpy as np
from tqdm import tqdm
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.npy import NDArray, onnxnumpy_np
from mlprodict.npy.onnx_numpy_annotation import NDArrayType
import mlprodict.npy.numpy_onnx_impl as npnx


# @jit(nopython=True)
def sum(a, b):
    return a + b

# @jit(nopython=True)


def sq_diff(a, b):
    return (a - b) * (a + b)

# @jit(nopython=True)


def rel_diff(a, b):
    return (a - b) / (a + b)

# @jit(nopython=True)


def square(a):
    # Note this is currently slower than `a ** 2 + b`, due to how LLVM
    # seems to lower the power intrinsic.  It's still faster than the naive
    # lowering as `exp(2 * log(a))`, though
    return a ** 2


def cube(a):
    return a ** 3

#########################################
# ONNX version
# ++++++++++
#
# The implementation uses the numpy API for ONNX to keep the same code.


@onnxnumpy_np(signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)),
              runtime="onnxruntime")
def onnx_sum_32(a, b):
    return a + b


@onnxnumpy_np(signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)),
              runtime="onnxruntime")
def onnx_sq_diff_32(a, b):
    return (a - b) * (a + b)


@onnxnumpy_np(signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)),
              runtime="onnxruntime")
def onnx_rel_diff_32(a, b):
    return (a - b) / (a + b)


@onnxnumpy_np(signature=NDArrayType(("T:all", ), dtypes_out=('T',)),
              runtime="onnxruntime")
def onnx_square_32(a):
    return a ** 2


@onnxnumpy_np(signature=NDArrayType(("T:all", ), dtypes_out=('T',)),
              runtime="onnxruntime")
def onnx_cube_32(a):
    return a ** 3


################################################
# numba optimized
# ++++++++++++

jitter = jit(nopython=True)
nu_sum = jitter(sum)
nu_sq_diff = jitter(sq_diff)
nu_rel_diff = jitter(rel_diff)
nu_square = jitter(square)
nu_cube = jitter(cube)

#######################################
# Benchmark
# ++++++++

obs = []

for n in tqdm([10, 100, 1000, 10000, 100000, 1000000]):
    number = 100 if n < 1000000 else 10
    for dtype in [numpy.float32, numpy.float64]:
        samples = [
            [numpy.random.uniform(1.0, 2.0, size=n).astype(dtype)],
            [numpy.random.uniform(1.0, 2.0, size=n).astype(dtype)
             for i in range(2)]]

        for fct1, fct2, fct3, n_inputs in [
                (sum, nu_sum, onnx_sum_32, 2),
                (sq_diff, nu_sq_diff, onnx_sq_diff_32, 2),
                (rel_diff, nu_rel_diff, onnx_rel_diff_32, 2),
                (square, nu_square, onnx_square_32, 1),
                (cube, nu_cube, onnx_cube_32, 1)]:
            sample = samples[n_inputs - 1]
            if n_inputs == 2:
                fct1(*sample)
                fct1(*sample)
                r = measure_time('fct1(a,b)', number=number, div_by_number=True,
                                 context={'fct1': fct1, 'a': sample[0], 'b': sample[1]})
                r.update(dict(dtype=dtype, name='numpy', n=n, fct=fct1.__name__))
                obs.append(r)

                fct2(*sample)
                fct2(*sample)
                r = measure_time('fct2(a,b)', number=number, div_by_number=True,
                                 context={'fct2': fct2, 'a': sample[0], 'b': sample[1]})
                r.update(dict(dtype=dtype, name='numba', n=n, fct=fct1.__name__))
                obs.append(r)

                fct3(*sample)
                fct3(*sample)
                r = measure_time('fct3(a,b)', number=number, div_by_number=True,
                                 context={'fct3': fct3, 'a': sample[0], 'b': sample[1]})
                r.update(dict(dtype=dtype, name='onnx', n=n, fct=fct1.__name__))
                obs.append(r)
            else:
                fct1(*sample)
                fct1(*sample)
                r = measure_time('fct1(a)', number=number, div_by_number=True,
                                 context={'fct1': fct1, 'a': sample[0]})
                r.update(dict(dtype=dtype, name='numpy', n=n, fct=fct1.__name__))
                obs.append(r)

                fct2(*sample)
                fct2(*sample)
                r = measure_time('fct2(a)', number=number, div_by_number=True,
                                 context={'fct2': fct2, 'a': sample[0]})
                r.update(dict(dtype=dtype, name='numba', n=n, fct=fct1.__name__))
                obs.append(r)

                fct3(*sample)
                fct3(*sample)
                r = measure_time('fct3(a)', number=number, div_by_number=True,
                                 context={'fct3': fct3, 'a': sample[0]})
                r.update(dict(dtype=dtype, name='onnx', n=n, fct=fct1.__name__))
                obs.append(r)

df = pandas.DataFrame(obs)
print(df)


#######################################
# Graphs
# +++++

fcts = list(sorted(set(df.fct)))
fig, ax = plt.subplots(len(fcts), 2, figsize=(14, len(fcts) * 3))

for i, fn in enumerate(fcts):
    piv = pandas.pivot(data=df[(df.fct == fn) & (df.dtype == numpy.float32)],
                       index="n", columns="name", values="average")
    piv.plot(title="fct=%s - float32" % fn,
             logx=True, logy=True, ax=ax[i, 0])
    piv = pandas.pivot(data=df[(df.fct == fn) & (df.dtype == numpy.float64)],
                       index="n", columns="name", values="average")
    piv.plot(title="fct=%s - float64" % fn,
             logx=True, logy=True, ax=ax[i, 1])
plt.show()
