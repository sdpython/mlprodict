"""
.. _l-b-reducesummean:

Compares implementations of ReduceMean
======================================

This example compares the *numpy* for the operator *ReduceMean*
to :epkg:`onnxruntime` implementation.
If available, :epkg:`tensorflow` and :epkg:`pytorch` are included as well.

.. contents::
    :local:

Available optimisation
++++++++++++++++++++++

The code shows which parallelisation optimisation could be used,
*AVX* or *SSE* and the number of available processors.
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxReduceMean
from cpyquickhelper.numbers import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())

###################################
# ReduceMean implementations
# +++++++++++++++++++++++++

try:
    from tensorflow.math import reduce_mean as tf_reduce_mean
    from tensorflow import convert_to_tensor
except ImportError:
    tf_reduce_mean = None
try:
    from torch import mean as torch_mean, from_numpy
except ImportError:
    torch_mean = None


def build_ort_reducemean(axes, op_version=14):  # opset=13, 14, ...
    node = OnnxReduceMean('x', axes=axes, op_version=op_version,
                          output_names=['z'])
    onx = node.to_onnx(inputs=[('x', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x})


def loop_fct(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


def benchmark_op(axes, repeat=2, number=5, name="ReduceMean",
                 shape_fct=None, max_dim=None):
    if shape_fct is None:
        def shape_fct(dim):
            return (3, dim, 1, 128, 64)
    ort_fct = build_ort_reducemean(axes)
    res = []
    for dim in tqdm([4, 8, 16, 32, 64, 100, 128, 200,
                     256, 400, 512, 1024]):
        if max_dim is not None and dim > max_dim:
            continue
        shape = shape_fct(dim)
        n_arrays = 10 if dim < 512 else 4
        xs = [numpy.random.rand(*shape).astype(numpy.float32)
              for _ in range(n_arrays)]
        ys = [numpy.array(axes, dtype=numpy.int64)
              for _ in range(n_arrays)]
        info = dict(axes=axes, shape=shape)

        # numpy
        fct = lambda x, y: numpy.mean(x, axis=tuple(y))
        ctx = dict(
            xs=xs, ys=ys,
            loop_fct=loop_fct)
        obs = measure_time(
            lambda: loop_fct(fct, xs, ys),
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy'
        obs.update(info)
        res.append(obs)

        # onnxruntime
        fct = ort_fct
        obs = measure_time(
            lambda: loop_fct(fct, xs, ys),
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort'
        obs.update(info)
        res.append(obs)

        if tf_reduce_mean is not None:
            # tensorflow
            fct = tf_reduce_mean
            ctx['xs'] = [convert_to_tensor(x) for x in xs]
            ctx['ys'] = ys
            obs = measure_time(
                lambda: loop_fct(fct, ctx['xs'], ctx['ys']),
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'tf'
            obs.update(info)
            res.append(obs)

        if torch_mean is not None:
            def torch_mean1(x, y):
                return torch_mean(x, y[0])

            def torch_mean2(x, y):
                return torch_mean(torch_mean(x, y[1]), y[0])

            # torch
            fct = torch_mean1 if len(axes) == 1 else torch_mean2
            ctx['xs'] = [from_numpy(x) for x in xs]
            ctx['ys'] = ys  # [from_numpy(y) for y in ys]
            obs = measure_time(
                lambda: loop_fct(fct, ctx['xs'], ctx['ys']),
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'torch'
            obs.update(info)
            res.append(obs)

    # Dataframes
    shape_name = str(shape).replace(str(dim), "N")
    df = pandas.DataFrame(res)
    df.columns = [_.replace('dim', 'N') for _ in df.columns]
    piv = df.pivot('N', 'fct', 'average')

    rs = piv.copy()
    for c in ['ort', 'torch', 'tf', 'tf_copy']:
        if c in rs.columns:
            rs[c] = rs['numpy'] / rs[c]
    rs['numpy'] = 1.

    # Graphs.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(logx=True, logy=True, ax=ax[0],
             title="%s benchmark\n%r - %r"
                   " lower better" % (name, shape_name, axes))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%r - %r"
                  " higher better" % (name, shape_name, axes))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})
    return df, rs, ax


dfs = []

###################################
# Reduction on a particular case KR
# +++++++++++++++++++++++++++++++++
#
# Consecutive axis not reduced and consecutive reduced
# axis are merged.
# KR means kept axis - reduced axis
#
# (8, 24, 48, N), axis=(3, )
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

axes = (3, )
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24, 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Reduction on a particular case RK
# +++++++++++++++++++++++++++++++++
#
# Consecutive axis not reduced and consecutive reduced
# axis are merged.
# RK means reduced axis - kept axis
#
# (8, 24, 48, N), axis=(0, )
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

axes = (0, )
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24, 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Reduction on a particular case KRK
# ++++++++++++++++++++++++++++++++++
#
# Consecutive axis not reduced and consecutive reduced
# axis are merged.
# KRK means kept axis - reduced axis - kept axis,
#
# (8, 24, 48, N), axis=(1, 2)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

axes = (1, 2)
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24, 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# (8, 24 * 48, N), axis=1
# ^^^^^^^^^^^^^^^^^^^^^^^

axes = (1, )
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24 * 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# (2, 8, 12, 24, 2, N), axis=(2, 3)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

axes = (2, 3)
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (2, 8, 12, 24, 2, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Reduction on a particular case RKR
# ++++++++++++++++++++++++++++++++++
#
# (N, 64, 16, 16), axis=(0, 2, 3)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

axes = (0, 2, 3)
df, piv, ax = benchmark_op(
    axes, shape_fct=lambda dim: (dim, 64, 16, 16))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Reduction on a particular case RKRK
# +++++++++++++++++++++++++++++++++++
#
# (8, 24, 48, N), axis=(0, 2)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

axes = (0, 2)
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24, 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Conclusion
# ++++++++++
#
# Some of the configurations should be investigated.
# :ref:`l-reducesum-problem1`. The reduction on tensorflow
# in one dimension seems to be lazy.

merged = pandas.concat(dfs)
name = "reducemean"
merged.to_csv("plot_%s.csv" % name, index=False)
merged.to_excel("plot_%s.xlsx" % name, index=False)
plt.savefig("plot_%s.png" % name)

plt.show()
