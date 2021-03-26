"""
.. _l-b-transpose:

Compares implementations of ReduceSum
=====================================

The following function benchmark different implementation
of function :epkg:`numpy:sum`.
It compares *numpy* implementation to :epkg:`onnxruntime` implementation.
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
from skl2onnx.algebra.onnx_ops import OnnxReduceSum_11
from mlprodict.tools import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c import code_optimisation
print(code_optimisation())

###################################
# ReduceSum implementations
# +++++++++++++++++++++++++

try:
    from tensorflow.math import reduce_sum as tf_reduce_sum
    from tensorflow import convert_to_tensor
except ImportError:
    reduce_sum = None
try:
    from torch import sum as torch_sum, from_numpy
except ImportError:
    torch_sum = None


def build_ort_reducesum(axes, op_version=12):
    node = OnnxReduceSum_11('x', axes=axes, op_version=op_version,
                            output_names=['z'])
    onx = node.to_onnx(inputs=[('x', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x})


def loop_fct(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


def benchmark_op(axes, repeat=5, number=5, name="reducesum", shape_fct=None):
    if shape_fct is None:
        def shape_fct(dim):
            return (3, dim, 1, 128, 64)
    ort_fct = build_ort_reducesum(axes)
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 400, 512, 1024]):
        shape = shape_fct(dim)
        n_arrays = 10 if dim < 512 else 4
        xs = [numpy.random.rand(*shape).astype(numpy.float32)
              for _ in range(n_arrays)]
        ys = [numpy.array(axes, dtype=numpy.int64)
              for _ in range(n_arrays)]
        info = dict(axes=axes, shape=shape)

        # numpy
        ctx = dict(
            xs=xs, ys=ys,
            fct=lambda x, y: numpy.sum(x, *y),
            loop_fct=loop_fct)
        obs = measure_time(
            "loop_fct(fct, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy'
        obs.update(info)
        res.append(obs)

        # onnxruntime
        ctx['fct'] = ort_fct
        obs = measure_time(
            "loop_fct(fct, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort'
        obs.update(info)
        res.append(obs)

        if tf_reduce_sum is not None:
            # tensorflow
            ctx['fct'] = tf_reduce_sum
            ctx['xs'] = [convert_to_tensor(x) for x in xs]
            ctx['ys'] = ys
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'tf'
            obs.update(info)
            res.append(obs)

        if torch_sum is not None:
            def torch_sum1(x, y):
                return torch_sum(x, y[0])

            def torch_sum2(x, y):
                return torch_sum(torch_sum(x, y[1]), y[0])

            # torch
            ctx['fct'] = torch_sum1 if len(axes) == 1 else torch_sum2
            ctx['xs'] = [from_numpy(x) for x in xs]
            ctx['ys'] = ys  # [from_numpy(y) for y in ys]
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
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
                   "\nlower better" % (name, shape_name, axes))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%r - %r"
                  "\nhigher better" % (name, shape_name, axes))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})
    return df, piv, ax


dfs = []

###################################
# Reduction on a particular case
# ++++++++++++++++++++++++++++++
#
# (8, 24, 48, N)
# ++++++++++++++

axes = (1, 2)
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24, 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

# (8, 24 * 48, N)
# +++++++++++++++

axes = (1, )
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24 * 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

# (8, 24 * 48, N)
# +++++++++++++++

axes = (2, )
df, piv, ax = benchmark_op(axes, shape_fct=lambda dim: (8, 24 * 48, dim))
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Reduction on one axis
# +++++++++++++++++++++
#
# (1, )
# ^^^^^

axes = (2, )
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# (2, )
# ^^^^^

axes = (2, )
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# last dimension
# ^^^^^^^^^^^^^^

axes = (4, )
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T

###################################
# Reduction on two dimensions
# +++++++++++++++++++++++++++
#
# (3, 4)
# ^^^^^^

axes = (3, 4)
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T

###################################
# (2, 4)
# ^^^^^^

axes = (2, 4)
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# (1, 4)
# ^^^^^^

axes = (1, 4)
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# (2, 3)
# ^^^^^^

axes = (2, 3)
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# .. _l-reducesum-problem1:
#
# (1, 3)
# ^^^^^^

axes = (1, 3)
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T

###################################
# (0, 3)
# ^^^^^^

axes = (0, 3)
df, piv, ax = benchmark_op(axes)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


####################################
# Conclusion
# ++++++++++
#
# Some of the configurations should be investigated.
# :ref:`l-reducesum-problem1`. The reduction on tensorflow
# in one dimension seems to be lazy.

merged = pandas.concat(dfs)
name = "reducesum"
merged.to_csv("plot_%s.csv" % name, index=False)
merged.to_excel("plot_%s.xlsx" % name, index=False)
plt.savefig("plot_%s.png" % name)

plt.show()
