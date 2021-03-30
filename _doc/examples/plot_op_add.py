"""
.. _l-b-transpose:

Compares implementations of Add
===============================

The following function benchmark different implementation
of function :epkg:`numpy:add`.
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
from skl2onnx.algebra.onnx_ops import OnnxAdd
from mlprodict.tools import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c import code_optimisation
print(code_optimisation())

###################################
# Add implementations
# +++++++++++++++++++

try:
    from tensorflow.math import add as tf_add
    from tensorflow import convert_to_tensor
except ImportError:
    reduce_sum = None
try:
    from torch import add as torch_add, from_numpy
except ImportError:
    torch_sum = None


def build_ort_add(op_version=12):
    node = OnnxAdd('x', 'y', op_version=op_version, output_names=['z'])
    onx = node.to_onnx(inputs=[('x', FloatTensorType()),
                               ('y', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x, 'y': y})


def loop_fct(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


def benchmark_op(repeat=5, number=5, name="add", shape_fcts=None):
    if shape_fcts is None:
        def shape_fct(dim):
            return (5, dim, dim)
        shape_fcts = (shape_fct, shape_fct)
    ort_fct = build_ort_add()
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 400, 512, 1024, 1536, 2048, 2560]):
        shape1 = shape_fcts[0](dim)
        shape2 = shape_fcts[1](dim)
        n_arrays = (16 if dim < 512 else 4) if dim < 2048 else 4
        if len(shape1) > 3:
            n_arrays = int(n_arrays / 4)
        xs = [numpy.random.rand(*shape1).astype(numpy.float32)
              for _ in range(n_arrays)]
        ys = [numpy.random.rand(*shape2).astype(numpy.float32)
              for _ in range(n_arrays)]
        info = dict(shape1=shape1, shape2=shape2)

        # numpy
        ctx = dict(
            xs=xs, ys=ys,
            fct=lambda x, y: numpy.add(x, y),
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

        if tf_add is not None:
            # tensorflow
            ctx['fct'] = tf_add
            ctx['xs'] = [convert_to_tensor(x) for x in xs]
            ctx['ys'] = [convert_to_tensor(y) for y in ys]
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'tf'
            obs.update(info)
            res.append(obs)

        if torch_add is not None:
            # torch
            ctx['fct'] = torch_add
            ctx['xs'] = [from_numpy(x) for x in xs]
            ctx['ys'] = [from_numpy(y) for y in ys]
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'torch'
            obs.update(info)
            res.append(obs)

    # Dataframes
    shape1_name = str(shape1).replace(str(dim), "N")
    shape2_name = str(shape2).replace(str(dim), "N")
    df = pandas.DataFrame(res)
    df.columns = [_.replace('dim', 'N') for _ in df.columns]
    piv = df.pivot('N', 'fct', 'average')

    rs = piv.copy()
    for c in ['ort', 'torch', 'tf']:
        if c in rs.columns:
            rs[c] = rs['numpy'] / rs[c]
    rs['numpy'] = 1.

    # Graphs.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(logx=True, logy=True, ax=ax[0],
             title="%s benchmark\n%s + %s"
                   "\nlower better" % (name, shape1_name, shape2_name))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%s + %s"
                  "\nhigher better" % (name, shape1_name, shape2_name))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})
    return df, piv, ax


dfs = []

###################################
# (5, N, N) + (5, N, N)
# +++++++++++++++++++++

df, piv, ax = benchmark_op()
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# (5, N, N) + (5, N, 1)
# +++++++++++++++++++++

shape_fcts = (lambda dim: (5, dim, dim),
              lambda dim: (5, dim, 1))

df, piv, ax = benchmark_op(shape_fcts=shape_fcts)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# (5, N, N) + (5, 1, N)
# +++++++++++++++++++++

shape_fcts = (lambda dim: (5, dim, dim),
              lambda dim: (5, 1, dim))

df, piv, ax = benchmark_op(shape_fcts=shape_fcts)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# (5, N, 5, N) + (1, N, 1, 1)
# +++++++++++++++++++++++++++

shape_fcts = (lambda dim: (5, dim, 5, dim),
              lambda dim: (1, dim, 1, 1))

df, piv, ax = benchmark_op(shape_fcts=shape_fcts)
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Conclusion
# ++++++++++
#
# *onnxruntime* is slower in anycase.
# *pytorch* is faster when there is zero or one broadcasted axis.
# *tf* is faster with two broadcasted axes.

merged = pandas.concat(dfs)
name = "add"
merged.to_csv("plot_%s.csv" % name, index=False)
merged.to_excel("plot_%s.xlsx" % name, index=False)
plt.savefig("plot_%s.png" % name)

plt.show()
