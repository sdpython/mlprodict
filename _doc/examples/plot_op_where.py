"""
.. _l-where:

Compares implementations of Where
=================================

This example compares implementations of function :epkg:`numpy:where`
from :epkg:`numpy`, :epkg:`onnxruntime`.
:epkg:`tensorflow` and :epkg:`pytorch` are included as well if available.
The benchmark also compares the operator *where* to an equivalent implementation
`where(c, x, y) = x * c - y * (c - 1)`.

.. contents::
    :local:

Available optimisation
++++++++++++++++++++++

"""
import numpy
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from skl2onnx.common.data_types import FloatTensorType, BooleanTensorType
from skl2onnx.algebra.onnx_ops import OnnxWhere, OnnxSub, OnnxMul
from cpyquickhelper.numbers import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())

###################################
# Where: common code
# ++++++++++++++++++

try:
    from tensorflow import where as tf_where, convert_to_tensor
except ImportError:
    tf_where = None
try:
    from torch import where as torch_where, from_numpy
except ImportError:
    torch_where = None


def build_ort_where(op_version=12):
    node = OnnxWhere('cond', 'x', 'y', op_version=op_version,
                     output_names=['z'])
    onx = node.to_onnx(inputs=[('cond', BooleanTensorType()),
                               ('x', FloatTensorType()),
                               ('y', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda cond, x, y: sess.run(None, {'cond': cond, 'x': x, 'y': y})


def build_ort_where_add(op_version=12):
    node = OnnxSub(
        OnnxMul('x', 'cond', op_version=op_version),
        OnnxMul('y',
                OnnxSub('cond', numpy.array([1], dtype=numpy.float32),
                        op_version=op_version),
                op_version=op_version),
        op_version=op_version, output_names=['z'])
    onx = node.to_onnx(inputs=[('cond', FloatTensorType()),
                               ('x', FloatTensorType()),
                               ('y', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda cond, x, y: sess.run(None, {'cond': cond, 'x': x, 'y': y})


def numpy_where_add(cond, x, y):
    cx = x * cond
    cy = cond - 1
    numpy.multiply(y, cy, out=y)
    numpy.subtract(cx, cy, out=cx)
    return cx


def loop_where(fct, conds, xs, ys):
    for cond, x, y in zip(conds, xs, ys):
        fct(cond, x, y)


def benchmark_equation():
    # equations
    ort_where = build_ort_where()
    ort_where_add = build_ort_where_add()
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 500, 512, 1024, 2048]):
        repeat = 5
        number = 10

        conds = [(numpy.random.rand(dim, dim) < 0.5).astype(numpy.bool_)
                 for _ in range(repeat)]
        xs = [numpy.random.rand(dim, dim).astype(numpy.float32)
              for _ in range(repeat)]
        ys = [numpy.random.rand(dim, dim).astype(numpy.float32)
              for _ in range(repeat)]

        # numpy
        ctx = dict(conds=conds, xs=xs, ys=ys, where=numpy.where,
                   loop_where=loop_where)
        obs = measure_time(
            "loop_where(where, conds, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy.where'
        res.append(obs)

        # numpy add
        ctx['where'] = numpy_where_add
        obs = measure_time(
            "loop_where(where, conds, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy_where_add'
        res.append(obs)

        # onnxruntime
        ctx['where'] = ort_where
        obs = measure_time(
            "loop_where(where, conds, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort_where'
        res.append(obs)

        # onnxruntime - 2
        ctx['where'] = ort_where_add
        ctx['conds'] = [c.astype(numpy.float32) for c in conds]
        obs = measure_time(
            "loop_where(where, conds, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort_where_add'
        res.append(obs)

        if tf_where is not None:
            # tensorflow
            ctx['where'] = tf_where
            ctx['conds'] = [convert_to_tensor(c) for c in conds]
            ctx['xs'] = [convert_to_tensor(x) for x in xs]
            ctx['ys'] = [convert_to_tensor(y) for y in ys]
            obs = measure_time(
                "loop_where(where, conds, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'tf_where'
            res.append(obs)

        if torch_where is not None:
            # torch
            ctx['where'] = torch_where
            ctx['conds'] = [from_numpy(c) for c in conds]
            ctx['xs'] = [from_numpy(x) for x in xs]
            ctx['ys'] = [from_numpy(y) for y in ys]
            obs = measure_time(
                "loop_where(where, conds, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'torch_where'
            res.append(obs)

    # Dataframes
    df = pandas.DataFrame(res)
    piv = df.pivot('dim', 'fct', 'average')

    rs = piv.copy()
    rs['ort_where'] = rs['numpy.where'] / rs['ort_where']
    rs['numpy_where_add'] = rs['numpy.where'] / rs['numpy_where_add']
    rs['ort_where_add'] = rs['numpy.where'] / rs['ort_where_add']
    if 'tf_where' in rs.columns:
        rs['tf_where'] = rs['numpy.where'] / rs['tf_where']
    if 'torch_where' in rs.columns:
        rs['torch_where'] = rs['numpy.where'] / rs['torch_where']
    rs['numpy.where'] = 1.

    # Graphs.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(logx=True, logy=True, ax=ax[0],
             title="Where benchmark -- (N, N)\nlower better")
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="Where Speedup, baseline=numpy -- (N, N)\nhigher better")
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})

    return df, rs, ax


############
# Benchmark
# +++++++++

df, piv, ax = benchmark_equation()
df.pivot("fct", "dim", "average")
dfs = [df]

####################################
# Conclusion
# ++++++++++
#
# The implementation of Where should be faster
# than the formula `where(c, x, y) = x * c - y * (c - 1)`.

merged = pandas.concat(dfs)
name = "where"
merged.to_csv("plot_%s.csv" % name, index=False)
merged.to_excel("plot_%s.xlsx" % name, index=False)
plt.savefig("plot_%s.png" % name)

plt.show()
