"""
.. _l-b-transpose:

Compares implementations of Tranpose
====================================

This example compares the :epkg:`numpy:transpose` from numpy,
to :epkg:`onnxruntime` implementation.
If available, :epkg:`tensorflow` and :epkg:`pytorch` are included as well.

.. contents::
    :local:

Available optimisation
++++++++++++++++++++++

The code shows which parallelisation optimisation could be used,
*AVX* or *SSE* and the number of available processors.
Both :epkg:`numpy` and :epkg:`torch` have lazy implementations,
the function switches dimensions and strides but does not move
any data. That's why function *contiguous* was called in both cases.
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxTranspose
from cpyquickhelper.numbers import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())

###################################
# Transpose implementations
# +++++++++++++++++++++++++
#
# Function einsum is used from tensorflow and pytorch
# instead of transpose. The equation reflects the required
# transposition.

try:
    from tensorflow import transpose as tf_transpose, convert_to_tensor
except ImportError:
    tf_transpose = None
try:
    from torch import einsum as torch_einsum, from_numpy
except ImportError:
    torch_einsum = None


def build_ort_transpose(perm, op_version=12):
    node = OnnxTranspose('x', perm=perm, op_version=op_version,
                         output_names=['z'])
    onx = node.to_onnx(inputs=[('x', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x})


def loop_fct(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


def perm2eq(perm):
    first = "".join(chr(97 + i) for i in range(len(perm)))
    second = "".join(first[p] for p in perm)
    return "%s->%s" % (first, second)


def benchmark_op(perm, repeat=5, number=5, name="Transpose", shape_fct=None):
    if shape_fct is None:
        def shape_fct(dim): return (3, dim, 1, 512)
    ort_fct = build_ort_transpose(perm)
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 400, 512, 1024]):
        shape = shape_fct(dim)
        n_arrays = 10 if dim < 512 else 4
        xs = [numpy.random.rand(*shape).astype(numpy.float32)
              for _ in range(n_arrays)]
        ys = [perm for _ in range(n_arrays)]
        equation = perm2eq(perm)
        info = dict(perm=perm, shape=shape)

        # numpy
        ctx = dict(
            xs=xs, ys=ys,
            fct=lambda x, y: numpy.ascontiguousarray(numpy.transpose(x, y)),
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

        if tf_transpose is not None:
            # tensorflow
            ctx['fct'] = tf_transpose
            ctx['xs'] = [convert_to_tensor(x) for x in xs]
            ctx['ys'] = [convert_to_tensor(y) for y in ys]
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'tf'
            obs.update(info)
            res.append(obs)

            # tensorflow with copy
            ctx['fct'] = lambda x, y: tf_transpose(
                convert_to_tensor(x)).numpy()
            ctx['xs'] = xs
            ctx['ys'] = ys
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'tf_copy'
            obs.update(info)
            res.append(obs)

        if torch_einsum is not None:
            # torch
            ctx['fct'] = lambda x, y: torch_einsum(equation, x).contiguous()
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
             title="%s benchmark\n%r - %r - %s"
                   " lower better" % (name, shape_name, perm, equation))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%r - %r - %s"
                  " higher better" % (name, shape_name, perm, equation))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})
    return df, rs, ax


dfs = []

###################################
# First permutation: (1, 0, 2, 3)
# +++++++++++++++++++++++++++++++

perm = (1, 0, 2, 3)
df, piv, ax = benchmark_op(perm)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Second permutation: (0, 1, 3, 2)
# ++++++++++++++++++++++++++++++++

perm = (1, 0, 3, 2)
df, piv, ax = benchmark_op(perm)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Third permutation: (0, 2, 1, 3)
# ++++++++++++++++++++++++++++++++
#
# This transposition is equivalent to a reshape
# because it only moves the empty axis.
# The comparison is entirely fair as the cost
# for onnxruntime includes a copy from numpy to
# onnxruntime, a reshape = another copy, than a copy
# back to numpy. Tensorflow and pytorch seems
# to have a lazy implementation in this case.

perm = (0, 2, 1, 3)
df, piv, ax = benchmark_op(perm)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Fourth permutation: (3, 1, 2, 0)
# ++++++++++++++++++++++++++++++++

perm = (3, 1, 2, 0)
df, piv, ax = benchmark_op(perm)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Fifth permutation: (1, 2, 3, 0)
# +++++++++++++++++++++++++++++++

perm = (1, 2, 3, 0)
df, piv, ax = benchmark_op(perm)
dfs.append(df)
df.pivot("fct", "N", "average")

###################################
# Six th permutation: (1, 2, 4, 3, 0)
# +++++++++++++++++++++++++++++++++++

perm = (1, 2, 4, 3, 0)
df, piv, ax = benchmark_op(perm, shape_fct=lambda dim: (3, dim, 1, 8, 512))
dfs.append(df)
df.pivot("fct", "N", "average")

####################################
# Conclusion
# ++++++++++
#
# All libraries have similar implementations.
# :epkg:`onnxruntime` measures includes 2 mores copies,
# one to copy from numpy container to onnxruntime container,
# another one to copy back from onnxruntime container to numpy.
# Parallelisation should be investigated.

merged = pandas.concat(dfs)
name = "transpose"
merged.to_csv("plot_%s.csv" % name, index=False)
merged.to_excel("plot_%s.xlsx" % name, index=False)
plt.savefig("plot_%s.png" % name)

plt.show()
