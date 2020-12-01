"""
.. _l-b-transpose:

Compares implementations of Tranpose
====================================

The following function benchmark different implementation
of function :epkg:`numpy:transpose`.
It compares *numpy* implementation to
:epkg:`onnxruntime` implementation.
If available, :epkg:`tensorflow` and :epkg:`pytorch` are included as well.

.. contents::
    :local:

Available optimisation
++++++++++++++++++++++

The code shows which parallelisation optimisation could be used,
*AVX* or *SSE* and the number of available processors.
"""
from mlprodict.testing.experimental_c import code_optimisation
print(code_optimisation())

###################################
# Transpose implementation
# ++++++++++++++++++++++++

import numpy
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from cpyquickhelper.numbers.speed_measure import measure_time
from onnxruntime import InferenceSession
from skl2onnx.algebra.onnx_ops import OnnxTranspose
from skl2onnx.common.data_types import FloatTensorType
import onnx
try:
    from tensorflow import transpose as tf_transpose, convert_to_tensor
    from tensorflow import einsum as tf_einsum
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


def benchmark_op(perm, repeat=5, number=1, name="transpose", shape_fct=None):
    if shape_fct is None:
        shape_fct = lambda dim: (dim, dim, 1, dim)
    ort_fct = build_ort_transpose(perm)
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 500, 512]):
        shape = shape_fct(dim)
        xs = [numpy.random.rand(*shape).astype(numpy.float32)
              for _ in range(5)]
        ys = [perm for _ in range(5)]
        equation = perm2eq(perm)

        # numpy
        ctx = dict(xs=xs, ys=ys, fct=numpy.transpose, loop_fct=loop_fct)
        obs = measure_time(
            "loop_fct(fct, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy'
        res.append(obs)

        # onnxruntime
        ctx['fct'] = ort_fct
        obs = measure_time(
            "loop_fct(fct, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort'
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
            res.append(obs)

        if torch_einsum is not None:
            # torch
            ctx['fct'] = lambda x, y: torch_einsum(equation, x)
            ctx['xs'] = [from_numpy(x) for x in xs]
            ctx['ys'] = ys  # [from_numpy(y) for y in ys]
            obs = measure_time(
                "loop_fct(fct, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'torch'
            res.append(obs)

    # Dataframes
    shape_name = str(shape).replace(str(dim), "N")
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
             title="%s benchmark\n%r - %r - %s"
                   "\nlower better" % (name, shape_name, perm, equation))
    ax[0].legend(prop={"size": 6})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%r - %r - %s"
                  "\nhigher better" % (name, shape_name, perm, equation))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 6})

    return df, piv, ax


###################################
# First permutation: (1, 0, 2, 3)
# +++++++++++++++++++++++++++++++

perm = (1, 0, 2, 3)
df, piv, ax = benchmark_op(perm)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# Second permutation: (0, 1, 3, 2)
# ++++++++++++++++++++++++++++++++

perm = (1, 0, 3, 2)
df, piv, ax = benchmark_op(perm)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# Third permutation: (0, 2, 1, 3)
# ++++++++++++++++++++++++++++++++

perm = (0, 2, 1, 3)
df, piv, ax = benchmark_op(perm)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# Fourth permutation: (3, 1, 2, 0)
# ++++++++++++++++++++++++++++++++

perm = (3, 1, 2, 0)
df, piv, ax = benchmark_op(perm)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


###################################
# Fifth permutation: (1, 2, 3, 0)
# +++++++++++++++++++++++++++++++

perm = (1, 2, 3, 0)
df, piv, ax = benchmark_op(perm)
df.pivot("fct", "N", "average")

####################################
# Ratios
piv.T


####################################
# Conclusion
# ++++++++++
#
# :epkg:`pytorch` implementation is much faster.
# Look at the signature of function `transpose
# <https://pytorch.org/docs/stable/generated/torch.transpose.html>`_,
# it appears that the function is optimized when only two dimensions
# are permuted.

plt.show()
