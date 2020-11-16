"""
.. _l-example-experimental:

Compares implementation of stanard function
===========================================

The following function benchmark different implementation
of standard function.



.. contents::
    :local:

Einsum
++++++
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.testing.experimental_c import custom_einsum_float
from onnxruntime import InferenceSession
from skl2onnx.algebra.onnx_ops import OnnxEinsum
from skl2onnx.common.data_types import FloatTensorType
import onnx
try:
    from tensorflow import einsum as tf_einsum, convert_to_tensor
except ImportError:
    tf_einsum = None
try:
    from torch import einsum as torch_einsum, from_numpy
except ImportError:
    torch_einsum = None


def build_ort_einsum(equation, op_version=12):
    node = OnnxEinsum('x', 'y', equation=equation,
                      op_version=op_version,
                      output_names=['z'])
    onx = node.to_onnx(inputs=[('x', FloatTensorType()),
                               ('y', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x, 'y': y})


def loop_einsum_eq(fct, equation, xs, ys):
    for x, y in zip(xs, ys):
        fct(equation, x, y)


def loop_einsum(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


equation = "bsnh,btnh->bnts"
ort_einsum = build_ort_einsum(equation)
res = []
for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                 256, 500, 512]):
    xs = [numpy.random.rand(1, dim, 12, 64).astype(numpy.float32)
          for _ in range(5)]
    ys = [numpy.random.rand(1, dim, 12, 64).astype(numpy.float32)
          for _ in range(5)]

    ctx = dict(equation=equation, xs=xs, ys=ys, einsum=numpy.einsum,
               loop_einsum=loop_einsum, loop_einsum_eq=loop_einsum_eq)
    obs = measure_time("loop_einsum_eq(einsum, equation, xs, ys)",
                       div_by_number=True, context=ctx,
                       repeat=5, number=1)
    obs['dim'] = dim
    obs['fct'] = 'numpy.einsum'
    res.append(obs)

    ctx['einsum'] = ort_einsum
    obs = measure_time("loop_einsum(einsum, xs, ys)",
                       div_by_number=True, context=ctx,
                       repeat=5, number=1)
    obs['dim'] = dim
    obs['fct'] = 'ort_einsum'
    res.append(obs)

    ctx['einsum'] = custom_einsum_float
    obs = measure_time("loop_einsum_eq(einsum, equation, xs, ys)",
                       div_by_number=True, context=ctx,
                       repeat=5, number=1)
    obs['dim'] = dim
    obs['fct'] = 'custom_einsum_float'
    res.append(obs)

    if tf_einsum is not None:
        ctx['einsum'] = tf_einsum
        ctx['xs'] = [convert_to_tensor(x) for x in xs]
        ctx['ys'] = [convert_to_tensor(y) for y in ys]
        obs = measure_time("loop_einsum_eq(einsum, equation, xs, ys)",
                           div_by_number=True, context=ctx,
                           repeat=5, number=1)
        obs['dim'] = dim
        obs['fct'] = 'tf_einsum'
        res.append(obs)

    if torch_einsum is not None:
        ctx['einsum'] = torch_einsum
        ctx['xs'] = [from_numpy(x) for x in xs]
        ctx['ys'] = [from_numpy(y) for y in ys]
        obs = measure_time("loop_einsum_eq(einsum, equation, xs, ys)",
                           div_by_number=True, context=ctx,
                           repeat=5, number=1)
        obs['dim'] = dim
        obs['fct'] = 'torch_einsum'
        res.append(obs)

df = pandas.DataFrame(res)
df
print(df.T)

###########################################
# Pivot

piv = df.pivot('dim', 'fct', 'average')
piv

###########################################
# Ratios

rs = piv.copy()
rs['custom_einsum_float'] = rs['numpy.einsum'] / rs['custom_einsum_float']
rs['ort_einsum'] = rs['numpy.einsum'] / rs['ort_einsum']
if 'tf_einsum' in rs.columns:
    rs['tf_einsum'] = rs['numpy.einsum'] / rs['tf_einsum']
if 'torch_einsum' in rs.columns:
    rs['torch_einsum'] = rs['numpy.einsum'] / rs['torch_einsum']
rs['numpy.einsum'] = 1.
rs

###########################################
# Graphs.
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
piv.plot(logx=True, logy=True, ax=ax[0],
         title="Einsum benchmark\n%s -- (1, N, 12, 64)" % equation)
rs.plot(logx=True, logy=True, ax=ax[1],
        title="Einsum Speedup, baseline=numpy\n%s -- (1, N, 12, 64)" % equation)
ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
plt.show()
