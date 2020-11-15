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
from mlprodict.testing.experimental_c import custom_einsum_double
from onnxruntime import InferenceSession
from skl2onnx.algebra.onnx_ops import OnnxEinsum
from skl2onnx.common.data_types import DoubleTensorType
import onnx


def build_ort_einsum(equation, op_version=12):
    node = OnnxEinsum('x', 'y', equation=equation,
                      op_version=op_version,
                      output_names=['z'])
    onx = node.to_onnx(inputs=[('x', DoubleTensorType()), ('y', DoubleTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x, 'y': y})


equation = "bsnh,btnh->bnts"
ort_einsum = build_ort_einsum(equation)
res = []
for dim in tqdm([8, 16, 32, 64, 128, 256, 512]):
    x = numpy.random.rand(1, dim, 12, 64)
    y = numpy.random.rand(1, dim, 12, 64)

    ort_einsum(x, y)

    ctx = dict(equation=equation, x=x, y=y, einsum=numpy.einsum)
    obs = measure_time("einsum(equation, x, y)", div_by_number=True, context=ctx,
                       repeat=5, number=5)
    obs['dim'] = dim
    obs['fct'] = 'numpy.einsum'
    res.append(obs)

    ctx['einsum'] = ort_einsum
    obs = measure_time("einsum(x, y)", div_by_number=True, context=ctx,
                       repeat=5, number=5)
    obs['dim'] = dim
    obs['fct'] = 'ort_einsum'
    res.append(obs)

    ctx['einsum'] = custom_einsum_double
    obs = measure_time("einsum(equation, x, y)", div_by_number=True, context=ctx,
                       repeat=5, number=5)
    obs['dim'] = dim
    obs['fct'] = 'custom_einsum_double'
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
rs['custom_einsum_double'] = rs['numpy.einsum'] / rs['custom_einsum_double']
rs['ort_einsum'] = rs['numpy.einsum'] / rs['ort_einsum']
rs['numpy.einsum'] = 1.
rs

###########################################
# Graphs.
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
piv.plot(logx=True, logy=True, ax=ax[0], title="Einsum benchmark")
rs.plot(logx=True, ax=ax[1], title="Einsum Speedup, baseline=numpy")
plt.show()
