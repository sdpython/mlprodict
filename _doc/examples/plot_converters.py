"""
.. _l-b-transformed-target-regressor:

A converter for a TransformedTargetRegressor
============================================

There is no easy way to convert a
:class:`sklearn.preprocessing.FunctionTransformer` or
a :epkg:`sklearn.compose.TransformedTargetRegressor` unless
the function is written in such a way the conversion is implicit.

"""
from typing import Any
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from mlprodict.onnx_conv import to_onnx
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.npy import onnxnumpy_default, NDArray
from mlprodict.onnxrt import OnnxInference
import mlprodict.npy.numpy_onnx_impl as npnx

########################################
# TransformedTargetRegressor
# ++++++++++++++++++++++++++


@onnxnumpy_default
def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
    return npnx.log1p(x)


@onnxnumpy_default
def onnx_exp_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
    return npnx.exp(x) - np.float32(1)


model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=onnx_log_1, inverse_func=onnx_exp_1)

x = np.arange(18).reshape((-1, 3)).astype(np.float32)
y = x.sum(axis=1)
model.fit(x, y)
expected = model.predict(x)
print(expected)

#####################################
# Conversion to ONNX

onx = to_onnx(model, x, rewrite_ops=True, target_opset=TARGET_OPSET)
oinf = OnnxInference(onx)
got = oinf.run({'X': x})
print(got)

###################################
# FunctionTransformer
# +++++++++++++++++++

model = FunctionTransformer(onnx_log_1)
model.fit(x, y)
expected = model.transform(x)
print(expected)

#####################################
# Conversion to ONNX

onx = to_onnx(model, x, rewrite_ops=True, target_opset=TARGET_OPSET)
oinf = OnnxInference(onx)
got = oinf.run({'X': x})
print(got)
