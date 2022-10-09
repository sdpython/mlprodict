"""
@brief      test tree node (time=7s)
"""
from typing import Any
import unittest
import numpy as np
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from mlprodict.onnx_conv import to_onnx
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.npy import onnxnumpy_default, NDArray
from mlprodict.testing.test_utils import dump_data_and_model
import mlprodict.npy.numpy_onnx_impl as npnx


class TestSklearnNewConverter(ExtTestCase):

    @ignore_warnings(UserWarning)
    def test_transformed_target_regressor(self):

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
        onx = to_onnx(model, x, rewrite_ops=True, target_opset=TARGET_OPSET)

        dump_data_and_model(
            x.astype(np.float32), model, onx,
            basename="TransformedTargetRegressor")


if __name__ == "__main__":
    unittest.main()
