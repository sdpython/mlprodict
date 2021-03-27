# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.linear_model import LinearRegression
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common._topology import Variable
from mlprodict.npy.onnx_sklearn_wrapper import (
    _common_shape_calculator_t, _common_shape_calculator_int_t,
    _common_converter_t, _common_converter_int_t)


class operator_dummy:
    
    def __init__(self, operator, inputs, outputs):
        self.raw_operator = operator
        self.inputs = inputs
        self.outputs = outputs


class container_dummy:
    def __init__(self):
        self.target_opset = 14


class TestWrappers(ExtTestCase):

    def test_shape_calculator(self):
        model = LinearRegression()
        vin = Variable('X', 'X', type=FloatTensorType([None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=FloatTensorType([None, None]), scope=None)
        vout = Variable('Y', 'Y', type=FloatTensorType([None]), scope=None)
        vout2 = Variable('Y2', 'Y2', type=FloatTensorType([None]), scope=None)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2])
        self.assertRaise(lambda: _common_shape_calculator_t(op),
                         AttributeError)
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_shape_calculator_t(op), RuntimeError)
        op = operator_dummy(model, inputs=[vin, vin2], outputs=[vout])
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_shape_calculator_t(op), RuntimeError)
        op = operator_dummy(model, inputs=[vin], outputs=[vout])
        op.onnx_numpy_fct_ = None
        _common_shape_calculator_t(op)
        
    def test_shape_calculator_int(self):
        model = LinearRegression()
        vin = Variable('X', 'X', type=FloatTensorType([None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=Int64TensorType([None, None]), scope=None)
        vout = Variable('Y', 'Y', type=FloatTensorType([None]), scope=None)
        vout2 = Variable('Y2', 'Y2', type=FloatTensorType([None]), scope=None)
        vout3 = Variable('Y3', 'Y3', type=FloatTensorType([None]), scope=None)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2, vout3])
        self.assertRaise(lambda: _common_shape_calculator_int_t(op),
                         AttributeError)
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_shape_calculator_int_t(op), RuntimeError)
        op = operator_dummy(model, inputs=[vin, vin2], outputs=[vout])
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_shape_calculator_int_t(op), RuntimeError)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2])
        op.onnx_numpy_fct_ = None
        _common_shape_calculator_int_t(op)
        
    def test_convert_calculator(self):
        model = LinearRegression()
        model.fit(numpy.random.randn(10, 2), numpy.random.randn(10))
        vin = Variable('X', 'X', type=FloatTensorType([None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=FloatTensorType([None, None]), scope=None)
        vout = Variable('Y', 'Y', type=FloatTensorType([None]), scope=None)
        vout2 = Variable('Y2', 'Y2', type=FloatTensorType([None]), scope=None)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2])
        scope = None
        container = container_dummy()
        self.assertRaise(lambda: _common_converter_t(scope, op, container),
                         AttributeError)
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_converter_t(scope, op, container),
                         RuntimeError)
        op = operator_dummy(model, inputs=[vin, vin2], outputs=[vout])
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_converter_t(scope, op, container),
                         RuntimeError)
        op = operator_dummy(model, inputs=[vin], outputs=[vout])
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_converter_t(scope, op, container),
                         AttributeError)
        
    def test_convert_calculator_int(self):
        model = LinearRegression()
        model.fit(numpy.random.randn(10, 2), numpy.random.randn(10))
        vin = Variable('X', 'X', type=FloatTensorType([None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=FloatTensorType([None, None]), scope=None)
        vout = Variable('Y', 'Y', type=FloatTensorType([None]), scope=None)
        vout2 = Variable('Y2', 'Y2', type=Int64TensorType([None]), scope=None)
        vout3 = Variable('Y2', 'Y2', type=FloatTensorType([None]), scope=None)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2, vout3])
        scope = None
        container = container_dummy()
        self.assertRaise(lambda: _common_converter_int_t(scope, op, container),
                         AttributeError)
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_converter_int_t(scope, op, container),
                         RuntimeError)
        op = operator_dummy(model, inputs=[vin, vin2], outputs=[vout])
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_converter_int_t(scope, op, container),
                         RuntimeError)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2])
        op.onnx_numpy_fct_ = None
        self.assertRaise(lambda: _common_converter_int_t(scope, op, container),
                         AttributeError)
        


if __name__ == "__main__":
    unittest.main()
