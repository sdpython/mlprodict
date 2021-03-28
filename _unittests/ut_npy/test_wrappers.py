# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from collections import OrderedDict
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.linear_model import LinearRegression
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common._topology import Variable
from mlprodict.npy.onnx_sklearn_wrapper import (
    _common_shape_calculator_t, _common_shape_calculator_int_t,
    _common_converter_t, _common_converter_int_t)
from mlprodict.npy.onnx_numpy_annotation import (
    NDArrayType, NDArrayTypeSameShape,
    NDArraySameTypeSameShape)


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
        vin = Variable('X', 'X', type=FloatTensorType(
            [None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=FloatTensorType(
            [None, None]), scope=None)
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
        vin = Variable('X', 'X', type=FloatTensorType(
            [None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=Int64TensorType(
            [None, None]), scope=None)
        vout = Variable('Y', 'Y', type=FloatTensorType([None]), scope=None)
        vout2 = Variable('Y2', 'Y2', type=FloatTensorType([None]), scope=None)
        vout3 = Variable('Y3', 'Y3', type=FloatTensorType([None]), scope=None)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2, vout3])
        self.assertRaise(lambda: _common_shape_calculator_int_t(op),
                         AttributeError)
        op.onnx_numpy_fct_ = None
        self.assertRaise(
            lambda: _common_shape_calculator_int_t(op), RuntimeError)
        op = operator_dummy(model, inputs=[vin, vin2], outputs=[vout])
        op.onnx_numpy_fct_ = None
        self.assertRaise(
            lambda: _common_shape_calculator_int_t(op), RuntimeError)
        op = operator_dummy(model, inputs=[vin], outputs=[vout, vout2])
        op.onnx_numpy_fct_ = None
        _common_shape_calculator_int_t(op)

    def test_convert_calculator(self):
        model = LinearRegression()
        model.fit(numpy.random.randn(10, 2), numpy.random.randn(10))
        vin = Variable('X', 'X', type=FloatTensorType(
            [None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=FloatTensorType(
            [None, None]), scope=None)
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
        vin = Variable('X', 'X', type=FloatTensorType(
            [None, None]), scope=None)
        vin2 = Variable('X2', 'X2', type=FloatTensorType(
            [None, None]), scope=None)
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

    def test_signature(self):
        # sig, args, kwargs, version
        f32 = numpy.float32
        i64 = numpy.int64
        bbb = numpy.bool
        sigs = [
            # 0
            (NDArraySameTypeSameShape("all"), ['X'], {}, (f32, )),
            (NDArraySameTypeSameShape("floats"), ['X'], {}, (f32, )),
            (NDArrayType("all_int"), ['X'], {}, (i64, )),
            (NDArrayType(("bool", "T:all"), dtypes_out=('T',)), ['X', 'C'], {},
             (bbb, f32)),
            (NDArrayType("all", nvars=True), ['X', 'Y'], {}, (f32, f32)),
            # 5
            (NDArrayType("all", nvars=True), [
             'X', 'Y', 'Z'], {}, (f32, f32, f32)),
            (NDArrayType(("all", "ints")), ['X', 'I'], {}, (f32, i64)),
            # 7
            (NDArrayType("T:all", "T"), ['X'], {}, (f32, )),
            (NDArrayTypeSameShape("all_bool"), ['B'], {}, (bbb, )),
            (NDArrayType(("T:all", "ints"), ("T", (numpy.int64,))),
             ['X', 'I'], {}, (f32, i64)),
        ]
        expected = [
            # 0
            ("[('X', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]"),
            ("[('X', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]"),
            ("[('X', Int64TensorType(shape=[]))]",
             "[('y', Int64TensorType(shape=[]))]"),
            ("[('X', BooleanTensorType(shape=[])), ('C', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]"),
            ("[('x0', FloatTensorType(shape=[])), ('x1', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]"),
            # 5
            (("[('x0', FloatTensorType(shape=[])), ('x1', FloatTensorType(shape=[])), "
              "('x2', FloatTensorType(shape=[]))]"), "[('y', FloatTensorType(shape=[]))]"),
            ("[('X', FloatTensorType(shape=[])), ('I', Int64TensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]"),
            # 7
            ("[('X', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]"),
            ("[('B', BooleanTensorType(shape=[]))]",
             "[('y', BooleanTensorType(shape=[]))]"),
            ("[('X', FloatTensorType(shape=[])), ('I', Int64TensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[])), ('z', Int64TensorType(shape=[]))]"),
        ]
        self.assertEqual(len(expected), len(sigs))
        for i, (sigt, expe) in enumerate(zip(sigs, expected)):
            sig, args, kwargs, version = sigt
            inputs, kwargs, outputs, optional = sig.get_inputs_outputs(
                args, kwargs, version)
            self.assertEqual(optional, 0)
            self.assertIsInstance(kwargs, (OrderedDict, dict))
            si, so = expe
            self.assertEqual(si, str(inputs))
            self.assertEqual(so, str(outputs))

    def test_signature_optional1(self):
        # sig, args, kwargs, version
        f32 = numpy.float32
        i64 = numpy.int64
        bbb = numpy.bool
        sigs = [
            # 0
            (NDArrayType(("all", numpy.int64), n_optional=1),
             ['X', 'I'], {}, (f32, i64)),
            (NDArrayType(("all", numpy.int64),
                         n_optional=1), ['X'], {}, (f32, )),
        ]
        expected = [
            ("[('X', FloatTensorType(shape=[])), ('I', Int64TensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 1),
            ("[('X', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 0),
        ]
        self.assertEqual(len(expected), len(sigs))
        for i, (sigt, expe) in enumerate(zip(sigs, expected)):
            sig, args, kwargs, version = sigt
            inputs, kwargs, outputs, optional = sig.get_inputs_outputs(
                args, kwargs, version)
            self.assertIsInstance(kwargs, (OrderedDict, dict))
            si, so, opt = expe
            self.assertEqual(optional, opt)
            self.assertEqual(si, str(inputs))
            self.assertEqual(so, str(outputs))

    def test_signature_optional2(self):
        # sig, args, kwargs, version
        f32 = numpy.float32
        i64 = numpy.int64
        bbb = numpy.bool
        sigs = [
            # 2: optional
            (NDArrayType(("all", "all", "all"), n_optional=2),
             ['X', 'Y', 'Z'], {}, (f32, f32, f32)),
            (NDArrayType(("all", "all", "all"), n_optional=2),
             ['X', 'Y'], {}, (f32, f32)),
            (NDArrayType(("all", "all", "all"), n_optional=2),
             ['X'], {}, (f32, f32)),
        ]
        expected = [
            ("[('X', FloatTensorType(shape=[])), ('Y', FloatTensorType(shape=[])), "
             "('Z', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 2),
            ("[('X', FloatTensorType(shape=[])), ('Y', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 1),
            ("[('X', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 0),
        ]
        self.assertEqual(len(expected), len(sigs))
        for i, (sigt, expe) in enumerate(zip(sigs, expected)):
            sig, args, kwargs, version = sigt
            inputs, kwargs, outputs, optional = sig.get_inputs_outputs(
                args, kwargs, version)
            self.assertIsInstance(kwargs, (OrderedDict, dict))
            si, so, opt = expe
            self.assertEqual(optional, opt)
            self.assertEqual(si, str(inputs))
            self.assertEqual(so, str(outputs))

    def test_signature_optional3_kwargs(self):
        # sig, args, kwargs, version
        f32 = numpy.float32
        i64 = numpy.int64
        bbb = numpy.bool
        sigs = [
            (NDArrayType(("T:all", numpy.int64, 'T'), n_optional=1),
             ['X', 'I'], {'mode': 'constant'}, (f32, i64, 'constant')),
            (NDArrayType(("T:all", numpy.int64, 'T'), n_optional=1),
             ['X', 'I', 'Y'], {}, (f32, i64, f32)),
            (NDArrayType(("T:all", numpy.int64, 'T'), n_optional=1),
             ['X', 'I', 'Y'], {'mode': 'constant'}, (f32, i64, f32, 'constant')),
        ]
        expected = [
            ("[('X', FloatTensorType(shape=[])), ('I', Int64TensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 0),
            ("[('X', FloatTensorType(shape=[])), ('I', Int64TensorType(shape=[])), "
             "('Y', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 1),
            ("[('X', FloatTensorType(shape=[])), ('I', Int64TensorType(shape=[])), "
             "('Y', FloatTensorType(shape=[]))]",
             "[('y', FloatTensorType(shape=[]))]", 1),
        ]
        self.assertEqual(len(expected), len(sigs))
        for i, (sigt, expe) in enumerate(zip(sigs, expected)):
            sig, args, kwargs, version = sigt
            inputs, kwargs, outputs, optional = sig.get_inputs_outputs(
                args, kwargs, version)
            self.assertIsInstance(kwargs, (OrderedDict, dict))
            si, so, opt = expe
            self.assertEqual(optional, opt)
            self.assertEqual(si, str(inputs))
            self.assertEqual(so, str(outputs))

    def test_signature_optional_errors(self):
        # sig, args, kwargs, version
        f32 = numpy.float32
        i64 = numpy.int64
        bbb = numpy.bool
        sigs = [
            (NDArrayType(("T:all", numpy.int64, 'T'), n_optional=1),
             ['X', 'I', 'R', 'T'], {'mode': 'constant'}, (f32, i64, 'constant')),
            (NDArrayType(("T:all", numpy.int64), 'T', n_optional=1),
             ['X', 'I', 'Y'], {}, (f32, i64, f32)),
            (NDArrayType(("T:all", numpy.int64), 'T', n_optional=1),
             ['X', 'I', 'Y'], {}, (f32, i64, f32, f32)),
        ]
        for i, sigt in enumerate(sigs):
            sig, args, kwargs, version = sigt
            self.assertRaise(
                lambda: sig.get_inputs_outputs(args, kwargs, version),
                RuntimeError)


if __name__ == "__main__":
    TestWrappers().test_signature_optional_errors()
    unittest.main()
