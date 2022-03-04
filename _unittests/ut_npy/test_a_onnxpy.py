# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase
from onnxruntime.capi._pybind_state import InvalidArgument as OrtInvalidArgument
from mlprodict.npy import OnnxNumpyCompiler as ONC, NDArray
from mlprodict.npy.onnx_variable import OnnxVar
from mlprodict.npy.onnx_numpy_annotation import _NDArrayAlias
from mlprodict.npy.onnx_numpy_wrapper import wrapper_onnxnumpy_np
from mlprodict.npy.xop_variable import Variable
from mlprodict.npy.xop import loadop


class TestOnnxPy(ExtTestCase):

    @staticmethod
    def onnx_abs(x: NDArray[Any, numpy.float32],
                 op_version=None) -> NDArray[Any, numpy.float32]:
        OnnxAbs = loadop('Abs')
        return OnnxAbs(x, op_version=op_version)

    @staticmethod
    def onnx_abs_shape(x: NDArray[(Any, Any), numpy.float32],
                       op_version=None) -> NDArray[(Any, Any), numpy.float32]:
        OnnxAbs = loadop('Abs')
        return OnnxAbs(x, op_version=op_version)

    def test_onnx_var(self):
        OnnxAbs = loadop('Abs')
        ov = OnnxVar('X')
        rp = repr(ov)
        self.assertEqual("OnnxVar('X')", rp)
        ov = OnnxVar('X', op=OnnxAbs)
        rp = repr(ov)
        self.assertEqual("OnnxVar('X', op=OnnxAbs)", rp)
        ov = OnnxVar('X', op='filter')
        rp = repr(ov)
        self.assertEqual("OnnxVar('X', op='filter')", rp)

    def test_process_dtype(self):
        for name in ['all', 'int', 'ints', 'floats', 'T']:
            res = _NDArrayAlias._process_type(  # pylint: disable=W0212
                name, {'T': 0}, 1)
            if name == 'all':
                self.assertIsInstance(res, tuple)
            if name == 'int':
                self.assertEqual(res, (numpy.int64, ))
            if name == 'ints':
                self.assertEqual(res, (numpy.int32, numpy.int64))
            if name == 'floats':
                self.assertEqual(res, (numpy.float32, numpy.float64))
            if name == 'T':
                self.assertEqual(res, ('T', ))
            s = repr(res)
            self.assertNotEmpty(s)
            self.assertIn('(', s)
            self.assertEndsWith(')', s)

    def test_annotation(self):
        cl = ONC(TestOnnxPy.onnx_abs, op_version=12)
        ann = cl._parse_annotation(None, None)  # pylint: disable=W0212
        inputs, outputs, _, __, ___ = ann
        self.assertIsInstance(inputs, list)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(inputs[0], Variable)
        self.assertIsInstance(outputs[0], Variable)
        self.assertEqual(inputs[0].name, 'x')
        self.assertEqual(outputs[0].name, 'y')
        self.assertEqual(inputs[0].dtype, numpy.float32)
        self.assertEqual(outputs[0].dtype, numpy.float32)

    def test_annotation_shape(self):
        cl = ONC(TestOnnxPy.onnx_abs_shape, op_version=12)
        ann = cl._parse_annotation(None, None)  # pylint: disable=W0212
        inputs, outputs, _, __, ___ = ann
        self.assertIsInstance(inputs, list)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(inputs[0], Variable)
        self.assertIsInstance(outputs[0], Variable)
        self.assertEqual(inputs[0].name, 'x')
        self.assertEqual(outputs[0].name, 'y')
        self.assertEqual(inputs[0].dtype, numpy.float32)
        self.assertEqual(outputs[0].dtype, numpy.float32)

    def test_wrong_runtime(self):
        self.assertRaise(
            lambda: ONC(TestOnnxPy.onnx_abs, op_version=12,
                        runtime="esoterique"),
            ValueError)

    def test_runtime(self):
        for rt in ['python', 'onnxruntime1', 'onnxruntime']:
            with self.subTest(rt=rt):
                cl = ONC(TestOnnxPy.onnx_abs, op_version=12, runtime=rt)
                x = numpy.array([[0.45, -0.56], [0.455, -0.565]],
                                dtype=numpy.float32)
                y = cl(x)
                self.assertEqualArray(numpy.abs(x), y)
                if rt == 'onnxruntime':
                    self.assertRaise(
                        lambda: cl(x.astype(numpy.float64)  # pylint: disable=W0640
                                   ),  # pylint: disable=W0640
                        (TypeError, OrtInvalidArgument))

    def test_wrapper_onnxnumpy_np(self):

        def fct(x):
            return x

        nb = wrapper_onnxnumpy_np(fct=fct, signature=None, fctsig=None)
        state = nb.__getstate__()
        nb.__setstate__(state)


if __name__ == "__main__":
    unittest.main()
