"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
from collections import OrderedDict
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIf, OnnxConstant, OnnxGreater, OnnxAdd, OnnxReduceSum,
    OnnxSub)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxrtPythonRuntimeControlIf(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_if(self):

        tensor_type = FloatTensorType
        op_version = get_opset_number_from_onnx()
        bthen = OnnxConstant(
            value_floats=numpy.array([0], dtype=numpy.float32),
            op_version=op_version, output_names=['res_then'])
        bthen.set_onnx_name_prefix('then')

        belse = OnnxConstant(
            value_floats=numpy.array([1], dtype=numpy.float32),
            op_version=op_version, output_names=['res_else'])
        belse.set_onnx_name_prefix('else')

        bthen_body = bthen.to_onnx(
            OrderedDict(), outputs=[('res_then', tensor_type())],
            target_opset=op_version)
        belse_body = belse.to_onnx(
            OrderedDict(),
            outputs=[('res_else', tensor_type())],
            target_opset=op_version)

        onx = OnnxIf(OnnxGreater('X', numpy.array([0], dtype=numpy.float32),
                                 op_version=op_version),
                     output_names=['Z'],
                     then_branch=bthen_body.graph,
                     else_branch=belse_body.graph,
                     op_version=op_version)

        x = numpy.array([1, 2], dtype=numpy.float32)
        y = numpy.array([1, 3], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32),
                                 'Y': y.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x, 'Y': y})
        self.assertEqualArray(numpy.array([0.], dtype=numpy.float32),
                              got['Z'])

        x = numpy.array([-1, -2], dtype=numpy.float32)
        y = numpy.array([-1, -3], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32),
                                 'Y': y.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x, 'Y': y})
        self.assertEqualArray(numpy.array([1.], dtype=numpy.float32),
                              got['Z'])

    @ignore_warnings(DeprecationWarning)
    def test_if2(self):

        opv = get_opset_number_from_onnx()
        x1 = numpy.array([[0, 3], [7, 0]], dtype=numpy.float32)
        x2 = numpy.array([[1, 0], [2, 0]], dtype=numpy.float32)

        node = OnnxAdd(
            'x1', 'x2', output_names=['absxythen'], op_version=opv)
        then_body = node.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['y'])
        model_def = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn("Gr_Greater -> Gr_C0;", dot)


if __name__ == "__main__":
    unittest.main()
