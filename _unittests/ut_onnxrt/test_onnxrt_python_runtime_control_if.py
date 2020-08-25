"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
from collections import OrderedDict
import numpy
from scipy.spatial.distance import squareform, pdist, cdist as scipy_cdist
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIf, OnnxConstant, OnnxSum, OnnxGreater, OnnxIdentity)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxrtPythonRuntimeControlIf(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_if(self):

        tensor_type = FloatTensorType
        op_version = get_opset_number_from_onnx()
        bthen = OnnxConstant(value_floats=numpy.array([0], dtype=numpy.float32),
                             op_version=op_version, output_names=['res'])
        belse = OnnxConstant(value_floats=numpy.array([1], dtype=numpy.float32),
                             op_version=op_version, output_names=['res'])
        bthen_body = bthen.to_onnx(
            OrderedDict(),
            outputs=[('res', tensor_type())],
            target_opset=op_version)
        belse_body = belse.to_onnx(
            OrderedDict(),
            outputs=[('res', tensor_type())],
            target_opset=op_version)

        node = OnnxSum('X', op_version=op_version)
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


if __name__ == "__main__":
    unittest.main()
