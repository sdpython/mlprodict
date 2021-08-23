"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxSub)
from mlprodict.onnx_tools.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import onnx_remove_node_unused
from mlprodict.onnx_tools.onnx_manipulations import (
    select_model_inputs_outputs)
from mlprodict.tools import get_opset_number_from_onnx


class TestOptimOnnxUnused(ExtTestCase):

    def test_onnx_remove_unused(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, "inter", remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 1)
        self.assertEqual(stats2['nnodes'], 1)
        self.assertEqual(stats3['nnodes'], 1)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertNotIn('final', y1)
        self.assertNotIn('final', y2)
        self.assertIn('inter', y1)
        self.assertIn('inter', y2)
        self.assertEqualArray(y1['inter'], y2['inter'])


if __name__ == "__main__":
    unittest.main()
