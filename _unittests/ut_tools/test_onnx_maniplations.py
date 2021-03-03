"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxSub)
from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.optim import onnx_remove_node_unused
from mlprodict.tools.onnx_manipulations import (
    select_model_inputs_outputs)
from mlprodict.tools import get_opset_number_from_onnx


class TestOptimOnnxUnused(ExtTestCase):

    def test_onnx_remove_unused_outputs(self):
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
            model_def, "inter", infer_shapes=True)
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

    def test_onnx_remove_unused_inputs(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', cop2,
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=get_opset_number_from_onnx()),
            cop3, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, inputs=["inter"], infer_shapes=True)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 1)
        self.assertEqual(stats2['ninits'], 1)
        self.assertEqual(stats3['ninits'], 0)
        self.assertEqual(stats2['nnodes'], 2)
        self.assertEqual(stats3['nnodes'], 2)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'inter': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'inter': x})
        self.assertIn('final', y1)
        self.assertIn('final', y2)
        self.assertNotIn('inter', y1)
        self.assertNotIn('inter', y2)
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_remove_unused_inputs_overwrite(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', cop2,
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=get_opset_number_from_onnx()),
            cop3, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, inputs=["inter"], infer_shapes=False,
            overwrite=dict(inter=(numpy.float32, [None, None]),
                           final=(numpy.float32, [None, None])))
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 1)
        self.assertEqual(stats2['ninits'], 1)
        self.assertEqual(stats3['ninits'], 0)
        self.assertEqual(stats2['nnodes'], 2)
        self.assertEqual(stats3['nnodes'], 2)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'inter': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'inter': x})
        self.assertIn('final', y1)
        self.assertIn('final', y2)
        self.assertNotIn('inter', y1)
        self.assertNotIn('inter', y2)
        self.assertEqualArray(y1['final'], y2['final'])


if __name__ == "__main__":
    unittest.main()
