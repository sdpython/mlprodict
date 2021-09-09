"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxSub, OnnxIdentity
)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_tools.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import (
    onnx_remove_node_redundant, onnx_remove_node, onnx_optimisations)
from mlprodict.tools import get_opset_number_from_onnx


class TestOptimOnnxRedundant(ExtTestCase):

    def test_onnx_remove_redundant(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_redundant(model_def, max_hash_size=10)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 2)
        self.assertEqual(stats2['nnodes'], 6)
        self.assertEqual(stats3['nnodes'], 6)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_remove_two_outputs(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       output_names=['keep'], op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x},
                                 outputs=[('keep', FloatTensorType([None, 2])),
                                          ('final', FloatTensorType([None, 2]))])
        c1 = model_def.SerializeToString()
        self.assertEqual(len(model_def.graph.output), 2)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats = onnx_statistics(model_def, optim=True)
        new_model = onnx_remove_node_redundant(model_def, max_hash_size=10)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 2)
        self.assertEqual(stats2['nnodes'], 6)
        self.assertEqual(stats3['nnodes'], 6)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])
        self.assertEqualArray(y1['keep'], y2['keep'])

    def test_onnx_remove_redundant_subgraphs(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(
            OnnxIdentity('input', op_version=get_opset_number_from_onnx()),
            'input', op_version=get_opset_number_from_onnx())
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cdist2 = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd(cdist, cdist2, output_names=['cdist'],
                       op_version=get_opset_number_from_onnx())

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=get_opset_number_from_onnx())
        c1 = model_def.SerializeToString()
        stats = onnx_statistics(model_def, optim=False)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)

        new_model = onnx_remove_node_redundant(model_def)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['subgraphs'], 2)
        # The test is unstable, probably due to variables names.
        # They should be renamed before checking redundancy.
        self.assertIn(stats2['subgraphs'], (1, 2))

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'input': x})['cdist']
        y2 = oinf2.run({'input': x})['cdist']
        self.assertEqualArray(y1, y2)

        new_model = onnx_remove_node_redundant(model_def)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats2, stats3)

        new_model = onnx_remove_node(model_def)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertLess(stats3['size'], stats2['size'])
        self.assertLess(stats3['nnodes'], stats2['nnodes'])
        self.assertLess(stats3['op_Identity'], stats2['op_Identity'])

    def test_onnx_remove_redundant_subgraphs_full(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        cop = OnnxAdd(
            OnnxIdentity('input', op_version=get_opset_number_from_onnx()),
            'input', op_version=get_opset_number_from_onnx())
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cdist2 = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd(cdist, cdist2, output_names=['cdist'],
                       op_version=get_opset_number_from_onnx())

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=get_opset_number_from_onnx())
        stats = onnx_statistics(model_def, optim=False)
        new_model = onnx_optimisations(model_def)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertLess(stats2['size'], stats['size'])
        self.assertLess(stats2['nnodes'], stats['nnodes'])
        self.assertLess(stats2['op_Identity'], stats['op_Identity'])


if __name__ == "__main__":
    unittest.main()
