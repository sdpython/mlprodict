"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxAdd
)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.optim.onnx_optimization import remove_node_identity


class TestOptimOnnx(ExtTestCase):

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnx_remove_identities(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(OnnxIdentity('input'), 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())])
        stats = onnx_statistics(model_def)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)

        new_model = remove_node_identity(model_def)
        stats2 = onnx_statistics(new_model)
        self.assertEqual(stats['subgraphs'], stats2['subgraphs'])
        self.assertLesser(stats2['op_Identity'], 2)

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'input': x})['cdist']
        y2 = oinf2.run({'input': x})['cdist']
        self.assertEqualArray(y1, y2)
        self.assertLesser(stats2['op_Identity'], 1)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnx_remove_identities2(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxIdentity('input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())])
        stats = onnx_statistics(model_def)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)

        new_model = remove_node_identity(model_def)
        stats2 = onnx_statistics(new_model)
        self.assertEqual(stats['subgraphs'], stats2['subgraphs'])
        self.assertLesser(stats2['op_Identity'], 2)

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'input': x})['cdist']
        y2 = oinf2.run({'input': x})['cdist']
        self.assertEqualArray(y1, y2)
        self.assertLesser(stats2['op_Identity'], 1)


if __name__ == "__main__":
    unittest.main()
