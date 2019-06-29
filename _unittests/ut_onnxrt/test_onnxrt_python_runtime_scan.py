"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.spatial.distance import squareform, pdist
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxAdd
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.complex_functions import squareform_cdist
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeScan(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_pdist(self):
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('input', 'input')
        cdist = squareform_cdist(cop)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': x}, outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})
        self.assertEqual(list(res.keys()), ['cdist'])

        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        self.assertEqualArray(exp, res['cdist'])


if __name__ == "__main__":
    # TestOnnxrtPythonRuntime().test_onnxt_runtime_topk2()
    unittest.main()
