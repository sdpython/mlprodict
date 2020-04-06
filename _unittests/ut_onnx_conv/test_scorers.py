"""
@brief      test log(time=4s)
"""
import os
import unittest
from logging import getLogger
from collections import OrderedDict
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from sklearn.metrics import make_scorer
try:
    from sklearn.metrics._scorer import _PredictScorer
except ImportError:
    from sklearn.metrics.scorer import _PredictScorer
from mlprodict.onnx_conv import to_onnx, register_scorers
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv.scorers.cdist_score import score_cdist_sum


class TestScorers(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_scorers()

    def test_score_cdist_sum(self):
        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        Y = X[:2].copy()
        Y[0, :] = 0

        res = score_cdist_sum(X, Y)
        self.assertEqualArray(res, numpy.array([32, 42, 336], dtype=float))
        scorer = make_scorer(score_cdist_sum, greater_is_better=False)
        self.assertIsInstance(scorer, _PredictScorer)

    def test_score_cdist_sum_onnx(self):
        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        Y = X[:2].copy()
        Y[0, :] = 0

        init_types = OrderedDict([('X', X), ('Y', Y)])

        options = {id(score_cdist_sum): {"cdist": "single-node"}}
        temp = get_temp_folder(__file__, 'temp_score_cdist_sum_onnx')

        for metric in ['sqeuclidean', 'euclidean']:
            scorer = make_scorer(
                score_cdist_sum, metric=metric, greater_is_better=False)
            self.assertRaise(lambda: to_onnx(scorer, X),  # pylint: disable=W0640
                             ValueError)

            monx1 = to_onnx(scorer, init_types)
            monx2 = to_onnx(scorer, init_types, options=options)

            oinf1 = OnnxInference(monx1)
            oinf2 = OnnxInference(monx2)
            res0 = score_cdist_sum(X, Y, metric=metric)
            res1 = oinf1.run({'X': X, 'Y': Y})['scores']
            res2 = oinf2.run({'X': X, 'Y': Y})['scores']
            self.assertEqual(res1, res0)
            self.assertEqual(res2, res0)

            name1 = os.path.join(temp, "cdist_scan_%s.onnx" % metric)
            with open(name1, 'wb') as f:
                f.write(monx1.SerializeToString())
            name2 = os.path.join(temp, "cdist_cdist_%s.onnx" % metric)
            with open(name2, 'wb') as f:
                f.write(monx2.SerializeToString())
            data = os.path.join(temp, "data_%s.txt" % metric)
            with open(data, "w") as f:
                f.write("X\n")
                f.write(str(X) + "\n")
                f.write("Y\n")
                f.write(str(Y) + "\n")
                f.write("expected\n")
                f.write(str(res0) + "\n")


if __name__ == "__main__":
    unittest.main()
