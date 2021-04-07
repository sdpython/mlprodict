"""
@brief      test log(time=4s)
"""
import os
import unittest
from logging import getLogger
from collections import OrderedDict
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, ignore_warnings
from sklearn.metrics import make_scorer
try:
    from sklearn.metrics._scorer import _PredictScorer
except ImportError:
    from sklearn.metrics.scorer import _PredictScorer
from mlprodict.onnx_conv import to_onnx, register_scorers
from mlprodict.onnx_conv.scorers.register import CustomScorerTransform
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv.scorers.cdist_score import score_cdist_sum
from mlprodict.tools.asv_options_helper import (
    get_opset_number_from_onnx)


class TestScorers(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_scorers()

    @ignore_warnings((DeprecationWarning, UserWarning))
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

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_custom_transformer(self):
        def mse(x, y):
            return x - y
        tr = CustomScorerTransform("mse", mse, {})
        rp = repr(tr)
        self.assertIn("CustomScorerTransform('mse'", rp)

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_score_cdist_sum_onnx(self):
        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        Y = X[:2].copy()
        Y[0, :] = 0

        init_types = OrderedDict([('X', X), ('Y', Y)])

        opsets = [11, get_opset_number_from_onnx()]
        options = {id(score_cdist_sum): {"cdist": "single-node"}}
        temp = get_temp_folder(__file__, 'temp_score_cdist_sum_onnx')

        for metric in ['sqeuclidean', 'euclidean', 'minkowski']:
            for opset in opsets:
                with self.subTest(metric=metric, opset=opset):
                    if metric == 'minkowski':
                        scorer = make_scorer(
                            score_cdist_sum, metric=metric,
                            greater_is_better=False, p=3)
                    else:
                        scorer = make_scorer(
                            score_cdist_sum, metric=metric,
                            greater_is_better=False)
                    self.assertRaise(
                        lambda: to_onnx(
                            scorer, X, target_opset=opset),  # pylint: disable=W0640
                        ValueError)

                    monx1 = to_onnx(scorer, init_types, target_opset=opset)
                    monx2 = to_onnx(scorer, init_types, options=options,
                                    target_opset=opset)

                    oinf1 = OnnxInference(monx1)
                    oinf2 = OnnxInference(monx2)
                    if metric == 'minkowski':
                        res0 = score_cdist_sum(X, Y, metric=metric, p=3)
                    else:
                        res0 = score_cdist_sum(X, Y, metric=metric)
                    res1 = oinf1.run({'X': X, 'Y': Y})['scores']
                    res2 = oinf2.run({'X': X, 'Y': Y})['scores']
                    self.assertEqualArray(res1, res0, decimal=5)
                    self.assertEqualArray(res2, res0, decimal=5)

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
