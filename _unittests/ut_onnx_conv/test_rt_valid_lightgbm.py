"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, unittest_require_at_least
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateLightGbm(ExtTestCase):

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @skipif_circleci("too long")
    def test_rt_lightgbm_regressor(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LGBMRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug, extended_list=True,
            filter_exp=lambda m, p: "-64" not in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @skipif_circleci("too long")
    def test_rt_lightgbm_regressor64(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LGBMRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug, extended_list=True,
            filter_exp=lambda m, p: "-64" in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @skipif_circleci("too long")
    def test_rt_lightgbm_classifier64(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LGBMClassifier"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug, extended_list=True,
            filter_exp=lambda m, p: "-64" in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @skipif_circleci("too long")
    def test_rt_lightgbm_classifier(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LGBMClassifier"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug, extended_list=True,
            filter_exp=lambda m, p: "-64" not in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
