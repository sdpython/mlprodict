"""
@brief      test log(time=218s)
"""
import unittest
from logging import getLogger
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate import (
    sklearn_operators,
    enumerate_validated_operator_opsets,
)


class TestRtValidateLightGBM(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators(extended=True)
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    def test_sklearn_operator_here(self):
        subfolders = ['ensemble'] + ['mlprodict.onnx_conv']
        for sub in sorted(subfolders):
            models = sklearn_operators(sub)
            if len(models) == 0:
                raise AssertionError(
                    "models is empty for subfolder '{}'.".format(sub))
            if sub == "mlprodict.onnx_conv":
                names = set(_['name'] for _ in models)
                self.assertIn("LGBMClassifier", names)

    def test_validate_LGBMClassifier(self):
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
            runtime='python', debug=debug,
            filter_exp=lambda m, p: '-64' not in p,
            extended_list=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
