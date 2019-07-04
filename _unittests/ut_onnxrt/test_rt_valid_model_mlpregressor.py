"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateMLPRegressor(ExtTestCase):

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_MLPRegressor_simple_test(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        clr = MLPRegressor()
        clr.fit(X_train, y_train)

        x2 = X_test.astype(numpy.float32)
        onx = to_onnx(clr, x2)
        pyrun = OnnxInference(onx, runtime="python")
        res = pyrun.run({'X': x2})
        self.assertIn('variable', res)
        self.assertEqual(res['variable'].shape, (38, 1))

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_MLPRegressor_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"MLPRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
