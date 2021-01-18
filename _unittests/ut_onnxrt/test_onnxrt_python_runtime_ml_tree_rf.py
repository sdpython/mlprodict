"""
@brief      test log(time=12s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.experimental import enable_hist_gradient_boosting  # pylint: disable=W0611
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import (
    ExtTestCase, skipif_appveyor,
    skipif_circleci
)
from mlprodict.onnx_conv import to_onnx, register_rewritten_operators
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestOnnxrtPythonRuntimeMlTreeRF(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_rewritten_operators()

    def onnxrt_python_RandomForestRegressor_dtype(
            self, dtype, n=37, full=False, use_hist=False, ntrees=10,
            runtime='python'):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11 if not full else 13)
        X_test = X_test.astype(dtype)
        if use_hist:
            if full:
                clr = HistGradientBoostingRegressor()
            else:
                clr = HistGradientBoostingRegressor(
                    max_iter=ntrees, max_depth=4)
        else:
            if full:
                clr = RandomForestRegressor(n_jobs=1)
            else:
                clr = RandomForestRegressor(
                    n_estimators=ntrees, n_jobs=1, max_depth=3)

        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(dtype),
                            rewrite_ops=True)
        oinf = OnnxInference(model_def)

        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        if full:
            n = 34
            X_test = X_test[n:n + 5]
        else:
            n = 37
            X_test = X_test[n:n + 5]
        X_test = numpy.vstack([X_test, X_test[:1].copy() * 1.01,
                               X_test[:1].copy() * 0.99])
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        if dtype == numpy.float32:
            self.assertEqualArray(lexp, y['variable'], decimal=5)
        else:
            try:
                self.assertEqualArray(lexp, y['variable'])
            except AssertionError as e:
                raise AssertionError(
                    "---------\n{}\n-----".format(model_def)) from e
        self.assertEqual(oinf.sequence_[0].ops_.rt_.same_mode_, True)
        self.assertNotEmpty(oinf.sequence_[0].ops_.rt_.nodes_modes_)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestRegressor32(self):
        self.onnxrt_python_RandomForestRegressor_dtype(numpy.float32)

    @skipif_circleci('too long')
    @skipif_appveyor("issue with opset 11")
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestRegressor64(self):
        self.onnxrt_python_RandomForestRegressor_dtype(numpy.float64)

    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_HistGradientBoostingRegressor32_hist(self):
        self.onnxrt_python_RandomForestRegressor_dtype(
            numpy.float32, use_hist=True)

    @skipif_circleci('too long')
    @skipif_appveyor("issue with opset 11")
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_HistGradientBoostingRegressor64_hist(self):
        self.onnxrt_python_RandomForestRegressor_dtype(
            numpy.float64, use_hist=True)

    @skipif_appveyor("issue with opset 11")
    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_HistGradientBoostingRegressor64_hist_compiled(self):
        self.onnxrt_python_RandomForestRegressor_dtype(
            numpy.float64, use_hist=True, runtime="python_compiled")

    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestRegressor_full32(self):
        self.onnxrt_python_RandomForestRegressor_dtype(
            numpy.float32, full=True)

    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestRegressor_full64(self):
        self.onnxrt_python_RandomForestRegressor_dtype(
            numpy.float64, full=True)

    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_rt_RandomForestRegressor_python64_compiled(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        debug = True
        buffer = []
        pps = []

        def pp(p):
            pps.append(p)
            return p

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"RandomForestRegressor"},
            opset_min=-1, fLOG=myprint,
            runtime='python_compiled', debug=debug,
            filter_exp=lambda m, p: pp(p) == "~b-reg-64"))
        if len(rows) == 0:
            raise AssertionError("Empty rows: {}".format(pps))

    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_rt_HistGradientBoostingRegressor_python64_compiled(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        debug = True
        buffer = []
        pps = []

        def pp(p):
            pps.append(p)
            return p

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"HistGradientBoostingRegressor"}, opset_min=-1, fLOG=myprint,
            runtime='python_compiled', debug=debug,
            filter_exp=lambda m, p: pp(p) == '~b-reg-64'))
        if len(rows) == 0:
            raise AssertionError("Empty rows: {}".format(pps))


if __name__ == "__main__":
    unittest.main()
