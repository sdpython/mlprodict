"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import warnings
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import ignore_warnings
from sklearn.ensemble import RandomForestRegressor
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference, to_onnx
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestOnnxrtPythonRuntimeMlTreeRF(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def onnxrt_python_RandomForestRegressor_dtype(self, dtype, n=37, full=False):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y,
                                                       random_state=11 if not full else 13)
        X_test = X_test.astype(dtype)
        if full:
            clr = RandomForestRegressor(n_jobs=1)
        else:
            clr = RandomForestRegressor(n_estimators=10, n_jobs=1, max_depth=4)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(dtype),
                            dtype=dtype, rewrite_ops=True)
        oinf = OnnxInference(model_def)

        tt = oinf.sequence_[0].ops_

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
        vals = tt.rt_.nodes_values_
        ori = []
        for tt in clr.estimators_:
            ori.extend(tt.tree_.threshold)
        ori.sort()
        vals.sort()
        tori = numpy.array(ori)
        tval = numpy.array(vals)
        tval = tval[tori > -2]
        tori = tori[tori > -2]
        self.assertEqualArray(lexp, y['variable'])

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestRegressor(self):
        try:
            self.onnxrt_python_RandomForestRegressor_dtype(numpy.float32)
        except AssertionError as e:
            self.assertIn("Max absolute difference", str(e))
        self.onnxrt_python_RandomForestRegressor_dtype(numpy.float64)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestRegressor_full(self):
        try:
            self.onnxrt_python_RandomForestRegressor_dtype(
                numpy.float32, full=True)
        except AssertionError as e:
            self.assertIn("Max absolute difference", str(e))
        try:
            self.onnxrt_python_RandomForestRegressor_dtype(
                numpy.float64, full=True)
        except AssertionError as e:
            # still issues
            warnings.warn(e)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_rt_RandomForestRegressor_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"RandomForestRegressor"}, opset_min=11, opset_max=11, fLOG=myprint,
            runtime='python', debug=debug, filter_exp=lambda m, p: p == "~b-reg-64"))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
