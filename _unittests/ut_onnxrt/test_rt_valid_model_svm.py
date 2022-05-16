"""
@brief      test log(time=16s)
"""
import unittest
from logging import getLogger
from uuid import uuid4
import numpy
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx.common.data_types import DoubleTensorType
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets
from mlprodict.onnxrt.validate.validate_problems import _modify_dimension


class TestRtValidateSVM(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_simple_test(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        for nf in range(16, 50):
            with self.subTest(nf=nf):
                iris = load_iris()
                X, y = iris.data, iris.target
                X = _modify_dimension(X, nf)
                X_train, X_test, y_train, _ = train_test_split(X, y)
                clr = SVR(kernel='linear')
                clr.fit(X_train, y_train)

                x2 = X_test.astype(numpy.float32)
                onx = to_onnx(clr, x2)
                pyrun = OnnxInference(onx, runtime="python")
                res = pyrun.run({'X': x2})
                self.assertIn('variable', res)
                self.assertEqual(res['variable'].shape, (38, ))
                self.assertEqualArray(
                    res['variable'], clr.predict(x2), decimal=2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_simple_test_double(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        for nf in range(16, 50):
            with self.subTest(nf=nf):
                iris = load_iris()
                X, y = iris.data, iris.target
                X = _modify_dimension(X, nf)
                X_train, X_test, y_train, _ = train_test_split(X, y)
                clr = SVR(kernel='linear')
                clr.fit(X_train, y_train)

                x2 = X_test.astype(numpy.float64)
                onx = to_onnx(clr, x2)
                pyrun = OnnxInference(onx, runtime="python")
                res = pyrun.run({'X': x2})
                self.assertIn('variable', res)
                self.assertEqual(res['variable'].shape, (38, ))
                self.assertEqualArray(
                    res['variable'], clr.predict(x2), decimal=2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_python_rbf(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"SVR"},
            fLOG=myprint, benchmark=False,
            n_features=[45],
            runtime='python', debug=debug,
            filter_exp=lambda m, p: "64" not in p,
            filter_scenario=lambda m, p, s, e, t: "rbf" in str(e)))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_python_linear(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"SVR"},
            fLOG=myprint, benchmark=False,
            n_features=[45],
            runtime='python', debug=debug,
            filter_exp=lambda m, p: "64" not in p,
            filter_scenario=lambda m, p, s, e, t: "linear" in str(e)))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    def test_svc_runtime(self):
        # See https://github.com/microsoft/onnxruntime/issues/11490.

        def samples_df() -> DataFrame:
            headers = ["feat_1", "feat_2", "feat_3", "member"]
            value = [
                [1000., 0., 0., "class_1"],
                [1001., 0., 0., "class_1"],
                [1002., 0., 0., "class_1"],
                [1003., 0., 0., "class_1"],
                [1004., 0., 0., "class_1"],
                #
                [1., 1000., 5., "class_2"],
                [2., 1002., 60., "class_2"],
                [3., 1004., 7000., "class_2"],
                [4., 1006., 8., "class_2"],
                [5., 1008., 9., "class_2"],
                #
                [6., 0., 1000., "class_3"],
                [7., 0., 1000., "class_3"],
                [8000., 0., 1000., "class_3"],
                [9., 0., 1000., "class_3"],
                [10., 0., 1000., "class_3"],
            ]
            df = DataFrame(data=value, columns=headers)
            df["uuid"] = [uuid4() for _ in range(len(df.index))]
            return df

        def instances_df():
            headers = ["feat_1", "feat_2", "feat_3"]
            value = [
                [1000., 0., 0.],
                [1., 1000., 0.],
                [0., 0., 1000.],
            ]
            df = DataFrame(data=value, columns=headers)
            df["uuid"] = [uuid4() for _ in range(len(df.index))]
            return df

        def classification_targets():
            return ["class_1", "class_2", "class_3"]

        def compare_skl_vs_onnx(samples, instances, targets):
            features = ["feat_1", "feat_2", "feat_3"]
            labels = "member"
            svc = SVC(
                C=9.725493894658872,
                gamma=1 / 3, kernel="linear", probability=True)
            svc.fit(X=samples[features], y=numpy.ravel(samples[labels]))
            classifications = svc.predict(instances[features])
            probas = svc.predict_proba(instances[features])

            initial_types = [(key, DoubleTensorType()) for key in features]
            onnx_model = to_onnx(
                svc, initial_types=initial_types, verbose=False,
                options={'zipmap': False}, rewrite_ops=True)

            inputs = {
                key: numpy.expand_dims(instances[key].to_numpy(dtype=numpy.float64), axis=1)
                for key in features}    

            oinf = OnnxInference(onnx_model)
            res = oinf.run(inputs)
            self.assertEqualArray(probas, res['probabilities'])
            self.assertEqualArray(classifications, res['output_label'])

        samples = samples_df()
        instances = instances_df()
        targets = classification_targets()
        for i in range(0,10):
            compare_skl_vs_onnx(samples, instances, targets)


if __name__ == "__main__":
    # TestRtValidateSVM().test_svc_runtime()
    unittest.main()
