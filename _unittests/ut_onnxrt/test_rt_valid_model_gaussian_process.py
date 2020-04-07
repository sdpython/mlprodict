"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
from mlprodict.onnxrt import OnnxInference


class TestRtValidateGaussianProcess(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_kernel_rbf1(self):
        from skl2onnx.operator_converters.gaussian_process import convert_kernel
        ker = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32,
                             op_version=10)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))])
        sess = OnnxInference(model_onnx)
        Xtest_ = numpy.arange(6).reshape((3, 2))
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_kernel_exp_sine_squared(self):
        from skl2onnx.operator_converters.gaussian_process import convert_kernel
        ker = ExpSineSquared()
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32,
                             op_version=10)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))])
        sess = OnnxInference(model_onnx)
        Xtest_ = numpy.arange(6).reshape((3, 2))
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2, decimal=5)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_onnxruntime_nofit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=debug,
            filter_exp=lambda m, s: "NF-std" in s))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_python_nofit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: "NF" in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_python_fit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: "nofit" not in s and "multi" not in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1 if debug else 0)
        optim_values = []
        for row in rows:
            optim_values.append(row.get('optim', ''))
        expcl = "<class 'sklearn.gaussian_process.gpr.GaussianProcessRegressor'>={'optim': 'cdist'}"
        expcl2 = "<class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>={'optim': 'cdist'}"
        exp = [{'', 'onnx/' + expcl, expcl, 'onnx'},
               {'', 'onnx/' + expcl, expcl},
               {'', 'onnx/' + expcl2, expcl2, 'onnx'},
               {'', 'onnx/' + expcl2, expcl2},
               {'', 'onnx'}]
        self.assertIn(set(optim_values), exp)
        piv = summary_report(DataFrame(rows))
        expcl = 'cdist'
        exp = [{'', 'onnx/' + expcl, expcl, 'onnx'},
               {'', 'onnx/' + expcl, expcl},
               {'', 'cdist', 'onnx'},
               {'', 'onnx'}]
        self.assertIn(set(piv['optim']), exp)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_debug(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True  # should be true
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: "reg-" in s and "cov" not in s))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_circleci("to investigate, shape of predictions are different")
    def test_rt_GaussianProcessRegressor_debug_std(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True  # should be true
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: s == "~b-reg-NSV-64"))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_circleci("to investigate, shape of predictions are different")
    def test_rt_GaussianProcessRegressor_debug_multi(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: s == '~m-reg-std-NSV-64'))
        self.assertGreater(len(rows), 0)

    def test_partial_float64(self):
        data = load_boston()
        X, y = data.data, data.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        gau = GaussianProcessRegressor(alpha=10, kernel=DotProduct())
        gau.fit(X_train, y_train)
        onnxgau48 = to_onnx(gau, X_train.astype(
            numpy.float32), dtype=numpy.float32)
        oinf48 = OnnxInference(onnxgau48, runtime="python")
        out = oinf48.run({'X': X_test.astype(numpy.float32)})
        y = out['GPmean']
        self.assertEqual(y.dtype, numpy.float32)


if __name__ == "__main__":
    unittest.main()
