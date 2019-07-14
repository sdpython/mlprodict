"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets
from mlprodict.onnxrt import OnnxInference


threshold = "1.5.0"


class TestRtValidateGaussianProcess(ExtTestCase):

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="scan")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_kernel_rbf1(self):
        from skl2onnx.operator_converters.gaussian_process import convert_kernel
        ker = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        onx = convert_kernel(ker, 'X', output_names=['Y'])
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType(['d1', 'd2']))])
        sess = OnnxInference(model_onnx)
        Xtest_ = numpy.arange(6).reshape((3, 2))
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="scan")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_kernel_exp_sine_squared(self):
        from skl2onnx.operator_converters.gaussian_process import convert_kernel
        ker = ExpSineSquared()
        onx = convert_kernel(ker, 'X', output_names=['Y'])
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType(['d1', 'd2']))])
        sess = OnnxInference(model_onnx)
        Xtest_ = numpy.arange(6).reshape((3, 2))
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2, decimal=5)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="int64 not implemented for constants")
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
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime2', debug=debug,
            filter_exp=lambda m, s: "nofit-std" in s))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="int64 not implemented for constants")
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
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: "nofit" in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="int64 not implemented for constants")
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
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: "nofit" not in s and "multi" not in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_debug(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: "regression" in s,
            disable_single=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="int64 not implemented for constants")
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

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug, filter_exp=lambda m, s: s == "reg-std-d2",
            disable_single=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
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
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, s: s == 'multi-reg-std-d2',
            disable_single=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
