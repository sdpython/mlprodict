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
from onnxruntime import __version__ as ort_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets
from mlprodict.onnxrt import OnnxInference


threshold = "0.4.0"


class TestRtValidateGaussianProcessOrt(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_kernel_rbf1(self):
        from skl2onnx.operator_converters.gaussian_process import convert_kernel
        ker = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType(['d1', 'd2']))])
        sess = OnnxInference(model_onnx, runtime='onnxruntime1')
        Xtest_ = numpy.arange(6).reshape((3, 2))
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_kernel_exp_sine_squared(self):
        from skl2onnx.operator_converters.gaussian_process import convert_kernel
        ker = ExpSineSquared()
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType(['d1', 'd2']))])
        sess = OnnxInference(model_onnx, runtime='onnxruntime1')
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
        verbose = 1

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, s: "nofit-std" in s))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_python_nofit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, s: "nofit" in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_python_fit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, s: "nofit" not in s and "multi" not in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_rt_GaussianProcessRegressor_debug(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, s: "reg-noshapevar" in s,
            disable_single=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_circleci("to investigate, shape of predictions are different")
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_rt_GaussianProcessRegressor_debug_std(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, s: s == "reg-std-d2-noshapevar",
            disable_single=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_circleci("to investigate, shape of predictions are different")
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_rt_GaussianProcessRegressor_debug_multi(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, s: s == 'mreg-std-d2-noshapevar',
            disable_single=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
