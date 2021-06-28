"""
@brief      test log(time=70s)
"""
import unittest
import warnings
from logging import getLogger
from pyquickhelper.pycode import ExtTestCase


from mlprodict.onnxrt.ops_cpu.op_qlinear_conv_ import (  # pylint: disable=W0611,E0611,E0401
    test_qlinear_qgemm_ii, test_qlinear_qgemm_ui,
    test_qlinear_qgemm_if, test_qlinear_qgemm_uf,
    test_qlinear_conv_Conv1D_U8S8,
    test_qlinear_conv_Conv2D_U8S8,
    test_qlinear_conv_Conv3D_U8S8,
    test_qlinear_conv_Conv1D_U8S8_Pointwise,
    test_qlinear_conv_Conv2D_U8S8_Pointwise,
    test_qlinear_conv_Conv2D_U8U8_Pointwise,
    test_qlinear_conv_Conv3D_U8S8_Pointwise,
    test_qlinear_conv_Conv1D_U8S8_Dilations,
    test_qlinear_conv_Conv2D_U8S8_Dilations,
    test_qlinear_conv_Conv3D_U8S8_Dilations,
    test_qlinear_conv_Conv1D_U8S8_Strides,
    test_qlinear_conv_Conv2D_U8S8_Strides,
    test_qlinear_conv_Conv3D_U8S8_Strides,
    test_qlinear_conv_Conv1D_U8S8_Depthwise,
    test_qlinear_conv_Conv2D_U8S8_Depthwise,
    test_qlinear_conv_Conv2D_U8U8_Depthwise,
    test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise,
    test_qlinear_conv_Conv3D_U8S8_Depthwise,
    test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias,
    test_qlinear_conv_Conv2D_U8S8_Requantize_Bias,
    test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel,
    test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise,
    test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise,
    test_qlinear_conv_Conv2D_U8S8_Groups,
    test_qlinear_conv_Conv3D_U8S8_Groups,
    test_qlinear_conv_Conv1D_U8S8_Groups,
    test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel)


def wraplog():
    # from datetime import datetime
    def wrapper(fct):
        def call_f(self):
            # no = datetime.now()
            # print('BEGIN %s' % fct.__name__)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", DeprecationWarning)
                fct(self)
            # print('DONE %s - %r' % (fct.__name__, datetime.now() - no))
        return call_f
    return wrapper


class TestOnnxrtPythonRuntime(ExtTestCase):  # pylint: disable=R0904

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @wraplog()
    def test_onnxt_runtime_qlinear_qgemm_cpp(self):
        with self.subTest(fct="test_qlinear_qgemm_ii"):
            test_qlinear_qgemm_ii()
        with self.subTest(fct="test_qlinear_qgemm_if"):
            test_qlinear_qgemm_if()
        with self.subTest(fct="test_qlinear_qgemm_ui"):
            test_qlinear_qgemm_ui()
        with self.subTest(fct="test_qlinear_qgemm_uf"):
            test_qlinear_qgemm_uf()

    @wraplog()
    def test_onnxt_runtime_qlinear_conv_cpp(self):
        fcts = [
            test_qlinear_conv_Conv1D_U8S8,
            test_qlinear_conv_Conv2D_U8S8,
            test_qlinear_conv_Conv3D_U8S8,
            test_qlinear_conv_Conv1D_U8S8_Pointwise,
            test_qlinear_conv_Conv2D_U8S8_Pointwise,
            test_qlinear_conv_Conv2D_U8U8_Pointwise,
            test_qlinear_conv_Conv3D_U8S8_Pointwise,
            test_qlinear_conv_Conv1D_U8S8_Dilations,
            test_qlinear_conv_Conv2D_U8S8_Dilations,
            test_qlinear_conv_Conv3D_U8S8_Dilations,
            test_qlinear_conv_Conv1D_U8S8_Strides,
            test_qlinear_conv_Conv2D_U8S8_Strides,
            test_qlinear_conv_Conv3D_U8S8_Strides,
            test_qlinear_conv_Conv1D_U8S8_Depthwise,
            test_qlinear_conv_Conv2D_U8S8_Depthwise,
            test_qlinear_conv_Conv2D_U8U8_Depthwise,
            test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise,
            test_qlinear_conv_Conv3D_U8S8_Depthwise,
            test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias,
            test_qlinear_conv_Conv2D_U8S8_Requantize_Bias,
            test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel,
            test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise,
            test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise,
            test_qlinear_conv_Conv2D_U8S8_Groups,
            test_qlinear_conv_Conv3D_U8S8_Groups,
            test_qlinear_conv_Conv1D_U8S8_Groups,
            test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel,
        ]

        for rnd in [False, True]:
            for fct in fcts:
                with self.subTest(fct=fct.__name__, rnd=rnd):
                    fct(rnd)


if __name__ == "__main__":
    unittest.main()
