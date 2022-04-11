"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdagrad, OnnxAdam)
from skl2onnx import __version__ as skl2onnx_version
from onnx.backend.test.case.node.adagrad import apply_adagrad
from onnx.backend.test.case.node.adam import apply_adam
from mlprodict.onnxrt import OnnxInference
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestOnnxrtPythonRuntimeTraining(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_runtime_adagrad(self):
        norm_coefficient = 0.001
        epsilon = 1e-5
        decay_factor = 0.1

        r = numpy.array(0.1, dtype=numpy.float32)  # scalar
        t = numpy.array(0, dtype=numpy.int64)  # scalar
        x = numpy.array([1.0], dtype=numpy.float32)
        g = numpy.array([-1.0], dtype=numpy.float32)
        h = numpy.array([2.0], dtype=numpy.float32)

        node = OnnxAdagrad(
            'R', 'T', 'X', 'G', 'H',
            output_names=['X_new', 'H_new'],
            norm_coefficient=norm_coefficient,
            epsilon=epsilon,
            decay_factor=decay_factor,
            domain="ai.onnx.preview.training",
            op_version=1)

        onx = node.to_onnx({'R': r, 'T': t, 'X': x, 'G': g, 'H': h},
                           target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got = oinf.run({'R': r, 'T': t, 'X': x, 'G': g, 'H': h})

        x_new, h_new = apply_adagrad(
            r, t, x, g, h, norm_coefficient, epsilon, decay_factor)
        self.assertEqualArray(x_new, got['X_new'])
        self.assertEqualArray(h_new, got['H_new'])

    def test_onnx_runtime_adagrad_multiple(self):
        norm_coefficient = 0.001
        epsilon = 1e-5
        decay_factor = 0.1

        r = numpy.array(0.1, dtype=numpy.float32)  # scalar
        t = numpy.array(0, dtype=numpy.int64)  # scalar
        x1 = numpy.array([1.0], dtype=numpy.float32)
        g1 = numpy.array([-1.0], dtype=numpy.float32)
        h1 = numpy.array([2.0], dtype=numpy.float32)
        x2 = numpy.array([1.0, 2.0], dtype=numpy.float32)
        g2 = numpy.array([-1.0, -3.0], dtype=numpy.float32)
        h2 = numpy.array([4.0, 1.0], dtype=numpy.float32)

        node = OnnxAdagrad(
            'R', 'T', 'X1', 'X2', 'G1', 'G2', 'H1', 'H2',
            output_names=['X1_new', 'X2_new', 'H1_new', 'H2_new'],
            norm_coefficient=norm_coefficient,
            epsilon=epsilon,
            decay_factor=decay_factor,
            domain="ai.onnx.preview.training",
            op_version=1)

        onx = node.to_onnx({'R': r, 'T': t,
                            'X1': x1, 'G1': g1, 'H1': h1,
                            'X2': x2, 'G2': g2, 'H2': h2},
                           target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got = oinf.run({'R': r, 'T': t,
                        'X1': x1, 'G1': g1, 'H1': h1,
                        'X2': x2, 'G2': g2, 'H2': h2})

        x1_new, h1_new = apply_adagrad(
            r, t, x1, g1, h1, norm_coefficient, epsilon, decay_factor)
        x2_new, h2_new = apply_adagrad(
            r, t, x2, g2, h2, norm_coefficient, epsilon, decay_factor)
        self.assertEqualArray(x1_new, got['X1_new'])
        self.assertEqualArray(h1_new, got['H1_new'])
        self.assertEqualArray(x2_new, got['X2_new'])
        self.assertEqualArray(h2_new, got['H2_new'])

    def test_onnxt_runtime_adam(self):
        norm_coefficient = 0.001
        alpha = 0.95
        beta = 0.1
        epsilon = 1e-7
        r = numpy.array(0.1, dtype=numpy.float32)  # scalar
        t = numpy.array(0, dtype=numpy.int64)  # scalar
        x = numpy.array([1.2, 2.8], dtype=numpy.float32)
        g = numpy.array([-0.94, -2.5], dtype=numpy.float32)
        v = numpy.array([1.7, 3.6], dtype=numpy.float32)
        h = numpy.array([0.1, 0.1], dtype=numpy.float32)

        node = OnnxAdam(
            'R', 'T', 'X', 'G', 'V', 'H',
            output_names=['X_new', 'V_new', 'H_new'],
            norm_coefficient=norm_coefficient,
            epsilon=epsilon, alpha=alpha, beta=beta,
            domain="ai.onnx.preview.training",
            op_version=1)

        onx = node.to_onnx({'R': r, 'T': t,
                            'X': x, 'G': g, 'H': h, 'V': v},
                           target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got = oinf.run({'R': r, 'T': t, 'X': x, 'G': g, 'H': h, 'V': v})
        x_new, v_new, h_new = apply_adam(r, t, x, g, v, h,
                                         norm_coefficient, 0.0, alpha, beta,
                                         epsilon)
        self.assertEqualArray(x_new, got['X_new'])
        self.assertEqualArray(v_new, got['V_new'])
        self.assertEqualArray(h_new, got['H_new'])

    def test_onnxt_runtime_adam_multiple(self):
        norm_coefficient = 0.001
        alpha = 0.95
        beta = 0.85
        epsilon = 1e-2
        r = numpy.array(0.1, dtype=numpy.float32)  # scalar
        t = numpy.array(0, dtype=numpy.int64)  # scalar
        x1 = numpy.array([1.0], dtype=numpy.float32)
        g1 = numpy.array([-1.0], dtype=numpy.float32)
        v1 = numpy.array([2.0], dtype=numpy.float32)
        h1 = numpy.array([0.5], dtype=numpy.float32)
        x2 = numpy.array([1.0, 2.0], dtype=numpy.float32)
        g2 = numpy.array([-1.0, -3.0], dtype=numpy.float32)
        v2 = numpy.array([4.0, 1.0], dtype=numpy.float32)
        h2 = numpy.array([1.0, 10.0], dtype=numpy.float32)

        node = OnnxAdam(
            'R', 'T', 'X1', 'X2', 'G1', 'G2', 'V1', 'V2', 'H1', 'H2',
            output_names=['X1_new', 'X2_new',
                          'V1_new', 'V2_new',
                          'H1_new', 'H2_new'],
            norm_coefficient=norm_coefficient,
            epsilon=epsilon, alpha=alpha, beta=beta,
            domain="ai.onnx.preview.training",
            op_version=1)

        onx = node.to_onnx({'R': r, 'T': t,
                            'X1': x1, 'G1': g1, 'H1': h1, 'V1': v1,
                            'X2': x2, 'G2': g2, 'H2': h2, 'V2': v2},
                           target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got = oinf.run({'R': r, 'T': t,
                        'X1': x1, 'G1': g1, 'H1': h1, 'V1': v1,
                        'X2': x2, 'G2': g2, 'H2': h2, 'V2': v2})
        x1_new, v1_new, h1_new = apply_adam(
            r, t, x1, g1, v1, h1, norm_coefficient, 0.0, alpha, beta, epsilon)
        x2_new, v2_new, h2_new = apply_adam(
            r, t, x2, g2, v2, h2, norm_coefficient, 0.0, alpha, beta, epsilon)
        self.assertEqualArray(x1_new, got['X1_new'])
        self.assertEqualArray(v1_new, got['V1_new'])
        self.assertEqualArray(h1_new, got['H1_new'])
        self.assertEqualArray(x2_new, got['X2_new'])
        self.assertEqualArray(v2_new, got['V2_new'], decimal=4)
        self.assertEqualArray(h2_new, got['H2_new'], decimal=4)


if __name__ == "__main__":
    unittest.main()
