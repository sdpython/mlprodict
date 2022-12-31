"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper import compare_module_version
import skl2onnx
from skl2onnx.algebra.custom_ops import (  # pylint: disable=E0611
    OnnxCDist, OnnxSolve)
from mlprodict.onnx_conv.onnx_ops import (
    OnnxFFT, OnnxRFFT, OnnxFFT2D,
    OnnxComplexAbs, OnnxYieldOp,
    OnnxBroadcastGradientArgs, OnnxFusedMatMul,
    OnnxSoftmaxGrad_13)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.validate_python import validate_python_inference
from mlprodict import __max_supported_opset__ as TARGET_OPSET


python_tested = []


class TestOnnxrtPythonRuntimeCustom(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        if __name__ == "__main__":
            import pprint
            print('\n-----------')
            pprint.pprint(
                list(sorted({_.__name__ for _ in python_tested})))
            print('-----------')

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_cdist(self):
        for metric in ['sqeuclidean', 'euclidean']:
            with self.subTest(metric=metric):
                X = numpy.array([[2, 1], [0, 1]], dtype=float)
                Y = numpy.array([[2, 1, 5], [0, 1, 3]], dtype=float).T
                Z = cdist(X, Y, metric=metric)

                onx = OnnxCDist('X', 'Y', output_names=['Z'],
                                metric=metric,
                                op_version=TARGET_OPSET)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                         'Y': Y.astype(numpy.float32)},
                                        outputs={'Z': Z.astype(numpy.float32)},
                                        target_opset=TARGET_OPSET)
                self.assertIn(f's: "{metric}"', str(model_def))
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X, 'Y': Y})
                self.assertEqual(list(sorted(got)), ['Z'])
                self.assertEqualArray(Z, got['Z'], decimal=6)

                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X.astype(numpy.float32),
                             'Y': Y.astype(numpy.float32)},
                    tolerance=1e-6)
        python_tested.append(OnnxCDist)

    @unittest.skipIf(compare_module_version(skl2onnx.__version__, "1.9.1") <= 0,
                     reason="Missing complex support.")
    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_complex_abs(self):
        for dtype in [numpy.complex64, numpy.complex128]:
            with self.subTest(dtype=dtype):
                X = numpy.array([[2, 1j], [0, 1j]], dtype=dtype)
                Z = numpy.absolute(X)

                onx = OnnxComplexAbs('X', output_names=['Z'],
                                     op_version=TARGET_OPSET)
                model_def = onx.to_onnx({'X': X},
                                        outputs={'Z': Z},
                                        target_opset=TARGET_OPSET)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Z'])
                self.assertEqualArray(Z, got['Z'], decimal=6)

                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X}, tolerance=1e-6)
                python_tested.append(OnnxComplexAbs)

    @unittest.skipIf(compare_module_version(skl2onnx.__version__, "1.9.1") <= 0,
                     reason="Missing complex support.")
    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_fft(self):
        for dim in [1, 2]:
            for axis in [-1, 0, 1]:
                if axis >= dim:
                    continue
                with self.subTest(dim=dim, axis=axis):
                    if dim == 1:
                        X = numpy.arange(16).astype(numpy.float32)
                    elif dim == 2:
                        X = numpy.arange(48).astype(
                            numpy.float32).reshape((3, -1))
                    Y = numpy.fft.fft(X.astype(numpy.float32), axis=axis)

                    onx = OnnxFFT('X', output_names=['Y'],
                                  axis=axis, op_version=TARGET_OPSET)
                    model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                            outputs={'Y': Y},
                                            target_opset=TARGET_OPSET)
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': X})
                    self.assertEqual(list(sorted(got)), ['Y'])
                    self.assertEqualArray(Y, got['Y'], decimal=6)

                    oinfpy = OnnxInference(
                        model_def, runtime="python", inplace=True)
                    validate_python_inference(
                        oinfpy, {'X': X.astype(numpy.float32)},
                        tolerance=1e-6)

        for dim in [1, 2]:
            for axis in [-1, 0, 1]:
                if axis >= dim:
                    continue
                with self.subTest(dim=dim, axis=axis, length=8):
                    if dim == 1:
                        X = numpy.arange(16).astype(numpy.float32)
                    elif dim == 2:
                        X = numpy.arange(48).astype(
                            numpy.float32).reshape((3, -1))
                    Y = numpy.fft.fft(X.astype(numpy.float32), 8, axis=axis)

                    onx = OnnxFFT('X', numpy.array([8], dtype=numpy.int64),
                                  output_names=['Y'], axis=axis,
                                  op_version=TARGET_OPSET)
                    model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                            outputs={'Y': Y},
                                            target_opset=TARGET_OPSET)
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': X})
                    self.assertEqual(list(sorted(got)), ['Y'])
                    self.assertEqualArray(Y, got['Y'], decimal=5)

                    oinfpy = OnnxInference(
                        model_def, runtime="python", inplace=True)
                    validate_python_inference(
                        oinfpy, {'X': X.astype(numpy.float32)},
                        tolerance=1e-5)
                    python_tested.append(OnnxFFT)

    @unittest.skipIf(compare_module_version(skl2onnx.__version__, "1.9.1") <= 0,
                     reason="Missing complex support.")
    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_rfft(self):
        for dim in [1, 2]:
            for axis in [-1, 0, 1]:
                if axis >= dim:
                    continue
                with self.subTest(dim=dim, axis=axis):
                    if dim == 1:
                        X = numpy.arange(16).astype(numpy.float32)
                    elif dim == 2:
                        X = numpy.arange(48).astype(
                            numpy.float32).reshape((3, -1))
                    Y = numpy.fft.rfft(X.astype(numpy.float32), axis=axis)

                    onx = OnnxRFFT('X', output_names=['Y'],
                                   axis=axis, op_version=TARGET_OPSET)
                    model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                            outputs={'Y': Y},
                                            target_opset=TARGET_OPSET)
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': X})
                    self.assertEqual(list(sorted(got)), ['Y'])
                    self.assertEqualArray(Y, got['Y'], decimal=6)

                    oinfpy = OnnxInference(
                        model_def, runtime="python", inplace=True)
                    validate_python_inference(
                        oinfpy, {'X': X.astype(numpy.float32)},
                        tolerance=1e-6)

        for dim in [1, 2]:
            for axis in [-1, 0, 1]:
                if axis >= dim:
                    continue
                with self.subTest(dim=dim, axis=axis, length=8):
                    if dim == 1:
                        X = numpy.arange(16).astype(numpy.float32)
                    elif dim == 2:
                        X = numpy.arange(48).astype(
                            numpy.float32).reshape((3, -1))
                    Y = numpy.fft.rfft(X.astype(numpy.float32), 8, axis=axis)

                    onx = OnnxRFFT('X', numpy.array([8], dtype=numpy.int64),
                                   output_names=['Y'], axis=axis,
                                   op_version=TARGET_OPSET)
                    try:
                        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                                outputs={'Y': Y},
                                                target_opset=TARGET_OPSET)
                    except NotImplementedError as e:
                        raise AssertionError(
                            "Unable to convert due to %r (version=%r)." % (
                                e, skl2onnx.__version__)) from e
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': X})
                    self.assertEqual(list(sorted(got)), ['Y'])
                    self.assertEqualArray(Y, got['Y'], decimal=5)

                    oinfpy = OnnxInference(
                        model_def, runtime="python", inplace=True)
                    validate_python_inference(
                        oinfpy, {'X': X.astype(numpy.float32)},
                        tolerance=1e-5)
                    python_tested.append(OnnxRFFT)

    @unittest.skipIf(compare_module_version(skl2onnx.__version__, "1.9.1") <= 0,
                     reason="Missing complex support.")
    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_fft2d(self):
        for dim in [2]:
            for axis in [None, (-2, -1)]:
                with self.subTest(dim=dim, axis=axis):
                    if dim == 1:
                        X = numpy.arange(16).astype(numpy.float32)
                    elif dim == 2:
                        X = numpy.arange(48).astype(
                            numpy.float32).reshape((3, -1))
                    Y = numpy.fft.fft2(X.astype(numpy.float32), axes=axis)

                    if axis is not None:
                        onx = OnnxFFT2D('X', output_names=['Y'],
                                        axes=axis if axis is None else list(
                                            axis),
                                        op_version=TARGET_OPSET)
                    else:
                        onx = OnnxFFT2D('X', output_names=['Y'],
                                        op_version=TARGET_OPSET)
                    model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                            outputs={'Y': Y},
                                            target_opset=TARGET_OPSET)
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': X})
                    self.assertEqual(list(sorted(got)), ['Y'])
                    self.assertEqualArray(Y, got['Y'], decimal=5)

                    oinfpy = OnnxInference(
                        model_def, runtime="python", inplace=True)
                    validate_python_inference(
                        oinfpy, {'X': X.astype(numpy.float32)},
                        tolerance=1e-5)

        for dim in [2]:
            for axis in [None, (-2, -1)]:
                with self.subTest(dim=dim, axis=axis, length=(8, 8)):
                    if dim == 1:
                        X = numpy.arange(16).astype(numpy.float32)
                    elif dim == 2:
                        X = numpy.arange(48).astype(
                            numpy.float32).reshape((3, -1))
                    Y = numpy.fft.fft2(
                        X.astype(numpy.float32), (8, 8), axes=axis)

                    if axis is not None:
                        onx = OnnxFFT2D('X', numpy.array([8, 8], dtype=numpy.int64),
                                        output_names=['Y'],
                                        axes=axis if axis is None else list(
                                            axis),
                                        op_version=TARGET_OPSET)
                    else:
                        onx = OnnxFFT2D('X', numpy.array([8, 8], dtype=numpy.int64),
                                        output_names=['Y'],
                                        op_version=TARGET_OPSET)
                    model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                            outputs={'Y': Y},
                                            target_opset=TARGET_OPSET)
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': X})
                    self.assertEqual(list(sorted(got)), ['Y'])
                    self.assertEqualArray(Y, got['Y'], decimal=5)

                    oinfpy = OnnxInference(
                        model_def, runtime="python", inplace=True)
                    validate_python_inference(
                        oinfpy, {'X': X.astype(numpy.float32)},
                        tolerance=1e-5)
                    python_tested.append(OnnxRFFT)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_solve(self):
        for transposed in [False, True]:
            with self.subTest(transposed=transposed):
                A = numpy.array([[2, 1], [0, 1]], dtype=float)
                Y = numpy.array([2, 1], dtype=float)
                X = solve(A, Y, transposed=transposed)

                onx = OnnxSolve('A', 'Y', output_names=['X'],
                                transposed=transposed,
                                op_version=TARGET_OPSET)
                model_def = onx.to_onnx({'A': A.astype(numpy.float32),
                                         'Y': Y.astype(numpy.float32)},
                                        outputs={'X': X.astype(numpy.float32)},
                                        target_opset=TARGET_OPSET)
                oinf = OnnxInference(model_def)
                got = oinf.run({'A': A, 'Y': Y})
                self.assertEqual(list(sorted(got)), ['X'])
                self.assertEqualArray(X, got['X'], decimal=6)

                python_tested.append(OnnxCDist)
                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'A': A.astype(numpy.float32),
                             'Y': Y.astype(numpy.float32)})
                python_tested.append(OnnxSolve)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_yield_op(self):
        for dtype in [numpy.float32, numpy.float64]:
            with self.subTest(dtype=dtype):
                X = numpy.array([[2, 1], [0, 1]], dtype=dtype)
                Z = X

                onx = OnnxYieldOp('X', output_names=['Z'],
                                  op_version=TARGET_OPSET)
                model_def = onx.to_onnx({'X': X},
                                        outputs={'Z': Z},
                                        target_opset=TARGET_OPSET)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Z'])
                self.assertEqualArray(Z, got['Z'], decimal=6)

                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X}, tolerance=1e-6)
                python_tested.append(OnnxYieldOp)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_broadcast_gradient_args(self):
        X = numpy.array([2, 16, 1024, 1024], dtype=numpy.int64)
        Y = numpy.array([1, 1, 1024, 1024], dtype=numpy.int64)
        Z1 = numpy.array([], dtype=numpy.int64)
        Z2 = numpy.array([1, 0], dtype=numpy.int64)
        onx = OnnxBroadcastGradientArgs(
            'X', 'Y', output_names=['Z1', 'Z2'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X, 'Y': Y}, outputs={'Z1': Z1, 'Z2': Z2},
            target_opset=TARGET_OPSET)

        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([2, 3, 4, 5], dtype=numpy.int64)
        Y = numpy.array([2, 3, 4, 5], dtype=numpy.int64)
        Z1 = numpy.array([], dtype=numpy.int64)
        Z2 = numpy.array([], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([2, 3, 4, 5], dtype=numpy.int64)
        Y = numpy.array([], dtype=numpy.int64)
        Z1 = numpy.array([], dtype=numpy.int64)
        Z2 = numpy.array([3, 2, 1, 0], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([2, 3, 4, 5], dtype=numpy.int64)
        Y = numpy.array([5], dtype=numpy.int64)
        Z1 = numpy.array([], dtype=numpy.int64)
        Z2 = numpy.array([2, 1, 0], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([4, 5], dtype=numpy.int64)
        Y = numpy.array([2, 3, 4, 5], dtype=numpy.int64)
        Z1 = numpy.array([1, 0], dtype=numpy.int64)
        Z2 = numpy.array([], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([1, 4, 5], dtype=numpy.int64)
        Y = numpy.array([2, 3, 1, 1], dtype=numpy.int64)
        Z1 = numpy.array([1, 0], dtype=numpy.int64)
        Z2 = numpy.array([3, 2], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([3, 4, 5], dtype=numpy.int64)
        Y = numpy.array([2, 1, 1, 1], dtype=numpy.int64)
        Z1 = numpy.array([0], dtype=numpy.int64)
        Z2 = numpy.array([3, 2, 1], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([3, 4, 5], dtype=numpy.int64)
        Y = numpy.array([2, 1, 1, 1], dtype=numpy.int64)
        Z1 = numpy.array([0], dtype=numpy.int64)
        Z2 = numpy.array([3, 2, 1], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([2, 16, 1, 1024], dtype=numpy.int64)
        Y = numpy.array([1, 1, 1024, 1024], dtype=numpy.int64)
        Z1 = numpy.array([2], dtype=numpy.int64)
        Z2 = numpy.array([1, 0], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqualArray(Z1, got['Z1'])
        self.assertEqualArray(Z2, got['Z2'])

        X = numpy.array([3, 4, 5], dtype=numpy.int64)
        Y = numpy.array([2, 1, 6, 1], dtype=numpy.int64)
        Z1 = numpy.array([], dtype=numpy.int64)
        Z2 = numpy.array([], dtype=numpy.int64)
        oinf = OnnxInference(model_def)
        self.assertRaise(lambda: oinf.run({'X': X, 'Y': Y}), RuntimeError)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_fused_matmul(self):
        idi = numpy.array([[1, 0], [1, 1]], dtype=numpy.float32)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        Y = numpy.dot(X, idi)

        onx = OnnxFusedMatMul(
            'X', idi, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                outputs={'Y': Y},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi), got['Y'], decimal=5)

        onx = OnnxFusedMatMul(
            'X', idi, transA=1, transB=1, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                outputs={'Y': Y},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi.T), got['Y'], decimal=5)

        onx = OnnxFusedMatMul(
            'X', idi, transA=1, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                outputs={'Y': Y},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi), got['Y'], decimal=5)

        onx = OnnxFusedMatMul(
            'X', idi, transB=1, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                outputs={'Y': Y},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T), got['Y'], decimal=5)

        onx = OnnxFusedMatMul(
            'X', idi, transB=1, output_names=['Y'],
            alpha=numpy.float32(1.),
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                outputs={'Y': Y},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T), got['Y'], decimal=5)

        onx = OnnxFusedMatMul(
            'X', idi, transB=1, output_names=['Y'],
            alpha=numpy.float32(1.),
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                outputs={'Y': Y},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T), got['Y'], decimal=5)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_softmax_grad_13(self):
        G = numpy.array([[-0.1, -0.1, 0.1]], dtype=numpy.float32)
        P = numpy.array([[0.1, 0.3, 0.5]], dtype=numpy.float32)
        Z = numpy.array([[-0.025, -0.015, 0.075]], dtype=numpy.float32)
        onx = OnnxSoftmaxGrad_13(
            'G', 'P', output_names=['Z'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'G': G, 'P': P}, outputs={'Z': Z},
            target_opset=TARGET_OPSET)

        oinf = OnnxInference(model_def)
        got = oinf.run({'G': P, 'P': P})
        self.assertEqualArray(Z, got['Z'], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
