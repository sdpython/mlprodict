"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.special import expit as logistic_sigmoid  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from sklearn.utils.extmath import softmax
from sklearn.utils.testing import ignore_warnings
import skl2onnx
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs, OnnxAdd, OnnxArgMax, OnnxArgMin,
    OnnxArrayFeatureExtractor, OnnxConcat,
    OnnxCeil, OnnxClip, OnnxClip_6,
    OnnxDiv, OnnxEqual, OnnxExp, OnnxFloor, OnnxGreater,
    OnnxGemm, OnnxIdentity, OnnxLog, OnnxMatMul, OnnxMean, OnnxMul,
    OnnxPow, OnnxReciprocal,
    OnnxReduceLogSumExp,
    OnnxReduceMean, OnnxShape,
    OnnxReduceProd, OnnxReduceSum,
    OnnxReduceSumSquare, OnnxReshape,
    OnnxSlice, OnnxSqrt, OnnxSub, OnnxSum,
    OnnxTopK, OnnxTranspose, OnnxRelu,
    OnnxSigmoid, OnnxSoftmax, OnnxSqueeze,
    OnnxConstantOfShape, OnnxNot, OnnxSin,
    OnnxMin, OnnxMax, OnnxSign, OnnxLpNormalization,
    OnnxFlatten, OnnxReduceMax, OnnxReduceMin,
)
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    def common_test_onnxt_runtime_unary(self, onnx_cl, np_fct,
                                        op_version=None, debug=False):
        onx = onnx_cl('X', output_names=['Y'])
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)}, target_opset=op_version)
        if debug:
            print(model_def)
        # no inplace
        oinf = OnnxInference(model_def, inplace=False)
        if debug:
            got = oinf.run({'X': X}, verbose=1, fLOG=print)
        else:
            got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(np_fct(X), got['Y'], decimal=6)
        # inplace
        oinf = OnnxInference(model_def, input_inplace=False, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(np_fct(X), got['Y'], decimal=6)
        # inplace2
        onx2 = OnnxIdentity(onnx_cl('X'), output_names=['Y'])
        model_def2 = onx2.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def2, input_inplace=False, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(np_fct(X), got['Y'], decimal=6)
        # input inplace
        expe = np_fct(X)
        oinf = OnnxInference(model_def, input_inplace=True, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(expe, got['Y'], decimal=6)

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    def common_test_onnxt_runtime_binary(self, onnx_cl, np_fct,
                                         dtype=numpy.float32):
        idi = numpy.identity(2)
        onx = onnx_cl('X', idi, output_names=['Y'])
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(dtype)})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = np_fct(X, idi)
        self.assertEqualArray(exp, got['Y'], decimal=6)

    def test_onnxt_runtime_abs(self):
        self.common_test_onnxt_runtime_unary(OnnxAbs, numpy.abs)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

    def test_onnxt_runtime_argmax(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxArgMax('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=0), got['Y'], decimal=6)

        onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_argmin(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxArgMin('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=0), got['Y'], decimal=6)

        onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                              got['Y'].ravel())

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_array_feature_extractor(self):
        onx = OnnxArrayFeatureExtractor('X', numpy.array([1], dtype=numpy.int64),
                                        output_names=['Y'])
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, (2, 1))
        self.assertEqualArray(X[:, 1:2], got['Y'], decimal=6)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_array_feature_extractor_cmp(self):
        X = numpy.array([3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404],
                        dtype=numpy.float32)
        indices = numpy.array([[4, 2, 0, 1, 3, ],
                               [0, 1, 2, 3, 4],
                               [0, 1, 2, 3, 4],
                               [3, 4, 2, 0, 1],
                               [0, 2, 3, 4, 1]],
                              dtype=numpy.int64)
        onx = OnnxArrayFeatureExtractor('X', indices,
                                        output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_array_feature_extractor_cmp2(self):
        X = numpy.array([[3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404],
                         [-3.3626876, -2.204158, -2.267245, -1.297554, -0.97023404]],
                        dtype=numpy.float32)
        indices = numpy.array([[2], [3]],
                              dtype=numpy.int64)
        onx = OnnxArrayFeatureExtractor('X', indices,
                                        output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    def test_onnxt_runtime_ceil(self):
        self.common_test_onnxt_runtime_unary(OnnxCeil, numpy.ceil)

    def test_onnxt_runtime_clip(self):
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, numpy.array([0], dtype=numpy.float32),
                output_names=output_names),
            lambda x: numpy.clip(x, 0, 1e5))
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, numpy.array([-1000], dtype=numpy.float32),
                numpy.array([0], dtype=numpy.float32),
                output_names=output_names),
            lambda x: numpy.clip(x, -1e5, 0))
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x,
                numpy.array([0.1], dtype=numpy.float32),
                numpy.array([2.1], dtype=numpy.float32),
                output_names=output_names),
            lambda x: numpy.clip(x, 0.1, 2.1))

    def test_onnxt_runtime_clip_10(self):
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip_6(
                x, min=0, max=1e5, output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, 0, 1e5),
            op_version=10)
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, min=0, max=1e5, output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, 0, 1e5),
            op_version=10)
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, max=0, output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, -1e5, 0),
            op_version=10)
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, min=0.1, max=2.1,
                output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, 0.1, 2.1),
            op_version=10)

    def test_onnxt_runtime_concat(self):
        cst = numpy.array([[1, 2]], dtype=numpy.float32)
        onx = OnnxConcat('X', 'Y', cst, output_names=['Z'])
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqual(list(sorted(got)), ['Z'])
        self.assertEqual(got['Z'].shape, (6, 2))
        exp = numpy.vstack([X, Y, cst])
        self.assertEqualArray(exp, got['Z'])

    def test_onnxt_runtime_constant_of_shape(self):
        x = numpy.array([2, 2], dtype=numpy.int64)
        y = numpy.zeros((2, 2))
        onx = OnnxConstantOfShape('X', output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())])
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

    def test_onnxt_runtime_div(self):
        self.common_test_onnxt_runtime_binary(OnnxDiv, lambda x, y: x / y)

    def test_onnxt_runtime_equal(self):
        self.common_test_onnxt_runtime_binary(OnnxEqual, numpy.equal)

    def test_onnxt_runtime_exp(self):
        self.common_test_onnxt_runtime_unary(OnnxExp, numpy.exp)

    def test_onnxt_runtime_flatten(self):
        shape = (2, 3, 4, 5)
        x = numpy.random.random_sample(shape).astype(numpy.float32)

        for i in range(len(shape)):
            node = OnnxFlatten('X', axis=i, output_names='Y')
            model_def = node.to_onnx(
                {'X': x}, outputs=[('Y', FloatTensorType())])
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})['Y']
            new_shape = (
                1, -1) if i == 0 else (numpy.prod(shape[0:i]).astype(int), -1)
            exp = numpy.reshape(x, new_shape)
            self.assertEqualArray(exp, got)

    def test_onnxt_runtime_floor(self):
        self.common_test_onnxt_runtime_unary(OnnxFloor, numpy.floor)

    def test_onnxt_runtime_gemm(self):
        idi = numpy.array([[1, 0], [1, 1]], dtype=numpy.float64)
        cst = numpy.array([4, 5], dtype=numpy.float32)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)

        onx = OnnxGemm('X', idi, cst, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transA=1, transB=1, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi.T) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transA=1, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T) + cst, got['Y'], decimal=6)

    def test_onnxt_runtime_greater(self):
        self.common_test_onnxt_runtime_binary(OnnxGreater, numpy.greater)

    def test_onnxt_runtime_identity(self):
        self.common_test_onnxt_runtime_unary(OnnxIdentity, lambda x: x)

    def test_onnxt_runtime_log(self):
        self.common_test_onnxt_runtime_unary(OnnxLog, numpy.log)

    def test_onnxt_runtime_lp_normalization(self):
        onx = OnnxLpNormalization('X', output_names=['Y'], p=2, axis=1)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        exp = numpy.array([[0.4472136, 0.8944272],
                           [0.6, -0.8]], dtype=numpy.float32)
        self.assertEqualArray(got['Y'], exp)

        onx = OnnxLpNormalization('X', output_names=['Y'], p=2, axis=0)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        exp = numpy.array([[0.3162278, 0.4472136],
                           [0.9486833, -0.8944272]], dtype=numpy.float32)
        self.assertEqualArray(got['Y'], exp)

    def test_onnxt_runtime_matmul(self):
        self.common_test_onnxt_runtime_binary(OnnxMatMul, lambda x, y: x @ y)

    def test_onnxt_runtime_max(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMax, lambda x, y: numpy.maximum(x, y))

    def test_onnxt_runtime_mean(self):
        idi = numpy.identity(2, dtype=numpy.float64)
        onx = OnnxMean('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray((idi + X) / 2, got['Y'], decimal=6)

    def test_onnxt_runtime_min(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMin, lambda x, y: numpy.minimum(x, y))

    def test_onnxt_runtime_mul(self):
        self.common_test_onnxt_runtime_binary(OnnxMul, lambda x, y: x * y)

    def test_onnxt_runtime_not(self):
        self.common_test_onnxt_runtime_unary(OnnxNot, numpy.logical_not)

    def test_onnxt_runtime_pow(self):
        self.common_test_onnxt_runtime_binary(OnnxPow, numpy.power)

    def test_onnxt_runtime_reciprocal(self):
        self.common_test_onnxt_runtime_unary(OnnxReciprocal, numpy.reciprocal)

    def test_onnxt_runtime_reduce_log_sum_exp(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceLogSumExp('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X)))
        self.assertEqualArray(res, got['Y'], decimal=6)

        onx = OnnxReduceLogSumExp('X', output_names=['Y'], axes=[1])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1))
        self.assertEqualArray(res, got['Y'].ravel())

        onx = OnnxReduceLogSumExp(
            'X', output_names=['Y'], axes=[1], keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1, keepdims=1))
        self.assertEqualArray(res.ravel(), got['Y'].ravel())

    def test_onnxt_runtime_reduce_max(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMax('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, keepdims=False, axis=None),  # pylint: disable=E1101
                              got['Y'], decimal=6)

        onx = OnnxReduceMax('X', output_names=['Y'], axes=[1])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, axis=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

        onx = OnnxReduceMax('X', output_names=['Y'], axes=[1], keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, axis=1, keepdims=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

    def test_onnxt_runtime_reduce_mean(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMean('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X), got['Y'], decimal=6)

        onx = OnnxReduceMean('X', output_names=['Y'], axes=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceMean('X', output_names=['Y'], axes=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_reduce_min(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMin('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, keepdims=False, axis=None),  # pylint: disable=E1101
                              got['Y'], decimal=6)

        onx = OnnxReduceMin('X', output_names=['Y'], axes=[1])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, axis=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

        onx = OnnxReduceMin('X', output_names=['Y'], axes=[1], keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, axis=1, keepdims=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

    def test_onnxt_runtime_reduce_prod(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceProd('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X), got['Y'], decimal=6)

        onx = OnnxReduceProd('X', output_names=['Y'], axes=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceProd('X', output_names=['Y'], axes=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_reduce_sum(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSum('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X), got['Y'], decimal=6)

        onx = OnnxReduceSum('X', output_names=['Y'], axes=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSum('X', output_names=['Y'], axes=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_reduce_sum_square(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X)), got['Y'], decimal=6)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_relu(self):
        self.common_test_onnxt_runtime_unary(
            OnnxRelu, lambda x: numpy.maximum(x, 0))

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    def common_test_onnxt_runtime_reshape(self):
        sh = numpy.array([1, 4], dtype=numpy.int64)
        onx = OnnxReshape('X', sh, output_names=['Y'])
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = X.reshape(sh.tolist())
        self.assertEqualArray(exp, got['Y'])

    def test_onnxt_runtime_shape(self):
        x = numpy.random.randn(20, 2).astype(numpy.float32)
        y = x.shape
        onx = OnnxShape('X', output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

    def test_onnxt_runtime_sigmoid(self):
        self.common_test_onnxt_runtime_unary(OnnxSigmoid, logistic_sigmoid)

    def test_onnxt_runtime_sign(self):
        self.common_test_onnxt_runtime_unary(OnnxSign, numpy.sign)

    def test_onnxt_runtime_sin(self):
        self.common_test_onnxt_runtime_unary(OnnxSin, numpy.sin)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_slice(self):
        x = numpy.random.randn(20, 10, 5).astype(numpy.float32)
        y = x[0:3, 0:10]
        starts = numpy.array([0, 0], dtype=numpy.int64)
        ends = numpy.array([3, 10], dtype=numpy.int64)
        onx = OnnxSlice('X', starts, ends, output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        x = numpy.random.randn(20, 10, 5).astype(numpy.float32)
        y = x[0:3, 0:10]
        starts = numpy.array([0, 0], dtype=numpy.int64)
        ends = numpy.array([3, 10], dtype=numpy.int64)
        axes = numpy.array([0, 1], dtype=numpy.int64)
        onx = OnnxSlice('X', starts, ends, axes, output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        x = numpy.random.randn(20, 10, 5).astype(numpy.float32)
        y = x[0:3:-1, 0:10:2]
        starts = numpy.array([0, 0], dtype=numpy.int64)
        ends = numpy.array([3, 10], dtype=numpy.int64)
        axes = numpy.array([0, 1], dtype=numpy.int64)
        steps = numpy.array([-1, 2], dtype=numpy.int64)
        onx = OnnxSlice('X', starts, ends, axes, steps, output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

    def test_onnxt_runtime_sqrt(self):
        self.common_test_onnxt_runtime_unary(OnnxSqrt, numpy.sqrt)

    def test_onnxt_runtime_squeeze(self):
        x = numpy.random.randn(20, 1).astype(numpy.float32)
        y = numpy.squeeze(x)
        onx = OnnxSqueeze('X', axes=[1], output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        x = numpy.random.randn(1, 20).astype(numpy.float32)
        y = numpy.squeeze(x)
        onx = OnnxSqueeze('X', axes=[0], output_names=['Y'])
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

    def test_onnxt_runtime_softmax(self):
        self.common_test_onnxt_runtime_unary(OnnxSoftmax, softmax)

    def test_onnxt_runtime_sub(self):
        self.common_test_onnxt_runtime_binary(OnnxSub, lambda x, y: x - y)

    def test_onnxt_runtime_sum(self):
        self.common_test_onnxt_runtime_binary(OnnxSum, lambda x, y: x + y)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_topk(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        # axis=1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64),
                       axis=1, output_names=['Y', 'Yi'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        exp = numpy.array([[4., 3.],
                           [5., 4.],
                           [5., 2.]],
                          dtype=numpy.float32)
        self.assertEqualArray(exp, got['Y'])
        exp = numpy.array([[4, 3],
                           [4, 3],
                           [3, 0]],
                          dtype=numpy.int64)
        self.assertEqualArray(exp, got['Yi'])

        # axis=0, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64),
                       axis=0, output_names=['Y', 'Yi'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        exp = numpy.array([[2., 1., 2., 5., 5.],
                           [1., -1., -2., 4., 4.]],
                          dtype=numpy.float32)
        self.assertEqualArray(exp, got['Y'])

        # axis=-1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64),
                       axis=-1, output_names=['Y', 'Yi'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        exp = numpy.array([[4., 3.],
                           [5., 4.],
                           [5., 2.]],
                          dtype=numpy.float32)
        self.assertEqualArray(exp, got['Y'])

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_topk2(self):
        X = numpy.array([[-0., -0.08000002, -2., -2.88000023]],
                        dtype=numpy.float32)

        # axis=-1, k=-1
        onx = OnnxTopK('X', numpy.array([1], dtype=numpy.int64),
                       axis=1, output_names=['Y', 'Yi'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        exp = numpy.array([[0.]],
                          dtype=numpy.float32)
        self.assertEqualArray(exp, got['Y'])
        exp = numpy.array([[0.]],
                          dtype=numpy.int64)
        self.assertEqualArray(exp, got['Yi'])

    def test_onnxt_runtime_transpose(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        onx = OnnxTranspose('X', perm=[0, 1], output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X, got['Y'])

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        onx = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X.T, got['Y'])


if __name__ == "__main__":
    TestOnnxrtPythonRuntime().test_onnxt_runtime_clip_10()
    unittest.main()
