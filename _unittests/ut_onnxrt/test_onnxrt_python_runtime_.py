"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
from contextlib import redirect_stdout
from io import StringIO
import numpy
from scipy.sparse import coo_matrix, csr_matrix, SparseEfficiencyWarning
from scipy.special import (  # pylint: disable=E0611
    expit as logistic_sigmoid,
    erf)
from scipy.spatial.distance import cdist
from onnx import TensorProto
from onnx.helper import make_sparse_tensor, make_tensor
from onnx.defs import onnx_opset_version
from pyquickhelper.pycode import ExtTestCase
from sklearn.utils.extmath import softmax
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs, OnnxAdd, OnnxArgMax, OnnxArgMin, OnnxAtan,
    OnnxBatchNormalization,
    OnnxConcat, OnnxConv,
    OnnxCeil, OnnxClip, OnnxConstant, OnnxConstantOfShape,
    OnnxDequantizeLinear,
    OnnxDiv,
    OnnxEinsum, OnnxEqual, OnnxErf, OnnxExp, OnnxEyeLike,
    OnnxFlatten, OnnxFloor,
    OnnxGreater, OnnxGemm, OnnxGlobalAveragePool,
    OnnxIdentity, OnnxIsNaN,
    OnnxLog, OnnxLpNormalization,
    OnnxMatMul, OnnxMax, OnnxMean, OnnxMin, OnnxMul,
    OnnxNeg, OnnxNot,
    OnnxPow,
    OnnxQuantizeLinear,
    OnnxReciprocal,
    OnnxReduceLogSumExp, OnnxReduceMax, OnnxReduceMean, OnnxReduceMin,
    OnnxReduceProd, OnnxReduceSum, OnnxReduceSumSquare,
    OnnxRelu, OnnxReshape,
    OnnxShape, OnnxSlice, OnnxSigmoid, OnnxSign, OnnxSin,
    OnnxSoftmax, OnnxSqueeze,
    OnnxSqrt, OnnxSub, OnnxSum,
    OnnxTopK, OnnxTranspose,
)
try:
    from skl2onnx.algebra.onnx_ops import OnnxCelu
except ImportError:
    OnnxCelu = None
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import (
    get_opset_number_from_onnx, get_ir_version_from_onnx)
from mlprodict.onnxrt.validate.validate_python import validate_python_inference
from mlprodict.onnxrt.ops_cpu.op_batch_normalization import _batchnorm_test_mode
from mlprodict.onnxrt.ops_cpu.op_global_average_pool import _global_average_pool
from mlprodict.onnxrt.ops_cpu._op_onnx_numpy import (  # pylint: disable=E0611
    topk_element_min_double, topk_element_max_double, topk_element_fetch_double,
    topk_element_min_float, topk_element_max_float, topk_element_fetch_float,
    topk_element_min_int64, topk_element_max_int64, topk_element_fetch_int64)
from mlprodict.onnxrt.ops_cpu.op_celu import _vcelu1, pycelu
from mlprodict.onnxrt.ops_cpu.op_topk import topk_sorted_implementation


sparse_support = []
sparse_no_numpy = []
python_tested = []


def make_coo_matrix(*args, **kwargs):
    coo = coo_matrix(*args, **kwargs)
    coo.row = coo.row.astype(numpy.int64)
    coo.col = coo.col.astype(numpy.int64)
    return coo


class TestOnnxrtPythonRuntime(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        if __name__ == "__main__":
            import pprint
            print('-----------')
            pprint.pprint(sparse_support)
            print('-----------')
            pprint.pprint(sparse_no_numpy)
            print('-----------')
            pprint.pprint(
                list(sorted({_.__name__ for _ in python_tested})))
            print('-----------')

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning,
                               SparseEfficiencyWarning, PendingDeprecationWarning))
    def common_test_onnxt_runtime_unary(self, onnx_cl, np_fct,
                                        op_version=None,
                                        outputs=None, debug=False):
        if op_version is None:
            op_version = get_opset_number_from_onnx()
        try:
            onx = onnx_cl('X', output_names=['Y'], op_version=op_version)
        except RuntimeError as e:
            raise RuntimeError('onnx.opset={} op_version={}'.format(
                get_opset_number_from_onnx(), op_version)) from e
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx(
            {'X': X.astype(numpy.float32)}, target_opset=op_version,
            outputs=outputs)
        if debug:
            print(model_def)

        # python code
        python_tested.append(onnx_cl)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(numpy.float32)})

        # no inplace
        oinf = OnnxInference(model_def, inplace=False)
        all_names = "\n".join(
            "%s>=v%d" % (op.ops_.__class__.__name__,
                         op.ops_._schema.since_version)  # pylint: disable=W0212
            for op in oinf.sequence_)
        if debug:
            got = oinf.run({'X': X}, verbose=1, fLOG=print)
        else:
            got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        try:
            self.assertEqualArray(np_fct(X), got['Y'], decimal=6)
        except AssertionError as e:
            raise AssertionError(
                'onnx.opset={} op_version={}\n--ONNX--\n{}\n--NAMES--\n{}'.format(
                    get_opset_number_from_onnx(), op_version, model_def,
                    all_names)) from e

        # inplace
        oinf = OnnxInference(model_def, input_inplace=False, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(np_fct(X), got['Y'], decimal=6)
        # inplace2
        onx2 = OnnxIdentity(
            onnx_cl('X', op_version=op_version),
            output_names=['Y'], op_version=op_version)
        model_def2 = onx2.to_onnx(
            {'X': X.astype(numpy.float32)}, target_opset=op_version,
            outputs=outputs)
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

        # sparse
        row = numpy.array([0, 0, 1, 3, 1])
        col = numpy.array([0, 2, 1, 3, 1])
        data = numpy.array([1, 1, 1, 1, 1])
        X = make_coo_matrix((data, (row.astype(numpy.int64), col.astype(numpy.int64))),
                            shape=(4, 4), dtype=numpy.float32)
        try:
            exp = np_fct(X)
        except (TypeError, NotImplementedError, ValueError) as e:
            # Function np_fct does not work on sparse data.
            sparse_no_numpy.append((onnx_cl.__name__, op_version, e))
            return

        model_def_sparse = onx.to_onnx(
            {'X': X.astype(numpy.float32)}, target_opset=op_version)
        oinf = OnnxInference(
            model_def_sparse, input_inplace=False, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualSparseArray(exp, got['Y'], decimal=6)
        sparse_support.append(('UnOp', op_version, onnx_cl.__name__))

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning,
                               SparseEfficiencyWarning, PendingDeprecationWarning))
    def common_test_onnxt_runtime_binary(self, onnx_cl, np_fct,
                                         dtype=numpy.float32,
                                         op_version=None, debug=False):
        if op_version is None:
            op_version = get_opset_number_from_onnx()
        idi = numpy.identity(2, dtype=dtype)
        onx = onnx_cl('X', idi, output_names=['Y'], op_version=op_version)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=op_version)
        oinf = OnnxInference(model_def)
        if debug:
            got = oinf.run({'X': X.astype(dtype)}, verbose=1, fLOG=print)
        else:
            got = oinf.run({'X': X.astype(dtype)})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = np_fct(X, idi)
        self.assertEqualArray(exp, got['Y'], decimal=6)

        # python code
        python_tested.append(onnx_cl)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(dtype)})

        # sparse
        idi = make_coo_matrix(numpy.identity(2)).astype(numpy.float32)
        X = make_coo_matrix(numpy.array(
            [[0, 2], [3, -4]], dtype=numpy.float32))
        try:
            exp = np_fct(X, idi)
        except (TypeError, NotImplementedError, ValueError) as e:
            # Function np_fct does not work on sparse data.
            sparse_no_numpy.append((onnx_cl.__name__, op_version, e))
            return

        onx = onnx_cl('X', idi, output_names=['Y'], op_version=op_version)
        model_def_sparse = onx.to_onnx({'X': X}, target_opset=op_version)
        try:
            oinf = OnnxInference(
                model_def_sparse, input_inplace=False, inplace=True)
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to load sparse model\n{}".format(
                    model_def_sparse)) from e
        if debug:
            got = oinf.run({'X': X}, verbose=1, fLOG=print)
        else:
            got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        if isinstance(exp, (coo_matrix, csr_matrix)):
            self.assertEqualSparseArray(exp, got['Y'], decimal=6)
        elif isinstance(exp, numpy.ndarray):
            self.assertEqualArray(exp, got['Y'], decimal=6)
        else:
            self.assertEqual(exp, got['Y'])
        sparse_support.append(('BinOp', op_version, onnx_cl.__name__))

    def test_onnxt_runtime_abs(self):
        self.common_test_onnxt_runtime_unary(OnnxAbs, numpy.abs)

    def test_onnxt_runtime_abs_debug(self):
        f = StringIO()
        with redirect_stdout(f):
            self.common_test_onnxt_runtime_unary(
                OnnxAbs, numpy.abs, debug=True)

    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

    def test_onnxt_runtime_argmax(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxArgMax('X', output_names=['Y'], keepdims=0,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=0), got['Y'], decimal=6)

        python_tested.append(OnnxArgMax)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(numpy.float32)})

        onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=0,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=1,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                              got['Y'].ravel())

        # sparse
        X = make_coo_matrix(X, dtype=numpy.float32)
        try:
            exp = numpy.argmax(X, axis=1)
        except (TypeError, NotImplementedError, ValueError) as e:
            # Function np_fct does not work on sparse data.
            sparse_no_numpy.append((OnnxArgMax.__name__, None, e))
            return

        model_def_sparse = onx.to_onnx({'X': X},
                                       target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def_sparse, input_inplace=False)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(exp, got['Y'], decimal=6)
        sparse_support.append(('UnOp', None, OnnxArgMax.__name__))
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

    @unittest.skipIf(onnx_opset_version() < 12, reason="needs onnx 1.7.0")
    def test_onnxt_runtime_argmax_12(self):
        self.assertGreater(onnx_opset_version(), 12)
        from skl2onnx.algebra.onnx_ops import OnnxArgMax_12  # pylint: disable=E0611
        X = numpy.array([[2, 2, 1], [0, 1, 1]], dtype=float)
        onx = OnnxArgMax_12('X', output_names=['Y'], keepdims=0, axis=1,
                            select_last_index=1, op_version=12)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.array([1, 2], dtype=numpy.int64),
                              got['Y'], decimal=6)

    def test_onnxt_runtime_argmin(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxArgMin('X', output_names=['Y'], keepdims=0,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=0), got['Y'], decimal=6)

        python_tested.append(OnnxArgMin)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(numpy.float32)})

        onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=0,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=1,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                              got['Y'].ravel())

        # sparse
        X = make_coo_matrix(X, dtype=numpy.float32)
        try:
            exp = numpy.argmin(X, axis=1)
        except (TypeError, NotImplementedError, ValueError) as e:
            # Function np_fct does not work on sparse data.
            sparse_no_numpy.append((OnnxArgMin.__name__, None, e))
            return

        model_def_sparse = onx.to_onnx({'X': X},
                                       target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def_sparse, input_inplace=False)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(exp, got['Y'], decimal=6)
        sparse_support.append(('UnOp', None, OnnxArgMin.__name__))

    @unittest.skipIf(onnx_opset_version() < 12, reason="needs onnx 1.7.0")
    def test_onnxt_runtime_argmin_12(self):
        self.assertGreater(onnx_opset_version(), 12)
        from skl2onnx.algebra.onnx_ops import OnnxArgMin_12  # pylint: disable=E0611
        X = numpy.array([[2, 1, 1], [0, 0, 1]], dtype=float)
        onx = OnnxArgMin_12('X', output_names=['Y'], keepdims=0, axis=1,
                            select_last_index=1, op_version=12)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.array([2, 1], dtype=numpy.int64),
                              got['Y'], decimal=6)

    def test_onnxt_runtime_atan(self):
        self.common_test_onnxt_runtime_unary(OnnxAtan, numpy.arctan)

    def test_onnxt_runtime_atan2(self):
        test_pairs = [[y, x]
                      for x in [3., -4., 0., -1., 1.]
                      for y in [5., -6., 0., -1., 1.]]
        y_val = numpy.array([y for y, x in test_pairs], dtype=numpy.float32)
        x_val = numpy.array([x for y, x in test_pairs], dtype=numpy.float32)

        def atan2(y, x):
            # size: 100000
            # timeit arctan: 0.00205
            # timeit arctan2: 0.00361
            # timeit atan2: 0.00599
            sx = numpy.sign(x)
            sy = numpy.sign(y)
            pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-numpy.pi / 2)
            atan_part = numpy.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
            return atan_part + pi_part

        self.assertEqualArray(
            numpy.arctan2(y_val, x_val), atan2(y_val, x_val))

    def test_onnxt_runtime_batch_normalization(self):
        # input size: (1, 2, 1, 3)
        x = numpy.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(numpy.float32)
        s = numpy.array([1.0, 1.5]).astype(numpy.float32)
        bias = numpy.array([0, 1]).astype(numpy.float32)
        mean = numpy.array([0, 3]).astype(numpy.float32)
        var = numpy.array([1, 1.5]).astype(numpy.float32)
        y = _batchnorm_test_mode(x, s, bias, mean, var).astype(numpy.float32)

        onx = OnnxBatchNormalization(
            'X', s, bias, mean, var, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y, got['Y'])

        # input size: (2, 3, 4, 5)
        x = numpy.random.randn(2, 3, 4, 5).astype(numpy.float32)
        s = numpy.random.randn(3).astype(numpy.float32)
        bias = numpy.random.randn(3).astype(numpy.float32)
        mean = numpy.random.randn(3).astype(numpy.float32)
        var = numpy.random.rand(3).astype(numpy.float32)
        epsilon = 1e-2
        y = _batchnorm_test_mode(
            x, s, bias, mean, var, epsilon).astype(numpy.float32)

        onx = OnnxBatchNormalization(
            'X', s, bias, mean, var,
            output_names=['Y'], epsilon=epsilon,
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxBatchNormalization)

    def test_onnxt_runtime_ceil(self):
        self.common_test_onnxt_runtime_unary(OnnxCeil, numpy.ceil)

    @unittest.skipIf(OnnxCelu is None, reason="onnx too recent")
    def test_onnxt_runtime_celu1(self):
        self.common_test_onnxt_runtime_unary(
            OnnxCelu, _vcelu1, op_version=12,
            outputs=[('Y', FloatTensorType([None, 2]))])

    @unittest.skipIf(OnnxCelu is None, reason="onnx too recent")
    def test_onnxt_runtime_celu2(self):
        _vcelu2 = numpy.vectorize(
            lambda x: pycelu(x, 1.), otypes=[numpy.float])
        self.common_test_onnxt_runtime_unary(
            OnnxCelu, _vcelu2, op_version=12,
            outputs=[('Y', FloatTensorType([None, 2]))])

    @unittest.skipIf(onnx_opset_version() < 11,
                     reason="Explicitely tests Clip >= 11")
    def test_onnxt_runtime_clip(self):
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=None: OnnxClip(
                x, numpy.array([0], dtype=numpy.float32),
                output_names=output_names, op_version=op_version),
            lambda x: numpy.clip(x, 0, 1e5))
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=None: OnnxClip(
                x, numpy.array([-1000], dtype=numpy.float32),
                numpy.array([0], dtype=numpy.float32),
                op_version=op_version,
                output_names=output_names),
            lambda x: numpy.clip(x, -1e5, 0))
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=None: OnnxClip(
                x,
                numpy.array([0.1], dtype=numpy.float32),
                numpy.array([2.1], dtype=numpy.float32),
                output_names=output_names,
                op_version=op_version),
            lambda x: numpy.clip(x, 0.1, 2.1))
        python_tested.append(OnnxClip)

    def test_onnxt_runtime_clip_10(self):
        from skl2onnx.algebra.onnx_ops import OnnxClip_6  # pylint: disable=E0611
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=10: OnnxClip_6(
                x, min=1e-5, max=1e5, output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, 1e-5, 1e5),
            op_version=10)
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=10: OnnxClip(
                x, min=1e-5, max=1e5, output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, 1e-5, 1e5),
            op_version=10)
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=10: OnnxClip(
                x, max=1e-5, output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, -1e5, 1e-5),
            op_version=10)
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None, op_version=10: OnnxClip(
                x, min=0.1, max=2.1,
                output_names=output_names,
                op_version=10),
            lambda x: numpy.clip(x, 0.1, 2.1),
            op_version=10)

    def test_onnxt_runtime_concat(self):
        cst = numpy.array([[1, 2]], dtype=numpy.float32)
        onx = OnnxConcat('X', 'Y', cst, output_names=['Z'],
                         op_version=get_opset_number_from_onnx())
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        self.assertEqual(list(sorted(got)), ['Z'])
        self.assertEqual(got['Z'].shape, (6, 2))
        exp = numpy.vstack([X, Y, cst])
        self.assertEqualArray(exp, got['Z'])

        python_tested.append(OnnxConstantOfShape)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(
            oinfpy, {'X': X.astype(numpy.float32),
                     'Y': Y.astype(numpy.float32)})
        python_tested.append(OnnxConcat)

    def test_onnxt_runtime_constant_of_shape(self):
        x = numpy.array([2, 2], dtype=numpy.int64)
        y = numpy.zeros((2, 2))
        onx = OnnxConstantOfShape('X', output_names=['Y'],
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        python_tested.append(OnnxConstantOfShape)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': x})

    def test_onnxt_runtime_conv(self):
        x = numpy.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                           [5., 6., 7., 8., 9.],
                           [10., 11., 12., 13., 14.],
                           [15., 16., 17., 18., 19.],
                           [20., 21., 22., 23., 24.]]]]).astype(numpy.float32)
        W = numpy.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                           [1., 1., 1.],
                           [1., 1., 1.]]]]).astype(numpy.float32)

        y_with_padding = numpy.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                                        [33., 54., 63., 72., 51.],
                                        [63., 99., 108., 117., 81.],
                                        [93., 144., 153., 162., 111.],
                                        [72., 111., 117., 123., 84.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

        python_tested.append(OnnxConv)

    def test_onnxt_runtime_cum_sum(self):
        from skl2onnx.algebra.onnx_ops import OnnxCumSum  # pylint: disable=E0611

        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 10., 15.]).astype(numpy.float64)
        onx = OnnxCumSum('X', 'axis', output_names=['Y'],
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x, 'axis': axis})
        self.assertEqualArray(exp, got['Y'])

        python_tested.append(OnnxCumSum)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': x, 'axis': axis})

        # reverse = 1
        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([15., 14., 12., 9., 5.]).astype(numpy.float64)
        onx = OnnxCumSum('X', 'axis', output_names=['Y'], reverse=1,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=get_opset_number_from_onnx())
        try:
            got = OnnxInference(model_def).run({'X': x, 'axis': axis})
            self.assertEqualArray(exp, got['Y'])
        except NotImplementedError:
            pass

        # exclusive = 1
        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([0., 1., 3., 6., 10.]).astype(numpy.float64)
        onx = OnnxCumSum('X', 'axis', output_names=['Y'], exclusive=1,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=get_opset_number_from_onnx())
        try:
            got = OnnxInference(model_def).run({'X': x, 'axis': axis})
            self.assertEqualArray(exp, got['Y'])
        except NotImplementedError:
            pass

        # 2d axis = 0
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float64).reshape((2, 3))
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([1., 2., 3., 5., 7., 9.]).astype(
            numpy.float64).reshape((2, 3))
        onx = OnnxCumSum('X', 'axis', output_names=['Y'],
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x, 'axis': axis})
        self.assertEqualArray(exp, got['Y'])

        # 2d axis = 1
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float64).reshape((2, 3))
        axis = numpy.array([-1]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 4., 9., 15.]).astype(
            numpy.float64).reshape((2, 3))
        onx = OnnxCumSum('X', 'axis', output_names=['Y'],
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x, 'axis': axis})
        self.assertEqualArray(exp, got['Y'])

        # 2d axis = 1, reverse
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float64).reshape((2, 3))
        axis = numpy.array([-1]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 4., 9., 15.]).astype(
            numpy.float64).reshape((2, 3))
        onx = OnnxCumSum('X', 'axis', output_names=['Y'], reverse=1,
                         op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=get_opset_number_from_onnx())
        try:
            got = OnnxInference(model_def).run({'X': x, 'axis': axis})
            self.assertEqualArray(exp, got['Y'])
        except NotImplementedError:
            pass

        # no axis
        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 10., 15.]).astype(numpy.float64)
        try:
            onx = OnnxCumSum('X', output_names=['Y'],
                             op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx(
                {'X': x}, outputs=[('Y', DoubleTensorType())],
                target_opset=get_opset_number_from_onnx())
            got = OnnxInference(model_def).run({'X': x})
            self.assertEqualArray(exp, got['Y'])
        except RuntimeError:
            pass

        # reverse = 1
        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([15., 14., 12., 9., 5.]).astype(numpy.float64)
        try:
            onx = OnnxCumSum('X', output_names=['Y'], reverse=1,
                             op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx(
                {'X': x}, outputs=[('Y', DoubleTensorType())],
                target_opset=get_opset_number_from_onnx())
            got = OnnxInference(model_def).run({'X': x})
            self.assertEqualArray(exp, got['Y'])
        except RuntimeError:
            pass

    def test_onnxt_runtime_dequantize_linear(self):
        X = numpy.array([[[[3, 89], [34, 200], [74, 59]],
                          [[5, 24], [24, 87], [32, 13]],
                          [[245, 99], [4, 142], [121, 102]], ], ],
                        dtype=numpy.uint8)
        x_scale = numpy.array([2, 4, 5], dtype=numpy.float32)
        x_zero_point = numpy.array([84, 24, 196], dtype=numpy.uint8)
        exp = ((X.astype(numpy.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(numpy.float32)) *
               x_scale.reshape(1, 3, 1, 1))
        onx = OnnxDequantizeLinear(
            'X', x_scale, x_zero_point, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        X = numpy.array([0, 3, 128, 255]).astype(numpy.uint8)
        x_scale = numpy.float32(2)
        x_zero_point = numpy.uint8(128)
        exp = numpy.array([-256, -250, 0, 254], dtype=numpy.float32)
        onx = OnnxDequantizeLinear(
            'X', x_scale, x_zero_point, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxDequantizeLinear)

    def test_onnxt_runtime_div(self):
        self.common_test_onnxt_runtime_binary(OnnxDiv, lambda x, y: x / y)

    def test_onnxt_runtime_einsum(self):
        X = numpy.random.randn(5, 2, 3).astype(numpy.float32)
        Y = numpy.random.randn(5, 3, 4).astype(numpy.float32)
        equation = 'bij, bjk -> bik'
        onx = OnnxEinsum(
            'X', 'Y', equation=equation, output_names=['Z'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        exp = numpy.einsum(equation, X, Y)
        self.assertEqualArray(exp, got['Z'])
        python_tested.append(OnnxEinsum)

    def test_onnxt_runtime_eyelike(self):
        onx = OnnxEyeLike('X', k=0, output_names=['Y'])
        X = numpy.array([2, 2], dtype=numpy.int64)
        model_def = onx.to_onnx({'X': X.astype(numpy.int64)},
                                target_opset=get_opset_number_from_onnx(),
                                outputs=[('Y', FloatTensorType())])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = numpy.eye(*X, k=0)
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxEyeLike)

    def test_onnxt_runtime_equal(self):
        self.common_test_onnxt_runtime_binary(OnnxEqual, numpy.equal)

    def test_onnxt_runtime_erf(self):
        self.common_test_onnxt_runtime_unary(OnnxErf, erf)

    def test_onnxt_runtime_exp(self):
        self.common_test_onnxt_runtime_unary(OnnxExp, numpy.exp)

    def test_onnxt_runtime_flatten(self):
        shape = (2, 3, 4, 5)
        x = numpy.random.random_sample(shape).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101

        for i in range(len(shape)):
            node = OnnxFlatten('X', axis=i, output_names='Y',
                               op_version=get_opset_number_from_onnx())
            model_def = node.to_onnx(
                {'X': x}, outputs=[('Y', FloatTensorType())],
                target_opset=get_opset_number_from_onnx())
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})['Y']
            new_shape = ((1, -1) if i == 0
                         else (numpy.prod(shape[0:i]).astype(int), -1))
            exp = numpy.reshape(x, new_shape)
            self.assertEqualArray(exp, got)

            python_tested.append(OnnxFlatten)
            oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
            validate_python_inference(oinfpy, {'X': x})

    def test_onnxt_runtime_floor(self):
        self.common_test_onnxt_runtime_unary(OnnxFloor, numpy.floor)

    def test_onnxt_runtime_gather_elements(self):
        from skl2onnx.algebra.onnx_ops import OnnxGatherElements  # pylint: disable=E0611
        # ex 1
        data = numpy.array([[1, 2],
                            [3, 4]], dtype=numpy.float32)
        indices = numpy.array([[0, 0],
                               [1, 0]], dtype=numpy.int64)

        onx = OnnxGatherElements('X', 'Y', output_names=['Z'], axis=1,
                                 op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': data, 'Y': indices})
        exp = numpy.array([[1, 1],
                           [4, 3]], dtype=numpy.float32)
        self.assertEqual(exp, got['Z'])

        python_tested.append(OnnxGatherElements)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': data, 'Y': indices})

        # ex 2
        data = numpy.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]], dtype=numpy.float32)
        indices = numpy.array([[1, 2, 0],
                               [2, 0, 0]], dtype=numpy.int32)

        onx = OnnxGatherElements('X', 'Y', output_names=['Z'], axis=0,
                                 op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': data, 'Y': indices})
        exp = numpy.array([[4, 8, 3],
                           [7, 2, 3]], dtype=numpy.float32)
        self.assertEqual(exp, got['Z'])

    def test_onnxt_runtime_gemm_python(self):
        self.do_test_onnxt_runtime_gemm("python")
        python_tested.append(OnnxGemm)

    def test_onnxt_runtime_gemm_onnxruntime(self):
        self.do_test_onnxt_runtime_gemm("onnxruntime1")

    def do_test_onnxt_runtime_gemm(self, runtime):
        idi = numpy.array([[1, 0], [1, 1]], dtype=numpy.float32)
        cst = numpy.array([4, 5], dtype=numpy.float32)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)

        onx = OnnxGemm('X', idi, cst, output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version_from_onnx()
        try:
            oinf = OnnxInference(model_def, runtime=runtime)
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to instantiate (runtime='{}')\n{}".format(
                    runtime, model_def)) from e
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transA=1, transB=1, output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version_from_onnx()
        try:
            oinf = OnnxInference(model_def, runtime=runtime)
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to instantiate (runtime='{}')\n{}".format(
                    runtime, model_def)) from e
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi.T) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transA=1, output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime=runtime)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime=runtime)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'],
                       alpha=numpy.float32(1.),
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime=runtime)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T) + cst, got['Y'], decimal=6)

        if runtime != 'onnxruntime1':
            onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'],
                           alpha=numpy.float32(1.),
                           op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx({'X': idi.astype(numpy.float64)},
                                    target_opset=get_opset_number_from_onnx())
            if 'onnxruntime' in runtime:
                model_def.ir_version = get_ir_version_from_onnx()
            oinf = OnnxInference(model_def, runtime=runtime)
            got = oinf.run({'X': X.astype(numpy.float32)})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(numpy.dot(X, idi.T) +
                                  cst, got['Y'], decimal=6)

    def test_onnxt_runtime_global_average_pool(self):
        x = x = numpy.random.randn(1, 3, 5, 5).astype(numpy.float32)
        y = _global_average_pool(x).astype(numpy.float32)

        onx = OnnxGlobalAveragePool(
            'X', output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y, got['Y'])

        x = numpy.array([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]]]).astype(numpy.float32)
        y = numpy.array([[[[5]]]]).astype(numpy.float32)
        onx = OnnxGlobalAveragePool(
            'X', output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y, got['Y'])

        python_tested.append(OnnxGlobalAveragePool)

    def test_onnxt_runtime_greater(self):
        self.common_test_onnxt_runtime_binary(OnnxGreater, numpy.greater)

    def test_onnxt_runtime_identity(self):
        self.common_test_onnxt_runtime_unary(OnnxIdentity, lambda x: x)

    def test_onnxt_runtime_isnan(self):
        self.common_test_onnxt_runtime_unary(OnnxIsNaN, numpy.isnan)

    def test_onnxt_runtime_log(self):
        self.common_test_onnxt_runtime_unary(OnnxLog, numpy.log)

    def test_onnxt_runtime_lp_normalization(self):
        onx = OnnxLpNormalization('X', output_names=['Y'], p=2, axis=1,
                                  op_version=get_opset_number_from_onnx())
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        exp = numpy.array([[0.4472136, 0.8944272],
                           [0.6, -0.8]], dtype=numpy.float32)
        self.assertEqualArray(got['Y'], exp)

        onx = OnnxLpNormalization('X', output_names=['Y'], p=2, axis=0,
                                  op_version=get_opset_number_from_onnx())
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        exp = numpy.array([[0.3162278, 0.4472136],
                           [0.9486833, -0.8944272]], dtype=numpy.float32)
        self.assertEqualArray(got['Y'], exp)
        python_tested.append(OnnxLpNormalization)

    def test_onnxt_runtime_matmul(self):
        self.common_test_onnxt_runtime_binary(OnnxMatMul, lambda x, y: x @ y)

    def test_onnxt_runtime_max(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMax, lambda x, y: numpy.maximum(x, y))

    def test_onnxt_runtime_mean(self):
        idi = numpy.identity(2, dtype=numpy.float64)
        onx = OnnxMean('X', idi, output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray((idi + X) / 2, got['Y'], decimal=6)
        python_tested.append(OnnxMean)

    def test_onnxt_runtime_min(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMin, lambda x, y: numpy.minimum(x, y))

    def test_onnxt_runtime_mul(self):
        self.common_test_onnxt_runtime_binary(OnnxMul, lambda x, y: x * y)

    def test_onnxt_runtime_nrg(self):
        self.common_test_onnxt_runtime_unary(OnnxNeg, numpy.negative)

    def test_onnxt_runtime_not(self):
        self.common_test_onnxt_runtime_unary(OnnxNot, numpy.logical_not)

    def test_onnxt_runtime_pow(self):
        self.common_test_onnxt_runtime_binary(OnnxPow, numpy.power)

    def test_onnxt_runtime_quantize_linear(self):
        X = numpy.array([[[[-162, 10], [-100, 232], [-20, -50]],
                          [[-76, 0], [0, 252], [32, -44]],
                          [[245, -485], [-960, -270], [-375, -470]], ], ],
                        dtype=numpy.float32)
        y_scale = numpy.array([2, 4, 5], dtype=numpy.float32)
        y_zero_point = numpy.array([84, 24, 196], dtype=numpy.uint8)
        exp = ((X / y_scale.reshape(1, 3, 1, 1) +
                y_zero_point.reshape(1, 3, 1, 1)).astype(numpy.uint8))
        onx = OnnxQuantizeLinear(
            'X', y_scale, y_zero_point, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        X = numpy.array([0, 2, 4, 1000, -254, -1000]).astype(numpy.float32)
        y_scale = numpy.float32(2)
        y_zero_point = numpy.uint8(128)
        exp = numpy.array([128, 129, 130, 255, 1, 0]).astype(numpy.uint8)
        onx = OnnxQuantizeLinear(
            'X', y_scale, y_zero_point, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxQuantizeLinear)

    def test_onnxt_runtime_reciprocal(self):
        self.common_test_onnxt_runtime_unary(OnnxReciprocal, numpy.reciprocal)

    def test_onnxt_runtime_reduce_log_sum_exp(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceLogSumExp('X', output_names=['Y'], keepdims=0,
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X)))
        self.assertEqualArray(res, got['Y'], decimal=6)

        onx = OnnxReduceLogSumExp('X', output_names=['Y'], axes=[1],
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1))
        self.assertEqualArray(res, got['Y'].ravel())

        onx = OnnxReduceLogSumExp(
            'X', output_names=['Y'], axes=[1], keepdims=1,
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1, keepdims=1))
        self.assertEqualArray(
            res.ravel(), got['Y'].ravel())  # pylint: disable=E1101
        python_tested.append(OnnxReduceLogSumExp)

    def test_onnxt_runtime_reduce_max(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMax('X', output_names=['Y'], keepdims=0,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, keepdims=False, axis=None),  # pylint: disable=E1101,E1123
                              got['Y'], decimal=6)

        onx = OnnxReduceMax('X', output_names=['Y'], axes=[1],
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, axis=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

        onx = OnnxReduceMax('X', output_names=['Y'], axes=[1], keepdims=1,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, axis=1, keepdims=1).ravel(),  # pylint: disable=E1101,E1123
                              got['Y'].ravel())
        python_tested.append(OnnxReduceMax)

    def test_onnxt_runtime_reduce_mean(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMean('X', output_names=['Y'], keepdims=0,
                             op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X), got['Y'], decimal=6)

        onx = OnnxReduceMean('X', output_names=['Y'], axes=1,
                             op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceMean('X', output_names=['Y'], axes=1, keepdims=1,
                             op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceMean)

    def test_onnxt_runtime_reduce_min(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMin('X', output_names=['Y'], keepdims=0,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, keepdims=False, axis=None),  # pylint: disable=E1101,E1123
                              got['Y'], decimal=6)

        onx = OnnxReduceMin('X', output_names=['Y'], axes=[1],
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, axis=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

        onx = OnnxReduceMin('X', output_names=['Y'], axes=[1], keepdims=1,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, axis=1, keepdims=1).ravel(),  # pylint: disable=E1101,E1123
                              got['Y'].ravel())
        python_tested.append(OnnxReduceMin)

    def test_onnxt_runtime_reduce_prod(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceProd('X', output_names=['Y'], keepdims=0,
                             op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X), got['Y'], decimal=6)

        onx = OnnxReduceProd('X', output_names=['Y'], axes=1,
                             op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceProd('X', output_names=['Y'], axes=1, keepdims=1,
                             op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceProd)

    def test_onnxt_runtime_reduce_sum(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSum('X', output_names=['Y'], keepdims=0,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X), got['Y'], decimal=6)

        onx = OnnxReduceSum('X', output_names=['Y'], axes=1,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSum('X', output_names=['Y'], axes=1, keepdims=1,
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceSum)

    def test_onnxt_runtime_reduce_sum_square(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], keepdims=0,
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X)), got['Y'], decimal=6)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1,
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1, keepdims=1,
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceSumSquare)

    def test_onnxt_runtime_relu(self):
        self.common_test_onnxt_runtime_unary(
            OnnxRelu, lambda x: numpy.maximum(x, 0))

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    def test_onnxt_runtime_reshape(self):
        sh = numpy.array([1, 4], dtype=numpy.int64)
        onx = OnnxReshape('X', sh, output_names=['Y'],
                          op_version=get_opset_number_from_onnx())
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = X.reshape(sh.tolist())
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxReshape)

    def test_onnxt_runtime_shape(self):
        x = numpy.random.randn(20, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = x.shape
        onx = OnnxShape('X', output_names=['Y'],
                        op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxShape)

    def test_onnxt_runtime_sigmoid(self):
        self.common_test_onnxt_runtime_unary(OnnxSigmoid, logistic_sigmoid)

    def test_onnxt_runtime_sign(self):
        self.common_test_onnxt_runtime_unary(OnnxSign, numpy.sign)

    def test_onnxt_runtime_sin(self):
        self.common_test_onnxt_runtime_unary(OnnxSin, numpy.sin)

    def test_onnxt_runtime_slice(self):
        x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = x[0:3, 0:10]
        starts = numpy.array([0, 0], dtype=numpy.int64)
        ends = numpy.array([3, 10], dtype=numpy.int64)
        onx = OnnxSlice('X', starts, ends, output_names=['Y'],
                        op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = x[0:3, 0:10]
        starts = numpy.array([0, 0], dtype=numpy.int64)
        ends = numpy.array([3, 10], dtype=numpy.int64)
        axes = numpy.array([0, 1], dtype=numpy.int64)
        onx = OnnxSlice('X', starts, ends, axes, output_names=['Y'],
                        op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = x[0:3:-1, 0:10:2]
        starts = numpy.array([0, 0], dtype=numpy.int64)
        ends = numpy.array([3, 10], dtype=numpy.int64)
        axes = numpy.array([0, 1], dtype=numpy.int64)
        steps = numpy.array([-1, 2], dtype=numpy.int64)
        onx = OnnxSlice('X', starts, ends, axes, steps, output_names=['Y'],
                        op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxSlice)

    def test_onnxt_runtime_sqrt(self):
        self.common_test_onnxt_runtime_unary(OnnxSqrt, numpy.sqrt)

    def test_onnxt_runtime_squeeze(self):
        x = numpy.random.randn(20, 1).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = numpy.squeeze(x)
        onx = OnnxSqueeze('X', axes=[1], output_names=['Y'],
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])

        x = numpy.random.randn(1, 20).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = numpy.squeeze(x)
        onx = OnnxSqueeze('X', axes=[0], output_names=['Y'],
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxSqueeze)

    def test_onnxt_runtime_softmax(self):
        self.common_test_onnxt_runtime_unary(OnnxSoftmax, softmax)

    def test_onnxt_runtime_sub(self):
        self.common_test_onnxt_runtime_binary(OnnxSub, lambda x, y: x - y)

    def test_onnxt_runtime_sum(self):
        self.common_test_onnxt_runtime_binary(OnnxSum, lambda x, y: x + y)

    def test_onnxt_runtime_topk(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        # axis=1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64),
                       axis=1, output_names=['Y', 'Yi'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=get_opset_number_from_onnx())
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
                       axis=0, output_names=['Y', 'Yi'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        exp = numpy.array([[2., 1., 2., 5., 5.],
                           [1., -1., -2., 4., 4.]],
                          dtype=numpy.float32)
        self.assertEqualArray(exp, got['Y'])

        # axis=-1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64),
                       axis=-1, output_names=['Y', 'Yi'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        exp = numpy.array([[4., 3.],
                           [5., 4.],
                           [5., 2.]],
                          dtype=numpy.float32)
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxTopK)

    def test_onnxt_runtime_topk2(self):
        X = numpy.array([[-0., -0.08000002, -2., -2.88000023]],
                        dtype=numpy.float32)

        # axis=-1, k=-1
        onx = OnnxTopK('X', numpy.array([1], dtype=numpy.int64),
                       axis=1, output_names=['Y', 'Yi'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=get_opset_number_from_onnx())
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

        onx = OnnxTranspose('X', perm=[0, 1], output_names=['Y'],
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X, got['Y'])

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        onx = OnnxTranspose('X', perm=[1, 0], output_names=['Y'],
                            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X.T, got['Y'])
        python_tested.append(OnnxTranspose)

    def test_cpp_topk_min_1(self):
        X = numpy.array([1, -1], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 1, 0, 0)
        to2 = topk_element_min_double(X, 1, False, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 0, 0)
        to2 = topk_element_min_double(X, 2, False, 50)
        self.assertEqual(set(to1[1]), set(to2))

        X = numpy.array([1, -1], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 0, 0)
        to2 = topk_element_min_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 0, 0)
        to2 = topk_element_min_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 3, 0, 0)
        to2 = topk_element_min_double(X, 3, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 4, 0, 0)
        to2 = topk_element_min_double(X, 4, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float32)
        to1 = topk_sorted_implementation(X, 4, 0, 0)
        to2 = topk_element_min_float(X, 4, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_float(X, to2)
        self.assertEqualArray(to1[0], v2)

    def test_cpp_topk_min_2(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.int64)
        to1 = topk_sorted_implementation(X, 2, 1, 0)
        to2 = topk_element_min_int64(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_int64(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)
        to1 = topk_sorted_implementation(X, 2, 1, 0)
        to2 = topk_element_min_float(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_float(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 1, 0)
        to2 = topk_element_min_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        to1 = topk_sorted_implementation(X, 3, 1, 0)
        to2 = topk_element_min_double(X, 3, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        to1 = topk_sorted_implementation(X, 4, 1, 0)
        to2 = topk_element_min_double(X, 4, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

    def test_cpp_topk_max_1(self):
        X = numpy.array([1, -1], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 1, 0, 1)
        to2 = topk_element_max_double(X, 1, False, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 0, 1)
        to2 = topk_element_max_double(X, 2, False, 50)
        self.assertEqual(set(to1[1]), set(to2))

        X = numpy.array([1, -1], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 0, 1)
        to2 = topk_element_max_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 0, 1)
        to2 = topk_element_max_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 3, 0, 1)
        to2 = topk_element_max_double(X, 3, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 4, 0, 1)
        to2 = topk_element_max_double(X, 4, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([1, -1, -2, 4, 5], dtype=numpy.float32)
        to1 = topk_sorted_implementation(X, 4, 0, 1)
        to2 = topk_element_max_float(X, 4, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_float(X, to2)
        self.assertEqualArray(to1[0], v2)

    def test_cpp_topk_max_2(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.int64)
        to1 = topk_sorted_implementation(X, 2, 1, 1)
        to2 = topk_element_max_int64(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_int64(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)
        to1 = topk_sorted_implementation(X, 2, 1, 1)
        to2 = topk_element_max_float(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_float(X, to2)
        self.assertEqualArray(to1[0], v2)

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float64)
        to1 = topk_sorted_implementation(X, 2, 1, 1)
        to2 = topk_element_max_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        to1 = topk_sorted_implementation(X, 3, 1, 1)
        to2 = topk_element_max_double(X, 3, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

        to1 = topk_sorted_implementation(X, 4, 1, 1)
        to2 = topk_element_max_double(X, 4, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

    def test_cpp_topk_max_openmp(self):
        X = numpy.random.randn(100, 10).astype(  # pylint: disable=E1101
            numpy.float64)  # pylint: disable=E1101
        to1 = topk_sorted_implementation(X, 2, 1, 1)
        to2 = topk_element_max_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

    def test_cpp_pairwise(self):
        X = numpy.full((20, 4), 1, dtype=numpy.float32)
        X[::2, 3] = 20
        X[1::5, 1] = 30
        X[::5, 2] = 40
        cd = cdist(X[:10], X[10:])
        to1 = topk_sorted_implementation(cd, 3, 1, 1)
        to2 = topk_element_max_double(cd, 3, True, 50)
        self.assertEqualArray(to1[1], to2)

    @unittest.skipIf(onnx_opset_version() < 12, reason="new API not available")
    def test_make_sparse_tensor_12(self):
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,  # pylint: disable=E1101
            dims=(5, ),
            vals=values
        )
        indices = [1, 3, 5, 7, 9]
        indices_tensor = make_tensor(
            name='test_indices',
            data_type=TensorProto.INT64,  # pylint: disable=E1101
            dims=(5, ),
            vals=indices
        )
        dense_shape = [10]
        sparse = make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        self.assertEqual(sparse.values, values_tensor)  # pylint: disable=E1101
        self.assertEqual(
            sparse.indices, indices_tensor)  # pylint: disable=E1101
        self.assertEqual(sparse.dims, dense_shape)  # pylint: disable=E1101

        X = numpy.array([0.1, 0.2], dtype=numpy.float32)
        cst = OnnxConstant(value_floats=X, op_version=12)
        onx = OnnxAdd('X', cst, op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        try:
            oinf = OnnxInference(model_def)
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to load the model:\n{}".format(model_def)) from e
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Ad_C0', 'Co_output0'])
        self.assertEqualArray(X * 2, got['Ad_C0'])


if __name__ == "__main__":
    unittest.main()
