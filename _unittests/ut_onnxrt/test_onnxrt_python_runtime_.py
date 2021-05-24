"""
@brief      test log(time=2s)
"""
import unittest
import warnings
import sys
from logging import getLogger
from contextlib import redirect_stdout
from io import StringIO
import numpy
from scipy.sparse import coo_matrix, csr_matrix, SparseEfficiencyWarning
from scipy.special import (  # pylint: disable=E0611
    expit as logistic_sigmoid, erf)
from scipy.spatial.distance import cdist
from onnx import TensorProto, __version__ as onnx_version
from onnx.helper import make_sparse_tensor, make_tensor
from onnx.defs import onnx_opset_version
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper import compare_module_version
from sklearn.utils.extmath import softmax
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs, OnnxAdd, OnnxAnd,
    OnnxArgMax_11, OnnxArgMax,
    OnnxArgMin_11, OnnxArgMin,
    OnnxBatchNormalization,
    OnnxAcos, OnnxAcosh, OnnxAsin, OnnxAsinh, OnnxAtan, OnnxAtanh,
    OnnxCast, OnnxCeil, OnnxClip,
    OnnxCompress,
    OnnxConcat, OnnxConv, OnnxConvTranspose,
    OnnxConstant, OnnxConstant_9, OnnxConstant_11,
    OnnxConstantOfShape,
    OnnxCos, OnnxCosh,
    OnnxDequantizeLinear,
    OnnxDet, OnnxDiv,
    OnnxDropout, OnnxDropout_7,
    OnnxEinsum, OnnxEqual, OnnxErf, OnnxExp, OnnxEyeLike,
    OnnxFlatten, OnnxFloor,
    OnnxGreater, OnnxGreaterOrEqual, OnnxGemm, OnnxGlobalAveragePool,
    OnnxIdentity, OnnxIsNaN,
    OnnxLess, OnnxLessOrEqual,
    OnnxLog, OnnxLpNormalization,
    OnnxMatMul, OnnxMax, OnnxMaxPool, OnnxMean, OnnxMin, OnnxMod, OnnxMul,
    OnnxNeg, OnnxNot,
    OnnxOr,
    OnnxPad, OnnxPow,
    OnnxQLinearConv, OnnxQuantizeLinear,
    OnnxReciprocal,
    OnnxReduceL1, OnnxReduceL2,
    OnnxReduceLogSumExp, OnnxReduceMax, OnnxReduceMean, OnnxReduceMin,
    OnnxReduceProd,
    OnnxReduceSum, OnnxReduceSumApi11, OnnxReduceSum_11, OnnxReduceSum_1,
    OnnxReduceSumSquare,
    OnnxRelu, OnnxReshape,
    OnnxRound,
    OnnxScatterElements, OnnxShape, OnnxSlice, OnnxSigmoid, OnnxSign,
    OnnxSin, OnnxSinh,
    OnnxSize, OnnxSoftmax,
    OnnxSplit, OnnxSplitApi11,
    OnnxSqrt, OnnxSub, OnnxSum,
    OnnxSqueeze, OnnxSqueezeApi11,
    OnnxTan, OnnxTanh, OnnxTopK, OnnxTranspose,
    OnnxUnsqueeze, OnnxUnsqueezeApi11
)
try:
    from skl2onnx.algebra.onnx_ops import OnnxCelu
except ImportError:
    OnnxCelu = None
try:
    from skl2onnx.algebra.onnx_ops import OnnxBatchNormalization_14
except ImportError:
    OnnxBatchNormalization_14 = None
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import (
    get_opset_number_from_onnx, get_ir_version_from_onnx)
from mlprodict.onnxrt.validate.validate_python import validate_python_inference
from mlprodict.onnxrt.ops_cpu.op_batch_normalization import (
    _batchnorm_test_mode, _batchnorm_training_mode)
from mlprodict.onnxrt.ops_cpu.op_global_average_pool import _global_average_pool
from mlprodict.onnxrt.ops_cpu._op_onnx_numpy import (  # pylint: disable=E0611,E0401
    topk_element_min_double, topk_element_max_double, topk_element_fetch_double,
    topk_element_min_float, topk_element_max_float, topk_element_fetch_float,
    topk_element_min_int64, topk_element_max_int64, topk_element_fetch_int64)
from mlprodict.onnxrt.ops_cpu.op_celu import _vcelu1, pycelu
from mlprodict.onnxrt.ops_cpu.op_topk import topk_sorted_implementation
from mlprodict.onnxrt.ops_cpu.op_pad import _pad_impl
from mlprodict.onnxrt.ops_cpu.op_max_pool import _pool_get_output_shape, _pool_impl
from mlprodict.onnxrt.ops_cpu.op_dropout import _dropout
from mlprodict.onnxrt.ops_cpu._op_helper import proto2dtype
from mlprodict.onnx_tools.onnx2py_helper import (
    guess_proto_dtype, _elem_type_as_str)
from mlprodict.tools.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType, StringTensorType,
    Int32TensorType, BooleanTensorType, UInt8TensorType,
    Int16TensorType, Int8TensorType, UInt16TensorType,
    UInt32TensorType, UInt64TensorType, Float16TensorType)


try:
    numpy_str = numpy.str
except ImportError:
    numpy_str = str

try:
    numpy_bool = numpy.bool
except ImportError:
    numpy_bool = bool


sparse_support = []
sparse_no_numpy = []
python_tested = []


def make_coo_matrix(*args, **kwargs):
    coo = coo_matrix(*args, **kwargs)
    coo.row = coo.row.astype(numpy.int64)
    coo.col = coo.col.astype(numpy.int64)
    return coo


def wraplog():
    def wrapper(fct):
        def call_f(self):
            # print('BEGIN %s' % fct.__name__)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", DeprecationWarning)
                fct(self)
            # print('DONE %s' % fct.__name__)
        return call_f
    return wrapper


class TestOnnxrtPythonRuntime(ExtTestCase):  # pylint: disable=R0904

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
                                        outputs=None, debug=False,
                                        do_sparse=True):
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
        python_tested.append(onnx_cl)

        # python code
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
        if do_sparse:
            row = numpy.array([0, 0, 1, 3, 1])
            col = numpy.array([0, 2, 1, 3, 1])
            data = numpy.array([1, 1, 1, 1, 1])
            X = make_coo_matrix((data, (row.astype(numpy.int64),
                                        col.astype(numpy.int64))),
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

    @wraplog()
    def test_onnxt_runtime_abs(self):
        self.common_test_onnxt_runtime_unary(OnnxAbs, numpy.abs)

    @wraplog()
    def test_onnxt_runtime_abs_debug(self):
        f = StringIO()
        with redirect_stdout(f):
            self.common_test_onnxt_runtime_unary(
                OnnxAbs, numpy.abs, debug=True)

    @wraplog()
    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

    @wraplog()
    def test_onnxt_runtime_and(self):
        self.common_test_onnxt_runtime_binary(OnnxAnd, numpy.logical_and)

    @wraplog()
    def test_onnxt_runtime_argmax(self):
        opsets = list(range(11, get_opset_number_from_onnx() + 1))
        opsets = ['11only'] + opsets
        for opset in opsets:
            with self.subTest(opset=opset):
                X = numpy.array([[2, 1], [0, 1]], dtype=float)

                if opset == '11only':
                    clarg = OnnxArgMax_11
                    opset = 11
                    br = True
                else:
                    clarg = OnnxArgMax
                    br = False
                onx = clarg('X', output_names=['Y'], keepdims=0,
                            op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmax(
                    X, axis=0), got['Y'], decimal=6)

                if br:
                    continue

                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X.astype(numpy.float32)})

                onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=0,
                                 op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                                      got['Y'].ravel())

                onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=1,
                                 op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
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
                                               target_opset=opset)
                oinf = OnnxInference(model_def_sparse, input_inplace=False)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(exp, got['Y'], decimal=6)
                X = numpy.array([[2, 1], [0, 1]], dtype=float)

        sparse_support.append(('UnOp', None, OnnxArgMax.__name__))
        python_tested.append(OnnxArgMax)

    @unittest.skipIf(onnx_opset_version() < 12, reason="needs onnx 1.7.0")
    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_argmin(self):
        opsets = list(range(11, get_opset_number_from_onnx() + 1))
        opsets = ['11only'] + opsets
        for opset in opsets:
            with self.subTest(opset=opset):
                if opset == '11only':
                    clarg = OnnxArgMin_11
                    opset = 11
                    br = True
                else:
                    clarg = OnnxArgMin
                    br = False
                X = numpy.array([[2, 1], [0, 1]], dtype=float)

                onx = clarg('X', output_names=['Y'], keepdims=0,
                            op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmin(
                    X, axis=0), got['Y'], decimal=6)
                if br:
                    continue

                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X.astype(numpy.float32)})

                onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=0,
                                 op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                                      got['Y'].ravel())

                onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=1,
                                 op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
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

                model_def_sparse = onx.to_onnx({'X': X}, target_opset=opset)
                oinf = OnnxInference(model_def_sparse, input_inplace=False)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(exp, got['Y'], decimal=6)

        sparse_support.append(('UnOp', None, OnnxArgMin.__name__))
        python_tested.append(OnnxArgMin)

    @unittest.skipIf(onnx_opset_version() < 12, reason="needs onnx 1.7.0")
    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_acos(self):
        self.common_test_onnxt_runtime_unary(OnnxAcos, numpy.arccos)

    @wraplog()
    def test_onnxt_runtime_acosh(self):
        self.common_test_onnxt_runtime_unary(OnnxAcosh, numpy.arccosh)

    @wraplog()
    def test_onnxt_runtime_asin(self):
        self.common_test_onnxt_runtime_unary(OnnxAsin, numpy.arcsin)

    @wraplog()
    def test_onnxt_runtime_asinh(self):
        self.common_test_onnxt_runtime_unary(OnnxAsinh, numpy.arcsinh)

    @wraplog()
    def test_onnxt_runtime_atan(self):
        self.common_test_onnxt_runtime_unary(OnnxAtan, numpy.arctan)

    @wraplog()
    def test_onnxt_runtime_atanh(self):
        self.common_test_onnxt_runtime_unary(OnnxAtanh, numpy.arctanh)

    @wraplog()
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
            numpy.arctan2(y_val, x_val), atan2(y_val, x_val), decimal=6)

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_batch_normalization_training_fct(self):
        x = numpy.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(numpy.float32)
        s = numpy.array([1.0, 1.5]).astype(numpy.float32)
        bias = numpy.array([0, 1]).astype(numpy.float32)
        mean = numpy.array([0, 3]).astype(numpy.float32)
        var = numpy.array([1, 1.5]).astype(numpy.float32)
        y, scale, bias, mean, var = (
            _batchnorm_training_mode(x, s, bias, mean, var))
        self.assertEqualArray(
            numpy.array([[[[-1.2247356, 0., 1.2247356]],
                          [[-0.8371035, 1., 2.8371034]]]],
                        dtype=numpy.float32), y)
        self.assertEqualArray(
            numpy.array([0., 3.], dtype=numpy.float32), scale)
        self.assertEqualArray(
            numpy.array([0.6666667, 0.6666667], dtype=numpy.float32), bias)
        self.assertEqualArray(
            numpy.array([0., 2.9999998], dtype=numpy.float32), mean)
        self.assertEqualArray(
            numpy.array([0.96666664, 1.4166666], dtype=numpy.float32), var)

    @wraplog()
    @unittest.skipIf(OnnxBatchNormalization_14 is None,
                     reason="onnx too old")
    def test_onnxt_runtime_batch_normalization_training(self):
        # input size: (1, 2, 1, 3)
        x = numpy.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(numpy.float32)
        s = numpy.array([1.0, 1.5]).astype(numpy.float32)
        bias = numpy.array([0, 1]).astype(numpy.float32)
        mean = numpy.array([0, 3]).astype(numpy.float32)
        var = numpy.array([1, 1.5]).astype(numpy.float32)
        y, scale, bias, mean, var = (
            _batchnorm_training_mode(x, s, bias, mean, var))

        onx = OnnxBatchNormalization_14(
            'X', s, bias, mean, var,
            output_names=['Y', 'scale', 'bias', 'mean', 'var'],
            training_mode=1, op_version=14)
        try:
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=14)
        except RuntimeError as e:
            if "Shape inference fails" in str(e):
                warnings.warn(str(e))
                return
            raise e
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(
            list(sorted(got)), ['Y', 'bias', 'mean', 'scale', 'var'])
        self.assertEqualArray(scale, got['scale'])
        self.assertEqualArray(bias, got['bias'])
        self.assertEqualArray(mean, got['mean'])
        # self.assertEqualArray(var, got['var'])
        # self.assertEqualArray(y, got['Y'])
        self.assertNotEmpty(y)
        self.assertNotEmpty(var)

    @wraplog()
    def test_onnxt_runtime_cast_out(self):
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float32)  # pylint: disable=E1101
        dest = [(TensorProto.FLOAT, numpy.float32, FloatTensorType),  # pylint: disable=E1101
                (TensorProto.DOUBLE, numpy.float64,  # pylint: disable=E1101
                 DoubleTensorType),  # pylint: disable=E1101
                (TensorProto.INT32, numpy.int32,  # pylint: disable=E1101
                 Int32TensorType),  # pylint: disable=E1101
                (TensorProto.INT64, numpy.int64,  # pylint: disable=E1101
                 Int64TensorType),  # pylint: disable=E1101
                (TensorProto.INT8, numpy.int8,  # pylint: disable=E1101
                 Int8TensorType),  # pylint: disable=E1101
                (TensorProto.INT16, numpy.int16,  # pylint: disable=E1101
                 Int16TensorType),  # pylint: disable=E1101
                (TensorProto.UINT8, numpy.uint8,  # pylint: disable=E1101
                 UInt8TensorType),  # pylint: disable=E1101
                (TensorProto.UINT32, numpy.uint32,  # pylint: disable=E1101
                 UInt32TensorType),  # pylint: disable=E1101
                (TensorProto.UINT16, numpy.uint16,  # pylint: disable=E1101
                 UInt16TensorType),  # pylint: disable=E1101
                (TensorProto.UINT64, numpy.uint64,  # pylint: disable=E1101
                 UInt64TensorType),  # pylint: disable=E1101
                (TensorProto.FLOAT16, numpy.float16,  # pylint: disable=E1101
                 Float16TensorType),  # pylint: disable=E1101
                (TensorProto.BOOL, numpy.bool_,  # pylint: disable=E1101
                 BooleanTensorType),  # pylint: disable=E1101
                (TensorProto.STRING, numpy.str_, StringTensorType), ]  # pylint: disable=E1101

        for opset in range(9, get_opset_number_from_onnx() + 1):
            for to, nptp, outp in dest:
                if nptp == numpy.bool_:
                    self.assertIn(proto2dtype(to), (nptp, bool))
                elif nptp == numpy.str_:
                    self.assertIn(proto2dtype(to), (nptp, str))
                else:
                    self.assertEqual(proto2dtype(to), nptp)
                self.assertEqual(to, guess_proto_dtype(nptp))
                self.assertNotEmpty(_elem_type_as_str(to))
                with self.subTest(opset=opset, to=to):
                    onx = OnnxCast('X', to=to, output_names=['Y'],
                                   op_version=opset)
                    model_def = onx.to_onnx(
                        {'X': x}, outputs=[('Y', outp())],
                        target_opset=opset)
                    got = OnnxInference(model_def).run({'X': x})
                    if nptp == numpy.str_:
                        self.assertEqual(
                            x.astype(nptp).tolist(), got['Y'].tolist())
                    else:
                        self.assertEqualArray(x.astype(nptp), got['Y'])

        python_tested.append(OnnxCast)

    @wraplog()
    def test_onnxt_runtime_cast_in(self):
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float32)  # pylint: disable=E1101
        dest = [(TensorProto.FLOAT, numpy.float32, FloatTensorType),  # pylint: disable=E1101
                (TensorProto.DOUBLE, numpy.float64,  # pylint: disable=E1101
                 DoubleTensorType),  # pylint: disable=E1101
                (TensorProto.INT32, numpy.int32,  # pylint: disable=E1101
                 Int32TensorType),  # pylint: disable=E1101
                (TensorProto.INT64, numpy.int64,  # pylint: disable=E1101
                 Int64TensorType),  # pylint: disable=E1101
                (TensorProto.INT8, numpy.int8,  # pylint: disable=E1101
                 Int8TensorType),  # pylint: disable=E1101
                (TensorProto.INT16, numpy.int16,  # pylint: disable=E1101
                 Int16TensorType),  # pylint: disable=E1101
                (TensorProto.UINT8, numpy.uint8,  # pylint: disable=E1101
                 UInt8TensorType),  # pylint: disable=E1101
                (TensorProto.UINT32, numpy.uint32,  # pylint: disable=E1101
                 UInt32TensorType),  # pylint: disable=E1101
                (TensorProto.UINT16, numpy.uint16,  # pylint: disable=E1101
                 UInt16TensorType),  # pylint: disable=E1101
                (TensorProto.UINT64, numpy.uint64,  # pylint: disable=E1101
                 UInt64TensorType),  # pylint: disable=E1101
                (TensorProto.FLOAT16, numpy.float16,  # pylint: disable=E1101
                 Float16TensorType),  # pylint: disable=E1101
                (TensorProto.BOOL, numpy.bool_,  # pylint: disable=E1101
                 BooleanTensorType),  # pylint: disable=E1101
                (TensorProto.STRING, numpy.str_, StringTensorType), ]  # pylint: disable=E1101

        for opset in range(9, get_opset_number_from_onnx() + 1):
            for to, nptp, _ in dest:
                if nptp == numpy.bool_:
                    self.assertIn(proto2dtype(to), (nptp, bool))
                elif nptp == numpy.str_:
                    self.assertIn(proto2dtype(to), (nptp, str))
                else:
                    self.assertEqual(proto2dtype(to), nptp)
                self.assertEqual(to, guess_proto_dtype(nptp))
                self.assertNotEmpty(_elem_type_as_str(to))
                with self.subTest(opset=opset, to=to):
                    xi = x.astype(nptp)
                    onx = OnnxCast('X', to=TensorProto.STRING,  # pylint: disable=E1101
                                   output_names=['Y'],
                                   op_version=opset)
                    model_def = onx.to_onnx(
                        {'X': xi}, outputs=[('Y', StringTensorType())],
                        target_opset=opset)
                    got = OnnxInference(model_def).run({'X': xi})
                    self.assertEqual(
                        xi.astype(str).tolist(), got['Y'].tolist())

        python_tested.append(OnnxCast)

    @wraplog()
    def test_onnxt_runtime_ceil(self):
        self.common_test_onnxt_runtime_unary(OnnxCeil, numpy.ceil)

    @unittest.skipIf(OnnxCelu is None, reason="onnx too recent")
    @wraplog()
    def test_onnxt_runtime_celu1(self):
        self.common_test_onnxt_runtime_unary(
            OnnxCelu, _vcelu1, op_version=12,
            outputs=[('Y', FloatTensorType([None, 2]))])

    @unittest.skipIf(OnnxCelu is None, reason="onnx too recent")
    @wraplog()
    def test_onnxt_runtime_celu2(self):
        _vcelu2 = numpy.vectorize(
            lambda x: pycelu(x, 1.), otypes=[numpy.float])
        self.common_test_onnxt_runtime_unary(
            OnnxCelu, _vcelu2, op_version=12,
            outputs=[('Y', FloatTensorType([None, 2]))])

    @unittest.skipIf(onnx_opset_version() < 11,
                     reason="Explicitely tests Clip >= 11")
    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_compress(self):
        # axis is None
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(numpy.float32)
        x = x.reshape((-1, 2))
        cond = numpy.array([False, True, False])
        onx = OnnxCompress('X', 'cond', output_names=['Y'],
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'cond': cond},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        exp = numpy.compress(cond, x)
        got = OnnxInference(model_def).run({'X': x, 'cond': cond})
        self.assertEqualArray(exp, got['Y'])

        python_tested.append(OnnxCompress)

    @wraplog()
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

    @wraplog()
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

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_conv0(self):
        x = numpy.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                           [5., 6., 7., 8., 9.],
                           [10., 11., 12., 13., 14.],
                           [15., 16., 17., 18., 19.],
                           [20., 21., 22., 23., 24.]]]]).astype(numpy.float32)
        W = numpy.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                           [1., 1., 1.],
                           [1., 1., 1.]]]]).astype(numpy.float32)

        # test 1
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
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y_with_padding, got['Y'])

        # test 2
        y_without_padding = numpy.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                           [99., 108., 117.],
                                           [144., 153., 162.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3], pads=[0, 0, 0, 0],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y_without_padding, got['Y'])

        # test 3
        y = numpy.array([[[[12., 27., 24.],
                           [63., 108., 81.],
                           [72., 117., 84.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3],
            auto_pad='SAME_LOWER', strides=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y, got['Y'])

        python_tested.append(OnnxConv)

    @wraplog()
    def test_onnxt_runtime_conv1(self):
        x = numpy.array([[[[0., 1., 2., 3., 4.],
                           [5., 6., 7., 8., 9.],
                           [10., 11., 12., 13., 14.],
                           [15., 16., 17., 18., 19.],
                           [20., 21., 22., 23., 24.],
                           [25., 26., 27., 28., 29.],
                           [30., 31., 32., 33., 34.]]]]).astype(numpy.float32)
        W = numpy.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                           [1., 1., 1.],
                           [1., 1., 1.]]]]).astype(numpy.float32)

        # test 1
        y_with_padding = numpy.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                                        [63., 108., 81.],
                                        [123., 198., 141.],
                                        [112., 177., 124.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y_with_padding, got['Y'])

        # test 2
        y_without_padding = numpy.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                           [144., 162.],
                                           [234., 252.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y_without_padding, got['Y'])

        # test 3
        y_with_asymmetric_padding = numpy.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                                   [99., 117.],
                                                   [189., 207.],
                                                   [171., 183.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3], pads=[1, 0, 1, 0], strides=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y_with_asymmetric_padding, got['Y'])

    @wraplog()
    def test_onnxt_runtime_conv2_B(self):
        x = numpy.random.rand(1, 3, 5, 4).astype(numpy.float32)
        W = numpy.random.rand(4, 3, 3, 3).astype(numpy.float32)
        B = numpy.array([100, 700, 1000, 7000], dtype=numpy.float32)
        onx = OnnxConv(
            'X', 'W', 'B', output_names=['Y'],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'W': W, 'B': B},
                                target_opset=get_opset_number_from_onnx())
        ys = []
        for rt in ['python', 'onnxruntime1']:
            oinf = OnnxInference(model_def, runtime=rt)
            got = oinf.run({'X': x, 'W': W, 'B': B})
            ys.append(got['Y'])
        self.assertEqualArray(ys[0], ys[1], decimal=4)

    @wraplog()
    def test_onnxt_runtime_conv_transpose(self):
        x = numpy.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                           [3., 4., 5.],
                           [6., 7., 8.]]]]).astype(numpy.float32)
        W = numpy.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                           [1., 1., 1.],
                           [1., 1., 1.]],
                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]]]]).astype(numpy.float32)

        y_with_padding = numpy.array([[[[0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                                        [3., 8., 15., 12., 7.],
                                        [9., 21., 36., 27., 15.],
                                        [9., 20., 33., 24., 13.],
                                        [6., 13., 21., 15., 8.]],

                                       [[0., 1., 3., 3., 2.],
                                        [3., 8., 15., 12., 7.],
                                        [9., 21., 36., 27., 15.],
                                        [9., 20., 33., 24., 13.],
                                        [6., 13., 21., 15., 8.]]]]).astype(numpy.float32)

        onx = OnnxConvTranspose(
            'X', W, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

        python_tested.append(OnnxConv)

    @wraplog()
    def test_onnxt_runtime_conv_transpose_B(self):
        x = numpy.random.rand(1, 3, 5, 4).astype(numpy.float32)
        W = numpy.random.rand(3, 4, 3, 3).astype(numpy.float32)
        B = numpy.array([100, 700, 1000, 7000], dtype=numpy.float32)
        onx = OnnxConvTranspose(
            'X', 'W', 'B', output_names=['Y'],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x, 'W': W, 'B': B},
                                target_opset=get_opset_number_from_onnx())
        ys = []
        for rt in ['python', 'onnxruntime1']:
            oinf = OnnxInference(model_def, runtime=rt)
            got = oinf.run({'X': x, 'W': W, 'B': B})
            ys.append(got['Y'])
        self.assertEqual(len(ys), 2)
        # self.assertEqualArray(ys[0], ys[1])

    @wraplog()
    def test_onnxt_runtime_conv_transpose_1d(self):
        x = numpy.array([[[0., 1., 2.]]]).astype(numpy.float32)
        W = numpy.array([[[1., 1., 1.],  # (1, 2, 3)
                          [1., 1., 1.]]]).astype(numpy.float32)

        y_with_padding = numpy.array(
            [[[0., 1., 3., 3., 2.],  # (1, 2, 5)
              [0., 1., 3., 3., 2.]]]).astype(numpy.float32)

        onx = OnnxConvTranspose(
            'X', W, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())

        oinf = OnnxInference(model_def, runtime="onnxruntime1")
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

        python_tested.append(OnnxConvTranspose)

    @wraplog()
    def test_onnxt_runtime_conv_transpose_3d(self):
        x = numpy.arange(60).reshape((1, 1, 3, 4, 5)).astype(numpy.float32)
        W = numpy.ones((1, 2, 3, 3, 3)).astype(numpy.float32)

        y_with_padding = numpy.array(
            [[[[[0., 1., 3., 6., 9., 7., 4.],  # (1, 2, 5, 6, 7)
                [5., 12., 21., 27., 33., 24., 13.],
                [15., 33., 54., 63., 72., 51., 27.],
                [30., 63., 99., 108., 117., 81., 42.],
                [25., 52., 81., 87., 93., 64., 33.],
                [15., 31., 48., 51., 54., 37., 19.]],

                [[20., 42., 66., 72., 78., 54., 28.],
                 [50., 104., 162., 174., 186., 128., 66.],
                 [90., 186., 288., 306., 324., 222., 114.],
                 [120., 246., 378., 396., 414., 282., 144.],
                 [90., 184., 282., 294., 306., 208., 106.],
                 [50., 102., 156., 162., 168., 114., 58.]],

                [[60., 123., 189., 198., 207., 141., 72.],
                 [135., 276., 423., 441., 459., 312., 159.],
                 [225., 459., 702., 729., 756., 513., 261.],
                 [270., 549., 837., 864., 891., 603., 306.],
                 [195., 396., 603., 621., 639., 432., 219.],
                 [105., 213., 324., 333., 342., 231., 117.]],

                [[60., 122., 186., 192., 198., 134., 68.],
                 [130., 264., 402., 414., 426., 288., 146.],
                 [210., 426., 648., 666., 684., 462., 234.],
                 [240., 486., 738., 756., 774., 522., 264.],
                 [170., 344., 522., 534., 546., 368., 186.],
                 [90., 182., 276., 282., 288., 194., 98.]],

                [[40., 81., 123., 126., 129., 87., 44.],
                 [85., 172., 261., 267., 273., 184., 93.],
                 [135., 273., 414., 423., 432., 291., 147.],
                 [150., 303., 459., 468., 477., 321., 162.],
                 [105., 212., 321., 327., 333., 224., 113.],
                 [55., 111., 168., 171., 174., 117., 59.]]],

              [[[0., 1., 3., 6., 9., 7., 4.],
                [5., 12., 21., 27., 33., 24., 13.],
                [15., 33., 54., 63., 72., 51., 27.],
                [30., 63., 99., 108., 117., 81., 42.],
                [25., 52., 81., 87., 93., 64., 33.],
                [15., 31., 48., 51., 54., 37., 19.]],

                [[20., 42., 66., 72., 78., 54., 28.],
                 [50., 104., 162., 174., 186., 128., 66.],
                 [90., 186., 288., 306., 324., 222., 114.],
                 [120., 246., 378., 396., 414., 282., 144.],
                 [90., 184., 282., 294., 306., 208., 106.],
                 [50., 102., 156., 162., 168., 114., 58.]],

                [[60., 123., 189., 198., 207., 141., 72.],
                 [135., 276., 423., 441., 459., 312., 159.],
                 [225., 459., 702., 729., 756., 513., 261.],
                 [270., 549., 837., 864., 891., 603., 306.],
                 [195., 396., 603., 621., 639., 432., 219.],
                 [105., 213., 324., 333., 342., 231., 117.]],

                [[60., 122., 186., 192., 198., 134., 68.],
                 [130., 264., 402., 414., 426., 288., 146.],
                 [210., 426., 648., 666., 684., 462., 234.],
                 [240., 486., 738., 756., 774., 522., 264.],
                 [170., 344., 522., 534., 546., 368., 186.],
                 [90., 182., 276., 282., 288., 194., 98.]],

                [[40., 81., 123., 126., 129., 87., 44.],
                 [85., 172., 261., 267., 273., 184., 93.],
                 [135., 273., 414., 423., 432., 291., 147.],
                 [150., 303., 459., 468., 477., 321., 162.],
                 [105., 212., 321., 327., 333., 224., 113.],
                 [55., 111., 168., 171., 174., 117., 59.]]]]]).astype(numpy.float32)

        onx = OnnxConvTranspose(
            'X', W, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

    @unittest.skipIf(True, reason="fails with output_shape")
    @wraplog()
    def test_onnxt_runtime_conv_transpose_output_shape(self):
        x = numpy.arange(9).reshape((1, 1, 3, 3)).astype(numpy.float32)
        W = numpy.ones((1, 2, 3, 3)).astype(numpy.float32)

        y_with_padding = numpy.array(
            [[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]],

              [[0., 0., 1., 1., 3., 2., 2., 0.],
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(numpy.float32)

        with self.subTest(part="output_shape"):
            onx = OnnxConvTranspose(
                'X', W, output_names=['Y'],
                strides=[3, 2], output_shape=[10, 8],
                op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=get_opset_number_from_onnx())

            oinf = OnnxInference(model_def, runtime="onnxruntime1")
            got = oinf.run({'X': x})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(y_with_padding, got['Y'])

            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(y_with_padding, got['Y'])

    @wraplog()
    def test_onnxt_runtime_conv_transpose_attributes(self):
        x = numpy.arange(9).reshape((1, 1, 3, 3)).astype(numpy.float32)
        W = numpy.ones((1, 2, 3, 3)).astype(numpy.float32)

        y_with_padding = numpy.array(
            [[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]],

              [[0., 0., 1., 1., 3., 2., 2., 0.],
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [0., 0., 1., 1., 3., 2., 2., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [3., 3., 7., 4., 9., 5., 5., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [6., 6., 13., 7., 15., 8., 8., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(numpy.float32)

        with self.subTest(part="output_padding"):
            onx = OnnxConvTranspose(
                'X', W, output_names=['Y'],
                strides=[3, 2], output_padding=[1, 1],
                op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=get_opset_number_from_onnx())
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(y_with_padding, got['Y'])

        with self.subTest(part="kernel_shape"):
            onx = OnnxConvTranspose(
                'X', W, output_names=['Y'],
                strides=[3, 2], output_shape=[10, 8],
                kernel_shape=[3, 3], output_padding=[1, 1],
                op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=get_opset_number_from_onnx())
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(y_with_padding, got['Y'])

    @wraplog()
    def test_onnxt_runtime_conv_transpose_dilation(self):
        x = numpy.array([[[[3., 8., 1.],  # (1, 1, 3, 3)
                           [9., 5., 7.],
                           [3., 2., 6.]]]]).astype(numpy.float32)
        W = numpy.array([[[[7., 2.],  # (1, 1, 2, 2)
                           [1., 9.]]]]).astype(numpy.float32)

        y_with_padding = numpy.array(
            [[[[21., 56., 13., 16., 2.],  # [1, 1, 5, 5]
                [63., 35., 67., 10., 14.],
                [24., 22., 76., 76., 21.],
                [9., 5., 88., 45., 63.],
                [3., 2., 33., 18., 54.]]]]).astype(numpy.float32)

        onx = OnnxConvTranspose(
            'X', W, output_names=['Y'], dilations=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

    @wraplog()
    def test_onnxt_runtime_conv_transpose_pads(self):
        x = numpy.arange(9).reshape((1, 1, 3, 3)).astype(numpy.float32)
        W = numpy.ones((1, 2, 3, 3)).astype(numpy.float32)

        y_with_padding = numpy.array(
            [[[[1., 1., 3.],  # (1, 2, 7, 3)
                [1., 1., 3.],
                [7., 4., 9.],
                [7., 4., 9.],
                [7., 4., 9.],
                [13., 7., 15.],
                [13., 7., 15.]],

              [[1., 1., 3.],
                [1., 1., 3.],
                [7., 4., 9.],
                [7., 4., 9.],
                [7., 4., 9.],
                [13., 7., 15.],
                [13., 7., 15.]]]]).astype(numpy.float32)

        onx = OnnxConvTranspose(
            'X', W, output_names=['Y'],
            strides=[3, 2], pads=[1, 2, 1, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y_with_padding, got['Y'])

    @wraplog()
    def test_onnxt_runtime_cos(self):
        self.common_test_onnxt_runtime_unary(OnnxCos, numpy.cos)

    @wraplog()
    def test_onnxt_runtime_cosh(self):
        self.common_test_onnxt_runtime_unary(OnnxCosh, numpy.cosh)

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_det(self):
        self.common_test_onnxt_runtime_unary(
            OnnxDet, lambda x: numpy.array([numpy.linalg.det(x)]),
            do_sparse=False)

    @wraplog()
    def test_onnxt_runtime_dequantize_linear(self):
        X = numpy.array([[[[3, 89], [34, 200], [74, 59]],
                          [[5, 24], [24, 87], [32, 13]],
                          [[245, 99], [4, 142], [121, 102]], ], ],
                        dtype=numpy.uint8)
        x_scale = numpy.array([2, 4, 5], dtype=numpy.float32)
        x_zero_point = numpy.array([84, 24, 196], dtype=numpy.uint8)
        exp = ((X.astype(numpy.float32) - x_zero_point.reshape(
                (1, 3, 1, 1)).astype(numpy.float32)) *
               x_scale.reshape((1, 3, 1, 1)))
        onx = OnnxDequantizeLinear(
            'X', x_scale, x_zero_point, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        X = numpy.array([0, 3, 128, 255]).astype(numpy.uint8)
        x_scale = numpy.array([2], dtype=numpy.float32)
        x_zero_point = numpy.array([128], dtype=numpy.uint8)
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

    @wraplog()
    def test_onnxt_runtime_div(self):
        self.common_test_onnxt_runtime_binary(OnnxDiv, lambda x, y: x / y)

    @wraplog()
    def test_onnxt_runtime_dropout_10(self):
        seed = numpy.int64(0)
        X = numpy.random.randn(3, 4, 5).astype(numpy.float32)
        onx = OnnxDropout_7('X', output_names=['Y'], op_version=10)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=10)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, X.shape)
        self.assertEqualArray(got['Y'], _dropout(X, seed=seed)[0])
        python_tested.append(OnnxDropout)

    @wraplog()
    def test_onnxt_runtime_dropout(self):
        seed = numpy.int64(0)
        X = numpy.random.randn(3, 4, 5).astype(numpy.float32)

        onx = OnnxDropout('X', output_names=['Y'], seed=seed,
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, X.shape)
        self.assertEqualArray(got['Y'], _dropout(X, seed=seed)[0])

        onx = OnnxDropout('X', output_names=['Y', 'Z'], seed=seed,
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType()),
                                         ('Z', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Z'])
        self.assertEqual(got['Y'].shape, X.shape)
        res = _dropout(X, seed=seed, return_mask=True)
        self.assertEqualArray(got['Y'], res[0])
        self.assertEqualArray(got['Z'], res[1])

        R = numpy.array([0.1], dtype=numpy.float32)
        onx = OnnxDropout('X', 'R', output_names=['Y'], seed=seed,
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'R': R.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'R': R})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, X.shape)
        self.assertEqualArray(
            got['Y'], _dropout(X, seed=seed, drop_probability=0.1)[0])

        R = numpy.array([0.75], dtype=numpy.float32)
        B = numpy.array([True])
        onx = OnnxDropout('X', 'R', 'B', output_names=['Y'], seed=seed,
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'R': R, 'B': B},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'R': R, 'B': B})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, X.shape)
        self.assertEqualArray(
            got['Y'], _dropout(X, seed=seed, drop_probability=0.75,
                               training_mode=True)[0])

        python_tested.append(OnnxDropout)

    @wraplog()
    def test_onnxt_runtime_einsum(self):
        X = numpy.random.randn(5, 2, 3).astype(numpy.float32)
        Y = numpy.random.randn(5, 3, 4).astype(numpy.float32)
        equation = 'bij,bjk->bik'
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

        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(numpy.float32),
                                           'Y': Y.astype(numpy.float32)})

    @wraplog()
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

        oinfpy = OnnxInference(model_def, runtime="python")
        validate_python_inference(oinfpy, {'X': X.astype(numpy.int64)})
        python_tested.append(OnnxEyeLike)

    @wraplog()
    def test_onnxt_runtime_equal(self):
        self.common_test_onnxt_runtime_binary(OnnxEqual, numpy.equal)

    @wraplog()
    def test_onnxt_runtime_erf(self):
        self.common_test_onnxt_runtime_unary(OnnxErf, erf)

    @wraplog()
    def test_onnxt_runtime_exp(self):
        self.common_test_onnxt_runtime_unary(OnnxExp, numpy.exp)

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_floor(self):
        self.common_test_onnxt_runtime_unary(OnnxFloor, numpy.floor)

    @wraplog()
    def test_onnxt_runtime_gather_elements0(self):
        from skl2onnx.algebra.onnx_ops import OnnxGatherElements  # pylint: disable=E0611
        # ex 1
        data = numpy.array([[1, 2],
                            [3, 4]], dtype=numpy.float32)
        indices = numpy.array([], dtype=numpy.int64)

        onx = OnnxGatherElements('X', 'Y', output_names=['Z'], axis=1,
                                 op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': data, 'Y': indices})
        self.assertEqual(got['Z'].size, 0)

    @wraplog()
    def test_onnxt_runtime_gather_elements0_fortran(self):
        from skl2onnx.algebra.onnx_ops import OnnxGatherElements  # pylint: disable=E0611
        # ex 1
        data = numpy.array([[1, 2],
                            [3, 4]], dtype=numpy.float32, order='F')
        indices = numpy.array([], dtype=numpy.int64, order='F')

        onx = OnnxGatherElements('X', 'Y', output_names=['Z'], axis=1,
                                 op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': data, 'Y': indices})
        self.assertEqual(got['Z'].size, 0)

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_gemm_python(self):
        self.do_test_onnxt_runtime_gemm("python")
        python_tested.append(OnnxGemm)

    @wraplog()
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

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_greater_or_equal(self):
        self.common_test_onnxt_runtime_binary(
            OnnxGreaterOrEqual, numpy.greater_equal)

    @wraplog()
    def test_onnxt_runtime_identity(self):
        self.common_test_onnxt_runtime_unary(OnnxIdentity, lambda x: x)

    @wraplog()
    def test_onnxt_runtime_isnan(self):
        self.common_test_onnxt_runtime_unary(OnnxIsNaN, numpy.isnan)

    @wraplog()
    def test_onnxt_runtime_less(self):
        self.common_test_onnxt_runtime_binary(OnnxLess, numpy.less)

    @wraplog()
    def test_onnxt_runtime_less_or_equal(self):
        self.common_test_onnxt_runtime_binary(
            OnnxLessOrEqual, numpy.less_equal)

    @wraplog()
    def test_onnxt_runtime_log(self):
        self.common_test_onnxt_runtime_unary(OnnxLog, numpy.log)

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_matmul(self):
        self.common_test_onnxt_runtime_binary(OnnxMatMul, lambda x, y: x @ y)

    @wraplog()
    def test_onnxt_runtime_max(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMax, lambda x, y: numpy.maximum(x, y))

    @wraplog()
    def test_onnxt_runtime_max_pool_1d_default(self):
        X = numpy.random.randn(1, 3, 32).astype(numpy.float32)
        kernel_shape = [2]
        strides = [1]
        out_shape = _pool_get_output_shape(
            b'VALID', X.shape[2:], kernel_shape, strides)
        exp = _pool_impl(
            X, X.shape, kernel_shape, strides, out_shape, [0], b'MAX')
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=kernel_shape,
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        self.assertEqual(got['Y'].dtype, X.dtype)

    @wraplog()
    def test_onnxt_runtime_max_pool_1d_default_64(self):
        X = numpy.random.randn(1, 3, 32).astype(numpy.float64)
        kernel_shape = [2]
        strides = [1]
        out_shape = _pool_get_output_shape(
            b'VALID', X.shape[2:], kernel_shape, strides)
        exp = _pool_impl(
            X, X.shape, kernel_shape, strides, out_shape, [0], b'MAX')
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=kernel_shape,
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'], decimal=5)
        self.assertEqual(got['Y'].dtype, X.dtype)
        self.assertEqual(got['Y'].dtype, numpy.float64)

    @wraplog()
    def test_onnxt_runtime_max_pool_2d(self):
        # ceil
        X = numpy.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]]).astype(numpy.float32)
        exp = numpy.array([[[[11, 12], [15, 16]]]]).astype(numpy.float32)
        kernel_shape = [3, 3]
        strides = [2, 2]
        ceil_mode = True
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=kernel_shape,
            strides=strides, ceil_mode=ceil_mode,
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        # default
        X = numpy.random.randn(1, 3, 32, 32).astype(numpy.float32)
        kernel_shape = [2, 2]
        strides = [1, 1]
        out_shape = _pool_get_output_shape(
            b'VALID', X.shape[2:], kernel_shape, strides)
        exp = _pool_impl(X, X.shape, kernel_shape, strides,
                         out_shape, (0, 0), b'MAX')
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=kernel_shape,
            strides=strides,
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        # dilations
        X = numpy.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]]).astype(numpy.float32)
        exp = numpy.array([[[[11, 12], [15, 16]]]]).astype(numpy.float32)
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=[2, 2],
            strides=[1, 1], dilations=[2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        # pads
        X = numpy.array([[[[1, 2, 3, 4, 5],
                           [6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15],
                           [16, 17, 18, 19, 20],
                           [21, 22, 23, 24, 25]]]]).astype(numpy.float32)
        exp = numpy.array([[[[13, 14, 15, 15, 15],
                             [18, 19, 20, 20, 20],
                             [23, 24, 25, 25, 25],
                             [23, 24, 25, 25, 25],
                             [23, 24, 25, 25, 25]]]]).astype(numpy.float32)
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        # precomputed_same_upper(self):
        X = numpy.array([[[[1, 2, 3, 4, 5],
                           [6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15],
                           [16, 17, 18, 19, 20],
                           [21, 22, 23, 24, 25]]]]).astype(numpy.float32)
        exp = numpy.array([[[[7, 9, 10],
                             [17, 19, 20],
                             [22, 24, 25]]]]).astype(numpy.float32)
        onx = OnnxMaxPool('X', output_names=['Y'],
                          kernel_shape=[3, 3],
                          strides=[2, 2], auto_pad=b'SAME_UPPER',
                          op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxMaxPool)

    @wraplog()
    def test_onnxt_runtime_max_pool_3d_default(self):
        X = numpy.random.randn(1, 3, 32, 32, 32).astype(numpy.float32)
        out_shape = _pool_get_output_shape(
            b'VALID', X.shape[2:], [2, 2, 2], [1, 1, 1])
        onx = OnnxMaxPool(
            'X', output_names=['Y'], kernel_shape=[2, 2, 2],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx(
            {'X': X}, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual([1, 3, 31, 31, 31], list(got['Y'].shape))
        try:
            exp = _pool_impl(X, X.shape, [2, 2, 2], [
                             1, 1, 1], out_shape, (0, 0), b'MAX')
        except IndexError:
            # remaining bug
            return
        self.assertEqualArray(exp, got['Y'])

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_min(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMin, lambda x, y: numpy.minimum(x, y))

    @wraplog()
    def test_onnxt_runtime_mod(self):
        self.common_test_onnxt_runtime_binary(
            OnnxMod, lambda x, y: numpy.nan_to_num(numpy.mod(x, y)),
            dtype=numpy.int64)

    @wraplog()
    def test_onnxt_runtime_mul(self):
        self.common_test_onnxt_runtime_binary(OnnxMul, lambda x, y: x * y)

    @wraplog()
    def test_onnxt_runtime_nrg(self):
        self.common_test_onnxt_runtime_unary(OnnxNeg, numpy.negative)

    @wraplog()
    def test_onnxt_runtime_not(self):
        self.common_test_onnxt_runtime_unary(OnnxNot, numpy.logical_not)

    @wraplog()
    def test_onnxt_runtime_or(self):
        self.common_test_onnxt_runtime_binary(OnnxOr, numpy.logical_or)

    @wraplog()
    def test_onnxt_runtime_pad(self):
        data = numpy.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                           dtype=numpy.float32)
        pads = numpy.array([0, 2, 0, 0], dtype=numpy.int64)
        constant_value = numpy.array([0.0], dtype=numpy.float32)
        exp = numpy.array([[0.0, 0.0, 1.0, 1.2],
                           [0.0, 0.0, 2.3, 3.4],
                           [0.0, 0.0, 4.5, 5.7]], dtype=numpy.float32)
        onx = OnnxPad(
            'data', 'pads', constant_value, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'data': data, 'pads': pads})
        self.assertEqualArray(exp, got['Y'])

        data = numpy.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                           dtype=numpy.float32)
        pads = numpy.array([0, 2, 0, 0], dtype=numpy.int64)
        constant_value = numpy.array([0.0], dtype=numpy.float32)
        exp = numpy.array([[1.0, 1.2, 1.0, 1.2],
                           [2.3, 3.4, 2.3, 3.4],
                           [4.5, 5.7, 4.5, 5.7]], dtype=numpy.float32)
        onx = OnnxPad(
            'data', 'pads', output_names=['Y'],
            mode='reflect', op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'data': data, 'pads': pads})
        self.assertEqualArray(exp, got['Y'])

        data = numpy.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                           dtype=numpy.float32)
        pads = numpy.array([0, 2, 0, 0], dtype=numpy.int64)
        constant_value = numpy.array([0.0], dtype=numpy.float32)
        exp = numpy.array([[1.0, 1.0, 1.0, 1.2],
                           [2.3, 2.3, 2.3, 3.4],
                           [4.5, 4.5, 4.5, 5.7]], dtype=numpy.float32)
        onx = OnnxPad(
            'data', 'pads', output_names=['Y'],
            mode='edge', op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'data': data, 'pads': pads})
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxPad)

    @wraplog()
    def test_onnxt_runtime_pad2(self):
        data = numpy.random.randn(1, 3, 4, 5).astype(numpy.float32)
        pads = numpy.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(numpy.int64)
        constant_value = numpy.array([1.2], dtype=numpy.float32)
        exp = _pad_impl(data, pads, 'constant', 1.2)
        onx = OnnxPad(
            'data', 'pads', constant_value, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'data': data, 'pads': pads})
        self.assertEqualArray(exp, got['Y'])

        for mode in ('edge', 'reflect'):
            onx = OnnxPad(
                'data', 'pads', output_names=['Y'],
                mode=mode, op_version=get_opset_number_from_onnx())
            model_def = onx.to_onnx({'data': data, 'pads': pads},
                                    target_opset=get_opset_number_from_onnx())

            data = numpy.random.randn(1, 3, 4, 5).astype(numpy.int32)
            pads = numpy.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(numpy.int64)
            exp = _pad_impl(data, pads, mode)
            oinf = OnnxInference(model_def)
            got = oinf.run({'data': data, 'pads': pads})
            self.assertEqualArray(exp, got['Y'])

    @wraplog()
    def test_onnxt_runtime_pow(self):
        self.common_test_onnxt_runtime_binary(OnnxPow, numpy.power)

    @wraplog()
    def test_onnxt_runtime_qlinear_conv(self):
        x = numpy.array(
            [[255, 174, 162, 25, 203, 168, 58],
             [15, 59, 237, 95, 129, 0, 64],
             [56, 242, 153, 221, 168, 12, 166],
             [232, 178, 186, 195, 237, 162, 237],
             [188, 39, 124, 77, 80, 102, 43],
             [127, 230, 21, 83, 41, 40, 134],
             [255, 154, 92, 141, 42, 148, 247], ],
            dtype=numpy.uint8).reshape((1, 1, 7, 7))

        x_scale = numpy.float32(0.00369204697)
        x_zero_point = numpy.uint8(132)

        w = numpy.array([0], dtype=numpy.uint8).reshape((1, 1, 1, 1))

        w_scale = numpy.array([0.00172794575], dtype=numpy.float32)
        w_zero_point = numpy.array([255], dtype=numpy.uint8)

        y_scale = numpy.float32(0.00162681262)
        y_zero_point = numpy.uint8(123)

        output = numpy.array(
            [[0, 81, 93, 230, 52, 87, 197],
             [240, 196, 18, 160, 126, 255, 191],
             [199, 13, 102, 34, 87, 243, 89],
             [23, 77, 69, 60, 18, 93, 18],
             [67, 216, 131, 178, 175, 153, 212],
             [128, 25, 234, 172, 214, 215, 121],
             [0, 101, 163, 114, 213, 107, 8], ],
            dtype=numpy.uint8).reshape((1, 1, 7, 7))

        node = OnnxQLinearConv('x', 'x_scale', 'x_zero_point', 'w',
                               'w_scale', 'w_zero_point', 'y_scale',
                               'y_zero_point', output_names=['y'],
                               op_version=get_opset_number_from_onnx())
        inputs = {'x': x, 'x_scale': x_scale, 'x_zero_point': x_zero_point,
                  'w': w, 'w_scale': w_scale, 'w_zero_point': w_zero_point,
                  'y_scale': y_scale, 'y_zero_point': y_zero_point}
        model_def = node.to_onnx(inputs,
                                 target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run(inputs)['y']
        self.assertEqualArray(output, got)
        python_tested.append(OnnxQLinearConv)

    @wraplog()
    def test_onnxt_runtime_quantize_linear(self):
        X = numpy.array([[[[-162, 10], [-100, 232], [-20, -50]],
                          [[-76, 0], [0, 252], [32, -44]],
                          [[245, -485], [-960, -270], [-375, -470]], ], ],
                        dtype=numpy.float32)
        y_scale = numpy.array([2, 4, 5], dtype=numpy.float32)
        y_zero_point = numpy.array([84, 24, 196], dtype=numpy.uint8)
        exp = ((X / y_scale.reshape((1, 3, 1, 1)) +
                y_zero_point.reshape((1, 3, 1, 1))).astype(numpy.uint8))
        onx = OnnxQuantizeLinear(
            'X', y_scale, y_zero_point, output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])

        X = numpy.array([0, 2, 4, 1000, -254, -1000]).astype(numpy.float32)
        y_scale = numpy.array([2], dtype=numpy.float32)
        y_zero_point = numpy.array([128], dtype=numpy.uint8)
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

    @wraplog()
    def test_onnxt_runtime_reciprocal(self):
        self.common_test_onnxt_runtime_unary(OnnxReciprocal, numpy.reciprocal)

    @wraplog()
    def test_onnxt_runtime_reduce_l1(self):
        def reduce_l1(x, axis, keepdims):
            return numpy.sum(numpy.abs(x), axis=axis, keepdims=keepdims)

        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceL1('X', output_names=['Y'], keepdims=0, axes=[1],
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l1(X, axis=1, keepdims=0),
                              got['Y'], decimal=6)

        onx = OnnxReduceL1('X', output_names=['Y'], axes=1,
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l1(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceL1('X', output_names=['Y'], axes=1, keepdims=1,
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l1(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceL2)

    @wraplog()
    def test_onnxt_runtime_reduce_l2(self):
        def reduce_l2(x, axis, keepdims):
            return numpy.sqrt(numpy.sum(x ** 2, axis=axis, keepdims=keepdims))

        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceL2('X', output_names=['Y'], keepdims=0, axes=[1],
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l2(X, axis=1, keepdims=0),
                              got['Y'], decimal=6)

        onx = OnnxReduceL2('X', output_names=['Y'], axes=1,
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l2(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceL2('X', output_names=['Y'], axes=1, keepdims=1,
                           op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l2(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceL2)

    @wraplog()
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

        X = numpy.array([[1., numpy.inf],
                         [numpy.inf, 1],
                         [1., -numpy.inf],
                         [-numpy.inf, 1]], dtype=float)
        onx = OnnxReduceLogSumExp('X', output_names=['Y'], keepdims=0, axes=[1],
                                  op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1))
        self.assertEqualArray(res, got['Y'], decimal=6)
        python_tested.append(OnnxReduceLogSumExp)

    @wraplog()
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

    @wraplog()
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

    @wraplog()
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

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_reduce_sum(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        for opset in (10, 11, 12, 13, get_opset_number_from_onnx()):
            if onnx_opset_version() < opset:
                continue
            if opset < 13:
                cl = OnnxReduceSum_11 if opset >= 11 else OnnxReduceSum_1
                onx = cl('X', output_names=['Y'], keepdims=0,
                         op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.sum(X), got['Y'], decimal=6)
                name = oinf.sequence_[0].ops_.__class__.__name__
                if opset < 11:
                    self.assertEqual(name, 'ReduceSum_1')
                else:
                    self.assertEqual(name, 'ReduceSum_11')

            onx = OnnxReduceSumApi11('X', output_names=['Y'], axes=1,
                                     op_version=opset)
            model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                    target_opset=opset)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': X})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(numpy.sum(X, axis=1).ravel(),
                                  got['Y'].ravel())
            name = oinf.sequence_[0].ops_.__class__.__name__
            if opset >= 13:
                self.assertEqual(name, 'ReduceSum_13')
            elif opset >= 11:
                self.assertEqual(name, 'ReduceSum_11')
            else:
                self.assertEqual(name, 'ReduceSum_1')

        for opset in (11, 12, 13):
            if onnx_opset_version() < opset:
                continue
            onx = OnnxReduceSumApi11('X', output_names=['Y'], axes=1, keepdims=1,
                                     op_version=opset)
            model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                    target_opset=opset)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': X})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(numpy.sum(X, axis=1, keepdims=1).ravel(),
                                  got['Y'].ravel())

        X = numpy.array([[1., numpy.inf],
                         [numpy.inf, 1],
                         [1., -numpy.inf],
                         [-numpy.inf, 1]], dtype=float)
        onx = OnnxReduceSumApi11(
            'X', output_names=['Y'], keepdims=0, axes=[1],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.sum(X, axis=1)
        self.assertEqualArray(res, got['Y'], decimal=6)
        python_tested.append(OnnxReduceSum)

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_reduce_sum_noop(self):
        X = numpy.array([], dtype=float).reshape((2, 0))

        for opset in (13, get_opset_number_from_onnx()):
            if onnx_opset_version() < opset:
                continue
            cl = OnnxReduceSum
            onx = cl('X', output_names=['Y'], keepdims=0,
                     op_version=opset, noop_with_empty_axes=0)
            model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                    target_opset=opset)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': X})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(numpy.sum(X), got['Y'], decimal=6)
            self.assertEqualArray(got['Y'], numpy.array([0], dtype=X.dtype))

    @wraplog()
    def test_onnxt_runtime_relu(self):
        self.common_test_onnxt_runtime_unary(
            OnnxRelu, lambda x: numpy.maximum(x, 0))

    @wraplog()
    def test_onnxt_runtime_round(self):
        self.common_test_onnxt_runtime_unary(OnnxRound, numpy.round)

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_scatter_elements1(self):
        for opset in [11, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                data = numpy.array(
                    [[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=numpy.float32)
                indices = numpy.array([[1, 3]], dtype=numpy.int64)
                updates = numpy.array([[1.1, 2.1]], dtype=numpy.float32)
                output = numpy.array(
                    [[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=numpy.float32)

                onx = OnnxScatterElements(
                    'X', indices, updates, axis=1,
                    output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': data}, target_opset=opset)
                got = OnnxInference(model_def).run({'X': data})
                self.assertEqualArray(output, got['Y'])

                onx = OnnxScatterElements(
                    'X', indices, updates, axis=-1,
                    output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': data}, target_opset=opset)
                got = OnnxInference(model_def).run({'X': data})
                self.assertEqualArray(output, got['Y'])

        python_tested.append(OnnxScatterElements)

    @wraplog()
    def test_onnxt_runtime_scatter_elements2(self):
        for opset in [11, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                x = numpy.arange(20).reshape((4, 5)).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                indices = numpy.array([[1, 1, 1, 1]], dtype=numpy.int64).T
                updates = numpy.array(
                    [[-1, -1, -1, -1]], dtype=numpy.float32).T
                y = x.copy()
                y[:, 1] = -1

                onx = OnnxScatterElements(
                    'X', indices, updates, axis=1,
                    output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x, 'indices': indices, 'updates': updates},
                    target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_sigmoid(self):
        self.common_test_onnxt_runtime_unary(OnnxSigmoid, logistic_sigmoid)

    @wraplog()
    def test_onnxt_runtime_sign(self):
        self.common_test_onnxt_runtime_unary(OnnxSign, numpy.sign)

    @wraplog()
    def test_onnxt_runtime_sin(self):
        self.common_test_onnxt_runtime_unary(OnnxSin, numpy.sin)

    @wraplog()
    def test_onnxt_runtime_sinh(self):
        self.common_test_onnxt_runtime_unary(OnnxSinh, numpy.sinh)

    @wraplog()
    def test_onnxt_runtime_size(self):
        x = numpy.random.randn(20, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = x.size
        onx = OnnxSize('X', output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxSize)

    @wraplog()
    def test_onnxt_runtime_slice(self):
        for opset in [9, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                # steps
                x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                y = x[0:3:2, 0:10:2]
                starts = numpy.array([0, 0], dtype=numpy.int64)
                ends = numpy.array([3, 10], dtype=numpy.int64)
                axes = numpy.array([0, 1], dtype=numpy.int64)
                steps = numpy.array([2, 2], dtype=numpy.int64)
                if opset < 10:
                    onx = OnnxSlice('X', starts=starts, ends=ends, axes=axes,
                                    output_names=['Y'], op_version=opset)
                    y = x[0:3, 0:10]
                else:
                    onx = OnnxSlice('X', starts, ends, axes, steps,
                                    output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

                # other
                x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
                    numpy.float32)
                y = x[0:3, 0:10]
                starts = numpy.array([0, 0], dtype=numpy.int64)
                ends = numpy.array([3, 10], dtype=numpy.int64)
                if opset < 10:
                    onx = OnnxSlice('X', starts=starts, ends=ends,
                                    output_names=['Y'], op_version=opset)
                else:
                    onx = OnnxSlice('X', starts, ends, output_names=['Y'],
                                    op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

                x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
                    numpy.float32)
                y = x[0:3, 0:10]
                starts = numpy.array([0, 0], dtype=numpy.int64)
                ends = numpy.array([3, 10], dtype=numpy.int64)
                axes = numpy.array([0, 1], dtype=numpy.int64)
                if opset < 10:
                    onx = OnnxSlice('X', starts=starts, ends=ends, axes=axes,
                                    output_names=['Y'], op_version=opset)
                else:
                    onx = OnnxSlice('X', starts, ends, axes, output_names=['Y'],
                                    op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

                if opset < 10:
                    continue
                x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
                    numpy.float32)
                y = x[0:3:-1, 0:10:2]
                starts = numpy.array([0, 0], dtype=numpy.int64)
                ends = numpy.array([3, 10], dtype=numpy.int64)
                axes = numpy.array([0, 1], dtype=numpy.int64)
                steps = numpy.array([-1, 2], dtype=numpy.int64)
                onx = OnnxSlice('X', starts, ends, axes, steps, output_names=['Y'],
                                op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxSlice)

    @wraplog()
    def test_onnxt_runtime_slice_step_none(self):
        for opset in [13, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                # steps
                x = numpy.random.randn(20, 10, 5).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                y = x[0:3, 0:10]
                starts = numpy.array([0, 0], dtype=numpy.int64)
                ends = numpy.array([3, 10], dtype=numpy.int64)
                axes = numpy.array([0, 1], dtype=numpy.int64)
                onx = OnnxSlice('X', starts, ends, axes,
                                output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

    @wraplog()
    def test_onnxt_runtime_split(self):
        for opset in [10, 11, 12, 13, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(numpy.float32)
                y = [numpy.array([1., 2.]).astype(numpy.float32),
                     numpy.array([3., 4.]).astype(numpy.float32),
                     numpy.array([5., 6.]).astype(numpy.float32)]
                onx = OnnxSplitApi11(
                    'X', axis=0, split=[2, 2, 2], output_names=['Y1', 'Y2', 'Y3'],
                    op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x.astype(numpy.float32)}, target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y[0], got['Y1'])
                self.assertEqualArray(y[1], got['Y2'])
                self.assertEqualArray(y[2], got['Y3'])

                onx = OnnxSplitApi11(
                    'X', axis=0, output_names=['Y1', 'Y2', 'Y3'],
                    op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x.astype(numpy.float32)}, target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y[0], got['Y1'])
                self.assertEqualArray(y[1], got['Y2'])
                self.assertEqualArray(y[2], got['Y3'])

                x = numpy.array([[1., 2., 3., 4., 5., 6.],
                                 [7., 8., 9., 10., 11., 12.]]).astype(numpy.float32)
                y = [numpy.array([[1., 2.], [7., 8.]]).astype(numpy.float32),
                     numpy.array([[3., 4., 5., 6.], [9., 10., 11., 12.]]).astype(numpy.float32)]
                onx = OnnxSplitApi11(
                    'X', axis=1, split=[2, 4], output_names=['Y1', 'Y2'],
                    op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x.astype(numpy.float32)}, target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y[0], got['Y1'])
                self.assertEqualArray(y[1], got['Y2'])
        python_tested.append(OnnxSplit)

    @wraplog()
    def test_onnxt_runtime_sqrt(self):
        self.common_test_onnxt_runtime_unary(OnnxSqrt, numpy.sqrt)

    @wraplog()
    def test_onnxt_runtime_squeeze(self):
        for opset in [10, 11, 12, 13, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                x = numpy.random.randn(20, 1).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                y = numpy.squeeze(x)
                onx = OnnxSqueezeApi11(
                    'X', axes=[1], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

                x = numpy.random.randn(1, 20).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                y = numpy.squeeze(x)
                onx = OnnxSqueezeApi11(
                    'X', axes=[0], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxSqueeze)

    @wraplog()
    def test_onnxt_runtime_softmax(self):
        self.common_test_onnxt_runtime_unary(OnnxSoftmax, softmax)

    @wraplog()
    def test_onnxt_runtime_sub(self):
        self.common_test_onnxt_runtime_binary(OnnxSub, lambda x, y: x - y)

    @wraplog()
    def test_onnxt_runtime_sum(self):
        self.common_test_onnxt_runtime_binary(OnnxSum, lambda x, y: x + y)

    @wraplog()
    def test_onnxt_runtime_tan(self):
        self.common_test_onnxt_runtime_unary(OnnxTan, numpy.tan)

    @wraplog()
    def test_onnxt_runtime_tanh(self):
        self.common_test_onnxt_runtime_unary(OnnxTanh, numpy.tanh)

    @wraplog()
    def test_onnxt_runtime_topk0(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        # axis=1, k=0
        onx = OnnxTopK('X', numpy.array([0], dtype=numpy.int64),
                       axis=1, output_names=['Y', 'Yi'],
                       op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        self.assertEqual(got['Y'].size, 0)
        self.assertEqual(got['Yi'].size, 0)

    @wraplog()
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

    @wraplog()
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

    @wraplog()
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

    @wraplog()
    def test_onnxt_runtime_unsqueeze(self):
        for opset in [10, 11, 12, 13, get_opset_number_from_onnx()]:
            if opset > get_opset_number_from_onnx():
                continue
            with self.subTest(opset=opset):
                x = numpy.random.randn(1, 3, 1, 5).astype(numpy.float32)
                y = numpy.expand_dims(x, axis=-2)
                onx = OnnxUnsqueezeApi11(
                    'X', axes=[-2], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])

                x = numpy.random.randn(3, 4, 5).astype(numpy.float32)
                y = numpy.expand_dims(x, axis=2)
                y = numpy.expand_dims(y, axis=4)
                y = numpy.expand_dims(y, axis=5)
                onx = OnnxUnsqueezeApi11(
                    'X', axes=[2, 4, 5], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxUnsqueeze)

    @wraplog()
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

    @wraplog()
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

    @wraplog()
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

    @wraplog()
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

    @wraplog()
    def test_cpp_topk_max_openmp(self):
        X = numpy.random.randn(100, 10).astype(  # pylint: disable=E1101
            numpy.float64)  # pylint: disable=E1101
        to1 = topk_sorted_implementation(X, 2, 1, 1)
        to2 = topk_element_max_double(X, 2, True, 50)
        self.assertEqualArray(to1[1], to2)
        v2 = topk_element_fetch_double(X, to2)
        self.assertEqualArray(to1[0], v2)

    @wraplog()
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
    @wraplog()
    def test_make_sparse_tensor_12(self):
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = make_tensor(
            name='test', data_type=TensorProto.FLOAT,  # pylint: disable=E1101
            dims=(5, ), vals=values)
        indices = [1, 3, 5, 7, 9]
        indices_tensor = make_tensor(
            name='test_indices', data_type=TensorProto.INT64,  # pylint: disable=E1101
            dims=(5, ), vals=indices)
        dense_shape = [10]
        sparse = make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        self.assertEqual(sparse.values, values_tensor)  # pylint: disable=E1101
        self.assertEqual(
            sparse.indices, indices_tensor)  # pylint: disable=E1101
        self.assertEqual(sparse.dims, dense_shape)  # pylint: disable=E1101

        opset_tests = [
            (get_opset_number_from_onnx(), OnnxConstant),
            (11, OnnxConstant_11)]

        if (not sys.platform.startswith('win') or
                compare_module_version(onnx_version, (1, 8, 0)) != 0):
            # to_onnx fails for opset, it is expected
            # but it makes python crash on python for onnx 1.8.0
            opset_tests.append((9, OnnxConstant_9))

        for opset, cls in opset_tests:
            for ty, nty in [('float', numpy.float32),
                            ('int', numpy.int64),
                            ('string', numpy_str)]:
                with self.subTest(opset=opset, type=ty):
                    X = numpy.array([0.1, 0.2], dtype=numpy.float32)
                    if opset >= 12:
                        if ty == 'float':
                            cst = cls(value_floats=X, op_version=opset,
                                      output_names=['cst'])
                            tty = FloatTensorType
                        elif ty == 'int':
                            cst = cls(value_ints=(X + 1).astype(nty), op_version=opset,
                                      output_names=['cst'])
                            tty = Int64TensorType
                        elif ty == 'string':
                            cst = cls(value_strings=X.astype(nty), op_version=opset,
                                      output_names=['cst'])
                            tty = StringTensorType
                        else:
                            raise AssertionError(
                                "{}-{} not tested.".format(ty, nty))
                    elif ty != 'float':
                        continue
                    else:
                        cst = cls(value=X, op_version=opset)
                        nty = numpy.float32
                        tty = FloatTensorType
                    onx = OnnxAdd('X', cst, op_version=opset,
                                  output_names=['Y'])
                    try:
                        model_def = onx.to_onnx(
                            {'X': X.astype(nty)}, target_opset=opset,
                            outputs=[('Y', tty()), ('cst', tty())])
                    except RuntimeError as e:
                        if opset == 9:
                            continue
                        raise e
                    try:
                        oinf = OnnxInference(model_def)
                    except RuntimeError as e:
                        raise AssertionError(
                            "Unable to load the model:\n{}".format(model_def)) from e
                    if tty == StringTensorType:
                        continue
                    try:
                        got = oinf.run({'X': X.astype(nty)})
                    except Exception as e:
                        rows = []

                        def bprint(*args):
                            rows.append(str(args))  # pylint: disable=W0640
                        try:
                            oinf.run({'X': X.astype(nty)},
                                     verbose=13, fLOG=bprint)
                        except Exception:  # pylint: disable=W0703
                            pass
                        raise AssertionError(
                            "Execution issue\n{}\n----\n{}".format(
                                "\n".join(map(str, rows)),
                                model_def)) from e
                    if ty == 'float':
                        vexp = X * 2
                    else:
                        vexp = X.astype(nty) + 1
                    if opset >= 11:
                        self.assertEqual(list(sorted(got)), [
                                         'Y', 'cst'])
                        self.assertEqualArray(vexp, got['Y'])
                    else:
                        self.assertEqual(list(sorted(got)), ['Y', 'cst'])
                        self.assertEqualArray(vexp, got['Y'])

    @wraplog()
    def test_make_constant(self):
        X = numpy.array([0.1, 0.2], dtype=numpy.float32)
        values = [1.1, 2.2]
        exp = numpy.array([1.2, 2.4], dtype=numpy.float32)

        opset_tests = [
            (get_opset_number_from_onnx(), OnnxConstant),
            (11, OnnxConstant_11)]

        if (not sys.platform.startswith('win') or
                compare_module_version(onnx_version, (1, 8, 0)) != 0):
            # to_onnx fails for opset, it is expected
            # but it makes python crash on python for onnx 1.8.0
            opset_tests.append((9, OnnxConstant_9))

        for opset, cls in opset_tests:
            with self.subTest(opset=opset):
                if opset >= 12:
                    cst = cls(value_floats=values, op_version=opset)
                else:
                    cst = cls(value=values, op_version=opset)
                onx = OnnxAdd('X', cst, op_version=opset)
                try:
                    model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                            target_opset=opset)
                except RuntimeError as e:
                    if opset == 9:
                        continue
                    raise e
                try:
                    oinf = OnnxInference(model_def)
                except RuntimeError as e:
                    raise AssertionError(
                        "Unable to load the model:\n{}".format(model_def)) from e
                got = oinf.run({'X': X})
                if opset >= 11:
                    self.assertEqual(list(sorted(got)), [
                                     'Ad_C0', 'Co_output0'])
                    self.assertEqualArray(exp, got['Ad_C0'])
                else:
                    self.assertEqual(list(sorted(got)), ['Ad_C0'])
                    self.assertEqualArray(exp, got['Ad_C0'])


if __name__ == "__main__":
    # TestOnnxrtPythonRuntime().test_onnxt_runtime_qlinear_conv()
    unittest.main()
