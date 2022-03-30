"""
@brief      test log(time=152s)
"""
import unittest
import pprint
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
import onnx
from onnx.backend.test.case.node.softmaxcrossentropy import softmaxcrossentropy
from onnx import TensorProto, __version__ as onnx_version
from onnx.helper import make_sparse_tensor, make_tensor
from onnx.defs import onnx_opset_version
from onnx.numpy_helper import from_array
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
    OnnxBatchNormalization, OnnxBitShift,
    OnnxAcos, OnnxAcosh, OnnxAsin, OnnxAsinh, OnnxAtan, OnnxAtanh,
    OnnxAveragePool,
    OnnxCast, OnnxCastLike, OnnxCeil, OnnxClip,
    OnnxCompress,
    OnnxConcat, OnnxConv, OnnxConvTranspose,
    OnnxConstant, OnnxConstant_9, OnnxConstant_11,
    OnnxConstant_12, OnnxConstant_13,
    OnnxConstantOfShape,
    OnnxCos, OnnxCosh,
    OnnxCumSum,
    OnnxDequantizeLinear,
    OnnxDet, OnnxDiv,
    OnnxDropout, OnnxDropout_7,
    OnnxEinsum, OnnxElu, OnnxEqual, OnnxErf, OnnxExp, OnnxExpand, OnnxEyeLike,
    OnnxFlatten, OnnxFloor,
    OnnxGreater, OnnxGreaterOrEqual, OnnxGemm, OnnxGlobalAveragePool,
    OnnxHardmax, OnnxHardSigmoid, OnnxHardSwish,
    OnnxIdentity, OnnxIsNaN,
    OnnxLeakyRelu, OnnxLess, OnnxLessOrEqual,
    OnnxLog, OnnxLogSoftmax, OnnxLpNormalization,
    OnnxMatMul, OnnxMax, OnnxMaxPool, OnnxMean, OnnxMin, OnnxMod, OnnxMul,
    OnnxNeg, OnnxNot,
    OnnxOr,
    OnnxPad, OnnxPow, OnnxPRelu,
    OnnxQLinearConv, OnnxQuantizeLinear,
    OnnxRange,
    OnnxReciprocal,
    OnnxReduceL1, OnnxReduceL2,
    OnnxReduceLogSum, OnnxReduceLogSumExp, OnnxReduceMax,
    OnnxReduceMean, OnnxReduceMin,
    OnnxReduceProd,
    OnnxReduceSum, OnnxReduceSumApi11,
    OnnxReduceSum_13, OnnxReduceSum_11, OnnxReduceSum_1,
    OnnxReduceSumSquare,
    OnnxRelu, OnnxReshape,
    OnnxRound,
    OnnxScatterElements,
    OnnxSelu, OnnxSequenceAt, OnnxSequenceConstruct,
    OnnxShape, OnnxSlice, OnnxSigmoid, OnnxSign,
    OnnxSin, OnnxSinh,
    OnnxSize, OnnxSoftmax, OnnxSoftmaxCrossEntropyLoss,
    OnnxSplit, OnnxSplitApi11,
    OnnxSqrt, OnnxSub, OnnxSum,
    OnnxSqueeze, OnnxSqueezeApi11,
    OnnxTan, OnnxTanh, OnnxTopK, OnnxTranspose, OnnxTrilu,
    OnnxUnsqueeze, OnnxUnsqueezeApi11,
    OnnxXor
)
try:
    from skl2onnx.algebra.onnx_ops import OnnxCelu
except ImportError:
    OnnxCelu = None
try:
    from skl2onnx.algebra.onnx_ops import OnnxBatchNormalization_14
except ImportError:
    OnnxBatchNormalization_14 = None
from skl2onnx import __version__ as skl2onnx_version, __max_supported_opset__
from mlprodict.onnxrt import OnnxInference, OnnxShapeInference
from mlprodict.onnxrt.validate.validate_python import validate_python_inference
from mlprodict.onnxrt.ops_cpu.op_batch_normalization import (
    _batchnorm_test_mode, _batchnorm_training_mode)
from mlprodict.onnxrt.ops_cpu.op_average_pool import (
    _get_output_shape, _pool, _get_pad_shape)
from mlprodict.onnxrt.ops_cpu.op_global_average_pool import _global_average_pool
from mlprodict.onnxrt.ops_cpu._op_onnx_numpy import (  # pylint: disable=E0611,E0401
    topk_element_min_double, topk_element_max_double,
    topk_element_fetch_double,
    topk_element_min_float, topk_element_max_float, topk_element_fetch_float,
    topk_element_min_int64, topk_element_max_int64, topk_element_fetch_int64)
from mlprodict.onnxrt.ops_cpu.op_celu import _vcelu1, pycelu
from mlprodict.onnxrt.ops_cpu.op_leaky_relu import _leaky_relu, _leaky_relu_inplace
from mlprodict.onnxrt.ops_cpu.op_topk import topk_sorted_implementation
from mlprodict.onnxrt.ops_cpu.op_pad import _pad_impl
from mlprodict.onnxrt.ops_cpu.op_max_pool import (
    _pool_get_output_shape, _pool_impl)
from mlprodict.onnxrt.ops_cpu.op_dropout import _dropout
from mlprodict.onnxrt.ops_cpu._op_helper import proto2dtype
from mlprodict.onnx_tools.onnx2py_helper import (
    guess_proto_dtype, _elem_type_as_str)
from mlprodict.testing.test_utils.quantized_tensor import (
    QuantizedTensor, QuantizedBiasTensor, test_qlinear_conv)
from mlprodict.onnxrt.ops_cpu.op_qlinear_conv_ import (  # pylint: disable=W0611,E0611,E0401
    test_qgemm0, test_qgemm1)
from mlprodict.onnxrt.ops_cpu.op_constant import Constant_12, Constant_11, Constant_9
from mlprodict.onnxrt.ops_shape.shape_excs import ShapeInferenceException
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict import __max_supported_opset__ as TARGET_OPSET, get_ir_version

from skl2onnx.common.data_types import (  # pylint: disable=C0412
    FloatTensorType, Int64TensorType, DoubleTensorType, StringTensorType,
    Int32TensorType, BooleanTensorType, UInt8TensorType,
    Int16TensorType, Int8TensorType, UInt16TensorType,
    UInt32TensorType, UInt64TensorType, Float16TensorType)

try:
    numpy_str = numpy.str_
except ImportError:
    numpy_str = str

try:
    numpy_bool = numpy.bool_
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

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        if __name__ == "__main__":
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
            (TARGET_OPSET, OnnxConstant),
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
                            oinf.run({'X': X.astype(nty)},  # opset=13, 14, ...
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
            (TARGET_OPSET, OnnxConstant),
            (13, OnnxConstant_13),
            (12, OnnxConstant_12),
            (11, OnnxConstant_11),
            (9, OnnxConstant_9)]

        expected_type = {15: Constant_12, 14: Constant_12,
                         12: Constant_12, 13: Constant_12,
                         11: Constant_11, 9: Constant_9}

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
                ope = oinf.sequence_[0].ops_
                self.assertIsInstance(ope, expected_type[opset])
                got = oinf.run({'X': X})
                if opset >= 11:
                    self.assertEqual(list(sorted(got)), ['Ad_C0'])
                    self.assertEqualArray(exp, got['Ad_C0'])
                else:
                    self.assertEqual(list(sorted(got)), ['Ad_C0'])
                    self.assertEqualArray(exp, got['Ad_C0'])

    def test_op_constant(self):
        for opv in [9, 10, 11, 12, 13, 14, 15]:  # opset=13, 14, ...
            for dtype in [numpy.float32, numpy.float64,
                          numpy.int32, numpy.int64]:
                with self.subTest(opv=opv, dtype=dtype):
                    X = numpy.array([1], dtype=dtype)
                    pX = from_array(X)
                    op = OnnxAdd('X', OnnxConstant(op_version=opv, value=pX),
                                 output_names=['Y'], op_version=opv)
                    onx = op.to_onnx({'X': X})
                    oinf = OnnxInference(onx)
                    res = oinf.run({'X': X})
                    self.assertEqualArray(res['Y'], X + X)

    def test_opset_skl2onnx(self):
        opset_mlprodict = TARGET_OPSET
        opset_skl2onnx = __max_supported_opset__
        self.assertGreater(opset_skl2onnx, opset_mlprodict)

    def _check_shape_inference(self, onnx_cl, model_def):
        if onnx_cl in {OnnxCastLike}:
            try:
                shapeinf = OnnxShapeInference(model_def)
            except Exception as e:
                raise AssertionError(
                    "Unable to infer shape for:\n%s"
                    "" % onnx_simple_text_plot(model_def)) from e
            try:
                shape_results = shapeinf.run()
            except Exception as e:
                raise AssertionError(
                    "Unable to infer shape %r in\n%r\n." % (
                        e, model_def)) from e
            shape = shape_results.get()
            try:
                self.assertIn('X', shape)
                self.assertIn('Y', shape)
                self.assertIn('Z', shape)
                self.assertEqual(shape['X'].shape, shape['Z'].shape)
                self.assertEqual(shape['Z'].dtype, shape['Y'].dtype)
            except Exception as e:
                raise AssertionError(
                    "Discrepancies in\n%s\n--ONNX--\n%s" % (
                        pprint.pformat(shape),
                        onnx_simple_text_plot(model_def))) from e

    def common_expected_shapes_types(self, oinf, inputs, got, onnx_cl, model_def,
                                     raise_shape=False):
        expected_types = oinf.infer_types()
        self.assertEqual(set(got) & set(expected_types), set(got))
        for k, v in got.items():
            if expected_types[k] in (str, numpy.str_):
                # Type mismatch: dtype('<U32') != <class 'str'>
                continue
            if v.dtype != expected_types[k]:
                raise AssertionError(
                    "Type mismatch: %r != %r\nexpected_types=%r\ngot=%r"
                    "\n----\n%r" % (
                        v.dtype, expected_types[k], expected_types, got,
                        model_def))

        try:
            expected_shapes = oinf.infer_shapes()
            self.assertEqual(set(got) & set(expected_shapes), set(got))
        except RuntimeError as e:
            if raise_shape:
                raise e
            warnings.warn("infer_shapes fails for operator %r." % onnx_cl)

        res = oinf.infer_sizes(inputs)
        self.assertIsInstance(res, dict)

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning,
                               SparseEfficiencyWarning, PendingDeprecationWarning))
    def common_test_onnxt_runtime_unary(self, onnx_cl, np_fct,
                                        op_version=None,
                                        outputs=None, debug=False,
                                        do_sparse=True, raise_shape=False):
        if op_version is None:
            op_version = TARGET_OPSET
        try:
            onx = onnx_cl('X', output_names=['Y'], op_version=op_version)
        except RuntimeError as e:
            raise RuntimeError('onnx.opset={} op_version={}'.format(
                TARGET_OPSET, op_version)) from e
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
                         op.ops_._schema.since_version
                         if op.ops_ is not None else 1)  # pylint: disable=W0212
            for op in oinf.sequence_)
        if debug:
            got = oinf.run({'X': X.astype(numpy.float32)},
                           verbose=1, fLOG=print)
        else:
            got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got, onnx_cl,
            model_def, raise_shape=raise_shape)

        try:
            self.assertEqualArray(np_fct(X), got['Y'], decimal=5)
        except AssertionError as e:
            raise AssertionError(
                'onnx.opset={} op_version={}\n--ONNX--\n{}\n--NAMES--\n{}'.format(
                    TARGET_OPSET, op_version, model_def,
                    all_names)) from e

        # inplace
        oinf = OnnxInference(model_def, input_inplace=False, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(np_fct(X), got['Y'], decimal=5)

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
        self.assertEqualArray(np_fct(X), got['Y'], decimal=5)

        # input inplace
        expe = np_fct(X)
        oinf = OnnxInference(model_def, input_inplace=True, inplace=True)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(expe, got['Y'], decimal=5)

        # shape
        if onnx_cl == OnnxNot:
            self.assertRaise(lambda: OnnxShapeInference(model_def),
                             ShapeInferenceException)
        else:
            shapeinf = OnnxShapeInference(model_def)
            try:
                shape_results = shapeinf.run()
            except Exception as e:
                raise AssertionError(
                    "Unable to infer shape %r in\n%r\n." % (
                        e, model_def)) from e
            shape = shape_results.get()
            self.assertIn('X', shape)
            self.assertIn('Y', shape)
            if onnx_cl == OnnxDet:
                self.assertEqual(shape['X'].dtype, shape['Y'].dtype)
                self.assertEqual(shape['Y'].shape, [])
            elif onnx_cl == OnnxIsNaN:
                self.assertEqual(shape['X'].shape, shape['Y'].shape)
                self.assertEqual(shape['Y'].dtype, numpy.bool_)
            else:
                self.assertEqual(shape['X'].shape, shape['Y'].shape)
                self.assertEqual(shape['X'].dtype, shape['Y'].dtype)

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
            self.assertEqualSparseArray(exp, got['Y'], decimal=5)
            sparse_support.append(('UnOp', op_version, onnx_cl.__name__))

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning,
                               SparseEfficiencyWarning, PendingDeprecationWarning))
    def common_test_onnxt_runtime_binary(self, onnx_cl, np_fct,
                                         dtype=numpy.float32,
                                         op_version=None, debug=False,
                                         raise_shape=False):
        if op_version is None:
            op_version = TARGET_OPSET
        idi = numpy.identity(2, dtype=dtype)
        onx = onnx_cl('X', idi, output_names=['Y'], op_version=op_version)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(dtype)},
                                target_opset=op_version)
        oinf = OnnxInference(model_def)
        if debug:
            got = oinf.run({'X': X.astype(dtype)}, verbose=1, fLOG=print)
        else:
            got = oinf.run({'X': X.astype(dtype)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(dtype)}, got, onnx_cl, model_def,
            raise_shape=raise_shape)
        exp = np_fct(X, idi)
        self.assertEqualArray(exp, got['Y'], decimal=5)

        # python code
        python_tested.append(onnx_cl)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(dtype)})

        # shape
        if onnx_cl not in {OnnxSum, OnnxMatMul}:
            shapeinf = OnnxShapeInference(model_def)
            try:
                shape_results = shapeinf.run()
            except Exception as e:
                raise AssertionError(
                    "Unable to infer shape %r in\n%r\n." % (
                        e, model_def)) from e
            shape = shape_results.get()
            self.assertIn('X', shape)
            self.assertIn('Y', shape)
            if onnx_cl in {OnnxSub, OnnxMul, OnnxDiv, OnnxAdd, OnnxAnd,
                           OnnxOr, OnnxMod, OnnxMax, OnnxMin, OnnxPow,
                           OnnxXor}:
                self.assertEqual(shape['X'].dtype, shape['Y'].dtype)
                self.assertIn(shape['Y'].shape[0], shape['X'].shape[0])
                self.assertEqual(shape['X'].shape[1], shape['Y'].shape[1])
            elif onnx_cl in {OnnxLessOrEqual, OnnxGreater, OnnxGreaterOrEqual,
                             OnnxLess, OnnxEqual}:
                self.assertEqual(shape['X'].dtype, numpy.float32)
                self.assertEqual(shape['Y'].dtype, numpy.bool_)
                self.assertIn(shape['Y'].shape[0], shape['X'].shape[0])
                self.assertEqual(shape['X'].shape[1], shape['Y'].shape[1])
            else:
                self.assertEqual(shape['X'].shape, shape['Y'].shape)
                self.assertEqual(shape['X'].dtype, shape['Y'].dtype)

        # sparse
        idi = make_coo_matrix(numpy.identity(2)).astype(numpy.float32)
        X = make_coo_matrix(numpy.array(
            [[0, 2], [3, -4]], dtype=numpy.float32))
        try:
            exp = np_fct(X, idi)
        except (TypeError, NotImplementedError, ValueError, AttributeError) as e:
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
            self.assertEqualSparseArray(exp, got['Y'], decimal=5)
        elif isinstance(exp, numpy.ndarray):
            self.assertEqualArray(exp, got['Y'], decimal=5)
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
    def test_onnxt_runtime_acos(self):
        self.common_test_onnxt_runtime_unary(OnnxAcos, numpy.arccos)

    @wraplog()
    def test_onnxt_runtime_acosh(self):
        self.common_test_onnxt_runtime_unary(OnnxAcosh, numpy.arccosh)

    @wraplog()
    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

    @wraplog()
    def test_onnxt_runtime_and(self):
        self.common_test_onnxt_runtime_binary(
            OnnxAnd, numpy.logical_and, dtype=numpy.bool_)

    @wraplog()
    def test_onnxt_runtime_argmax(self):
        opsets = list(range(11, TARGET_OPSET + 1))
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
                self._check_shape_inference(OnnxArgMax, model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmax(
                    X, axis=0), got['Y'], decimal=5)
                self.common_expected_shapes_types(
                    oinf, {'X': X}, got, clarg, model_def)

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
                self._check_shape_inference(OnnxArgMax, model_def)
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
                self._check_shape_inference(OnnxArgMax, model_def)

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
                self.assertEqualArray(exp, got['Y'], decimal=5)
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
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.array([1, 2], dtype=numpy.int64),
                              got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxArgMax_12, model_def)

    @wraplog()
    def test_onnxt_runtime_argmin(self):
        opsets = list(range(11, TARGET_OPSET + 1))
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
                self._check_shape_inference(clarg, model_def)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmin(
                    X, axis=0), got['Y'], decimal=5)
                if br:
                    continue

                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X.astype(numpy.float32)})
                self.common_expected_shapes_types(
                    oinfpy, {'X': X.astype(numpy.float32)},
                    got, clarg, model_def)

                onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=0,
                                 op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxArgMin, model_def)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                                      got['Y'].ravel())

                onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=1,
                                 op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxArgMin, model_def)
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
                self.assertEqualArray(exp, got['Y'], decimal=5)

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
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.array([2, 1], dtype=numpy.int64),
                              got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxArgMin_12, model_def)

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
            numpy.arctan2(y_val, x_val), atan2(y_val, x_val), decimal=5)

    def _expect_average_pool(self, node, inputs, outputs, opset=None):
        if opset is None:
            opset = TARGET_OPSET
        ginputs = [
            onnx.helper.make_tensor_value_info(
                node.input[0], TensorProto.FLOAT, []),  # pylint: disable=E1101,
        ]
        goutputs = [
            onnx.helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, []),  # pylint: disable=E1101,
        ]
        model_def = onnx.helper.make_model(
            opset_imports=[onnx.helper.make_operatorsetid('', opset)],
            graph=onnx.helper.make_graph(
                name='test_average_pool', inputs=ginputs, outputs=goutputs,
                nodes=[node]))
        oinf = OnnxInference(model_def)
        got = oinf.run({n: v for n, v in zip(node.input, inputs)})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(outputs[0], got['y'])

    @wraplog()
    def test_onnxt_runtime_average_pool(self):
        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[2, 2], auto_pad='SAME_UPPER')
        x = numpy.random.randn(1, 3, 32, 32).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = _get_output_shape(
            'SAME_UPPER', x_shape[2:], kernel_shape, strides)
        pad_shape = _get_pad_shape(
            'SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = numpy.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', constant_values=numpy.nan)
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[3, 3], pads=[2, 2, 2, 2],
            count_include_pad=1)
        x = numpy.random.randn(1, 3, 28, 28).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = 2
        pad_top = 2
        pad_right = 2
        pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = _get_output_shape(
            'VALID', numpy.add(x_shape[2:], pad_shape), kernel_shape, strides)
        padded = numpy.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', constant_values=0)
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape,
            pad_shape, 'AVG', count_include_pad=1)
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[2, 2], auto_pad='SAME_LOWER')
        x = numpy.random.randn(1, 3, 32, 32).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = _get_output_shape(
            'SAME_LOWER', x_shape[2:], kernel_shape, strides)
        pad_shape = _get_pad_shape(
            'SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
        pad_bottom = pad_shape[0] // 2
        pad_top = pad_shape[0] - pad_bottom
        pad_right = pad_shape[1] // 2
        pad_left = pad_shape[1] - pad_right
        padded = numpy.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', constant_values=numpy.nan)
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[3, 3], pads=[2, 2, 2, 2])
        x = numpy.random.randn(1, 3, 28, 28).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = 2
        pad_top = 2
        pad_right = 2
        pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = _get_output_shape(
            'VALID', numpy.add(x_shape[2:], pad_shape), kernel_shape, strides)
        padded = numpy.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', constant_values=numpy.nan)
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[2])
        x = numpy.random.randn(1, 3, 32).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = [2]
        strides = [1]
        out_shape = _get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = _pool(padded, x_shape, kernel_shape,
                  strides, out_shape, [0], 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[2, 2])
        x = numpy.random.randn(1, 3, 32, 32).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = _get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[5, 5], strides=[3, 3])
        x = numpy.random.randn(1, 3, 32, 32).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape = _get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[2, 2, 2])
        x = numpy.random.randn(1, 3, 32, 32, 32).astype(numpy.float32)
        x_shape = numpy.shape(x)
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape = _get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = _pool(
            padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'AVG')
        self._expect_average_pool(node, inputs=[x], outputs=[y])

        python_tested.append(OnnxAveragePool)

    @wraplog()
    def test_onnxt_runtime_average_pool_ceil(self):
        node = onnx.helper.make_node(
            'AveragePool', inputs=['x'], outputs=['y'],
            kernel_shape=[3, 3], strides=[2, 2], ceil_mode=True)
        x = numpy.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]]]).astype(numpy.float32)
        y = numpy.array([[[
            [6, 7.5], [12, 13.5]]]]).astype(numpy.float32)
        self._expect_average_pool(node, inputs=[x], outputs=[y])

    @wraplog()
    def test_onnxt_runtime_average_pool_big(self):

        with self.subTest(name='test_averagepool_2d_precomputed_pads'):
            node = onnx.helper.make_node(
                'AveragePool', inputs=['x'], outputs=['y'],
                kernel_shape=[5, 5], pads=[2, 2, 2, 2])
            x = numpy.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]]]]).astype(numpy.float32)
            y = numpy.array([[[[7, 7.5, 8, 8.5, 9],
                            [9.5, 10, 10.5, 11, 11.5],
                            [12, 12.5, 13, 13.5, 14],
                            [14.5, 15, 15.5, 16, 16.5],
                            [17, 17.5, 18, 18.5, 19]]]]).astype(numpy.float32)
            self._expect_average_pool(node, inputs=[x], outputs=[y])

        with self.subTest(name='test_averagepool_2d_precomputed_pads_count_include_pad'):
            node = onnx.helper.make_node(
                'AveragePool', inputs=['x'], outputs=['y'],
                kernel_shape=[5, 5], pads=[2, 2, 2, 2], count_include_pad=1)
            x = numpy.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]]]]).astype(numpy.float32)
            y = numpy.array([[[[2.5200, 3.6000, 4.8000, 4.0800, 3.2400],
                            [4.5600, 6.4000, 8.4000, 7.0400, 5.5200],
                            [7.2000, 10.0000, 13.0000, 10.8000, 8.4000],
                            [6.9600, 9.6000, 12.4000, 10.2400, 7.9200],
                            [6.1200, 8.4000, 10.8000, 8.8800, 6.8400]]]]).astype(numpy.float32)
            self._expect_average_pool(node, inputs=[x], outputs=[y])

        with self.subTest(name='test_averagepool_2d_precomputed_same_upper'):
            node = onnx.helper.make_node(
                'AveragePool', inputs=['x'], outputs=['y'],
                kernel_shape=[3, 3], strides=[2, 2], auto_pad='SAME_UPPER')
            x = numpy.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]]]]).astype(numpy.float32)
            y = numpy.array([[[[4, 5.5, 7],
                            [11.5, 13, 14.5],
                            [19, 20.5, 22]]]]).astype(numpy.float32)
            self._expect_average_pool(node, inputs=[x], outputs=[y])

        with self.subTest(name='test_averagepool_2d_precomputed_strides'):
            node = onnx.helper.make_node(
                'AveragePool', inputs=['x'], outputs=['y'],
                kernel_shape=[2, 2], strides=[2, 2])
            x = numpy.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]]]]).astype(numpy.float32)
            y = numpy.array([[[[4, 6],
                            [14, 16]]]]).astype(numpy.float32)
            self._expect_average_pool(node, inputs=[x], outputs=[y])

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxBatchNormalization, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x}, got, OnnxBatchNormalization, model_def)

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxBatchNormalization, model_def)
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
    def test_onnxt_runtime_bitshift(self):
        x = numpy.array([16, 4, 1]).astype(numpy.uint32)
        y = numpy.array([1, 2, 3]).astype(numpy.uint32)

        onx = OnnxBitShift('X', 'Y', direction=b'LEFT',
                           op_version=14, output_names=['Z'])
        model_def = onx.to_onnx({'X': x, 'Y': y}, {'Z': x},
                                target_opset=14)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x, 'Y': y})
        self.assertEqualArray(got['Z'], x << y)

        onx = OnnxBitShift('X', 'Y', direction=b'RIGHT',
                           op_version=14, output_names=['Z'])
        model_def = onx.to_onnx({'X': x, 'Y': y}, {'Z': x},
                                target_opset=14)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x, 'Y': y})
        self.assertEqualArray(got['Z'], x >> y)

        python_tested.append(OnnxBitShift)

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

        for opset in range(9, TARGET_OPSET + 1):
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
                    self._check_shape_inference(OnnxCast, model_def)
                    oinf = OnnxInference(model_def)
                    got = oinf.run({'X': x})
                    if nptp == numpy.str_:
                        self.assertEqual(
                            x.astype(nptp).tolist(), got['Y'].tolist())
                    else:
                        self.assertEqualArray(x.astype(nptp), got['Y'])
                    self.common_expected_shapes_types(
                        oinf, {'X': x}, got, OnnxCast, model_def)

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

        for opset in range(9, TARGET_OPSET + 1):
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
                    self._check_shape_inference(OnnxCast, model_def)
                    got = OnnxInference(model_def).run({'X': xi})
                    self.assertEqual(
                        xi.astype(str).tolist(), got['Y'].tolist())

        python_tested.append(OnnxCast)

    @wraplog()
    def test_onnxt_runtime_cast_like(self):
        x = numpy.array([1.5, 2.1, 3.1, 4.1]).astype(
            numpy.float32)  # pylint: disable=E1101
        y = numpy.array([1.]).astype(numpy.int64)  # pylint: disable=E1101

        for opset in range(15, TARGET_OPSET + 1):
            with self.subTest(opset=opset):
                onx = OnnxCastLike('X', 'Y', output_names=['Z'],
                                   op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x, 'Y': y},
                    outputs=[('Z', Int64TensorType([None]))],
                    target_opset=opset)
                self._check_shape_inference(OnnxCastLike, model_def)
                got = OnnxInference(model_def).run({'X': x, 'Y': y})
                self.assertEqual(x.astype(numpy.int64), got['Z'])

        python_tested.append(OnnxCastLike)

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
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'cond': cond},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxCompress, model_def)
        exp = numpy.compress(cond, x)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x, 'cond': cond})
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x, 'cond': cond}, got, OnnxCompress, model_def)

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
                         op_version=TARGET_OPSET)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConcat, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32),
                        'Y': Y.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Z'])
        self.assertEqual(got['Z'].shape, (6, 2))
        exp = numpy.vstack([X, Y, cst])
        self.assertEqualArray(exp, got['Z'])
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32),
                   'Y': Y.astype(numpy.float32)},
            got, OnnxConcat, model_def)

        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(
            oinfpy, {'X': X.astype(numpy.float32),
                     'Y': Y.astype(numpy.float32)})
        python_tested.append(OnnxConcat)

    @wraplog()
    def test_onnxt_runtime_constant_of_shape(self):
        x = numpy.array([2, 2], dtype=numpy.int64)
        y = numpy.zeros((2, 2), dtype=numpy.float32)
        onx = OnnxConstantOfShape('X', output_names=['Y'],
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.int64)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConstantOfShape, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x.astype(numpy.int64)})
        self.assertEqualArray(y, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x.astype(numpy.int64)}, got,
            OnnxConstantOfShape, model_def)

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConv, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': x})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(y_without_padding, got['Y'])
                if rt == 'python':
                    self.common_expected_shapes_types(
                        oinf, {'X': x}, got, OnnxConv, model_def)
                else:
                    self.assertRaise(
                        lambda: self.common_expected_shapes_types(
                            oinf, {'X': x}, got, OnnxConv, model_def),
                        RuntimeError)

        # test 3
        y = numpy.array([[[[12., 27., 24.],
                           [63., 108., 81.],
                           [72., 117., 84.]]]]).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            kernel_shape=[3, 3],
            auto_pad='SAME_LOWER', strides=[2, 2],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConvTranspose, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'W': W, 'B': B},
                                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConvTranspose, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'W': W, 'B': B},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConvTranspose, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConvTranspose, model_def)

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        self._check_shape_inference(OnnxConvTranspose, model_def)
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
                op_version=TARGET_OPSET)
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=TARGET_OPSET)

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
                op_version=TARGET_OPSET)
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=TARGET_OPSET)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(y_with_padding, got['Y'])

        with self.subTest(part="kernel_shape"):
            onx = OnnxConvTranspose(
                'X', W, output_names=['Y'],
                strides=[3, 2], output_shape=[10, 8],
                kernel_shape=[3, 3], output_padding=[1, 1],
                op_version=TARGET_OPSET)
            model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                    target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConvTranspose, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxConvTranspose, model_def)
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
        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 10., 15.]).astype(numpy.float64)
        onx = OnnxCumSum('X', 'axis', output_names=['Y'],
                         op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxCumSum, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x.astype(numpy.float64),
                        'axis': axis})
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x.astype(numpy.float64),
                   'axis': axis},
            got, OnnxCumSum, model_def)

        python_tested.append(OnnxCumSum)
        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': x, 'axis': axis})

        # reverse = 1
        x = numpy.array([1., 2., 3., 4., 5.]).astype(numpy.float64)
        axis = numpy.array([0]).astype(numpy.int32)
        exp = numpy.array([15., 14., 12., 9., 5.]).astype(numpy.float64)
        onx = OnnxCumSum('X', 'axis', output_names=['Y'], reverse=1,
                         op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=TARGET_OPSET)
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
                         op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=TARGET_OPSET)
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
                         op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=TARGET_OPSET)
        got = OnnxInference(model_def).run({'X': x, 'axis': axis})
        self.assertEqualArray(exp, got['Y'])

        # 2d axis = 1
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float64).reshape((2, 3))
        axis = numpy.array([-1]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 4., 9., 15.]).astype(
            numpy.float64).reshape((2, 3))
        onx = OnnxCumSum('X', 'axis', output_names=['Y'],
                         op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=TARGET_OPSET)
        got = OnnxInference(model_def).run({'X': x, 'axis': axis})
        self.assertEqualArray(exp, got['Y'])

        # 2d axis = 1, reverse
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(
            numpy.float64).reshape((2, 3))
        axis = numpy.array([-1]).astype(numpy.int32)
        exp = numpy.array([1., 3., 6., 4., 9., 15.]).astype(
            numpy.float64).reshape((2, 3))
        onx = OnnxCumSum('X', 'axis', output_names=['Y'], reverse=1,
                         op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x, 'axis': axis},
                                outputs=[('Y', DoubleTensorType())],
                                target_opset=TARGET_OPSET)
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
                             op_version=TARGET_OPSET)
            model_def = onx.to_onnx(
                {'X': x}, outputs=[('Y', DoubleTensorType())],
                target_opset=TARGET_OPSET)
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
                             op_version=TARGET_OPSET)
            model_def = onx.to_onnx(
                {'X': x}, outputs=[('Y', DoubleTensorType())],
                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxDequantizeLinear, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxDequantizeLinear, model_def)

        X = numpy.array([0, 3, 128, 255]).astype(numpy.uint8)
        x_scale = numpy.array([2], dtype=numpy.float32)
        x_zero_point = numpy.array([128], dtype=numpy.uint8)
        exp = numpy.array([-256, -250, 0, 254], dtype=numpy.float32)
        onx = OnnxDequantizeLinear(
            'X', x_scale, x_zero_point, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxDequantizeLinear, model_def)
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
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxDropout_7, model_def)
        python_tested.append(OnnxDropout)

    @wraplog()
    def test_onnxt_runtime_dropout(self):
        seed = numpy.int64(0)
        X = numpy.random.randn(3, 4, 5).astype(numpy.float32)

        onx = OnnxDropout('X', output_names=['Y'], seed=seed,
                          op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, X.shape)
        self.assertEqualArray(got['Y'], _dropout(X, seed=seed)[0])
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxDropout, model_def)

        onx = OnnxDropout('X', output_names=['Y', 'Z'], seed=seed,
                          op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType()),
                                         ('Z', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxDropout, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Z'])
        self.assertEqual(got['Y'].shape, X.shape)
        res = _dropout(X, seed=seed, return_mask=True)
        self.assertEqualArray(got['Y'], res[0])
        self.assertEqualArray(got['Z'], res[1])

        R = numpy.array([0.1], dtype=numpy.float32)
        onx = OnnxDropout('X', 'R', output_names=['Y'], seed=seed,
                          op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'R': R.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'R': R})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, X.shape)
        self.assertEqualArray(
            got['Y'], _dropout(X, seed=seed, drop_probability=0.1)[0])

        R = numpy.array([0.75], dtype=numpy.float32)
        B = numpy.array([True])
        onx = OnnxDropout('X', 'R', 'B', output_names=['Y'], seed=seed,
                          op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'R': R, 'B': B},
                                outputs=[('Y', FloatTensorType())],
                                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X, 'Y': Y})
        exp = numpy.einsum(equation, X, Y)
        self.assertEqualArray(exp, got['Z'])
        self.common_expected_shapes_types(
            oinf, {'X': X, 'Y': Y}, got, OnnxEinsum, model_def)
        python_tested.append(OnnxEinsum)

        oinfpy = OnnxInference(model_def, runtime="python", inplace=True)
        validate_python_inference(oinfpy, {'X': X.astype(numpy.float32),
                                           'Y': Y.astype(numpy.float32)})

    @wraplog()
    def test_onnxt_runtime_elu(self):
        self.common_test_onnxt_runtime_unary(
            OnnxElu, lambda x: numpy.where(x > 0, x, (numpy.exp(x) - 1)))

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    @wraplog()
    def test_onnxt_runtime_expand(self):
        sh = numpy.array([2, 2, 1], dtype=numpy.int64)
        onx = OnnxExpand('X', 'sh', output_names=['Y'],
                         op_version=TARGET_OPSET)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32), 'sh': sh},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxExpand, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.copy(), 'sh': sh})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = X * numpy.ones(sh.tolist())
        self.assertEqualArray(exp, got['Y'])

        X = numpy.array([[1.], [2.], [3.]], dtype=numpy.float32)
        sh = numpy.array([2, 1, 6], dtype=numpy.int64)
        exp = X * numpy.ones(sh.tolist())
        got = oinf.run({'X': X.copy(), 'sh': sh})
        self.assertEqualArray(exp, got['Y'])

        X = numpy.array([[1.], [2.], [3.]], dtype=numpy.float32)
        sh = numpy.array([3, 4], dtype=numpy.int64)
        exp = numpy.tile(X, 4)
        got = oinf.run({'X': X.copy(), 'sh': sh})
        self.assertEqualArray(exp, got['Y'])

        python_tested.append(OnnxExpand)

    @wraplog()
    def test_onnxt_runtime_eyelike(self):
        onx = OnnxEyeLike('X', k=0, output_names=['Y'])
        X = numpy.array([2, 2], dtype=numpy.int64)
        model_def = onx.to_onnx({'X': X.astype(numpy.int64)},
                                target_opset=TARGET_OPSET,
                                outputs=[('Y', FloatTensorType())])
        self._check_shape_inference(OnnxEyeLike, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = numpy.eye(*X, k=0)
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxEyeLike, model_def)

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
                               op_version=TARGET_OPSET)
            model_def = node.to_onnx(
                {'X': x}, outputs=[('Y', FloatTensorType())],
                target_opset=TARGET_OPSET)
            self._check_shape_inference(OnnxFlatten, model_def)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': x})
            new_shape = ((1, -1) if i == 0
                         else (numpy.prod(shape[0:i]).astype(int), -1))
            exp = numpy.reshape(x, new_shape)
            self.assertEqualArray(exp, got['Y'])
            self.common_expected_shapes_types(
                oinf, {'X': x}, got, OnnxFlatten, model_def)

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
                                 op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxGatherElements, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': data, 'Y': indices})
        self.assertEqual(got['Z'].size, 0)
        self.common_expected_shapes_types(
            oinf, {'X': data, 'Y': indices}, got,
            OnnxGatherElements, model_def)

    @wraplog()
    def test_onnxt_runtime_gather_elements0_fortran(self):
        from skl2onnx.algebra.onnx_ops import OnnxGatherElements  # pylint: disable=E0611
        # ex 1
        data = numpy.array([[1, 2],
                            [3, 4]], dtype=numpy.float32, order='F')
        indices = numpy.array([], dtype=numpy.int64, order='F')

        onx = OnnxGatherElements('X', 'Y', output_names=['Z'], axis=1,
                                 op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=TARGET_OPSET)
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
                                 op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=TARGET_OPSET)
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
                                 op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': data, 'Y': indices},
                                outputs=[('Z', FloatTensorType())],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxGatherElements, model_def)
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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version(TARGET_OPSET)
        try:
            oinf = OnnxInference(model_def, runtime=runtime)
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to instantiate (runtime='{}')\n{}".format(
                    runtime, model_def)) from e
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi) + cst, got['Y'], decimal=5)

        onx = OnnxGemm('X', idi, cst, transA=1, transB=1, output_names=['Y'],
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version(TARGET_OPSET)
        try:
            oinf = OnnxInference(model_def, runtime=runtime)
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to instantiate (runtime='{}')\n{}".format(
                    runtime, model_def)) from e
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi.T) + cst, got['Y'], decimal=5)

        onx = OnnxGemm('X', idi, cst, transA=1, output_names=['Y'],
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxGemm, model_def)
        model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime=runtime)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi) + cst, got['Y'], decimal=5)

        onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'],
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime=runtime)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T) + cst, got['Y'], decimal=5)

        onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'],
                       alpha=numpy.float32(1.),
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxGemm, model_def)
        if 'onnxruntime' in runtime:
            model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime=runtime)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T) + cst, got['Y'], decimal=5)

        if runtime != 'onnxruntime1':
            onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'],
                           alpha=numpy.float32(1.),
                           op_version=TARGET_OPSET)
            model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                    target_opset=TARGET_OPSET)
            if 'onnxruntime' in runtime:
                model_def.ir_version = get_ir_version(TARGET_OPSET)
            oinf = OnnxInference(model_def, runtime=runtime)
            got = oinf.run({'X': X.astype(numpy.float32)})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(
                numpy.dot(X, idi.T) + cst, got['Y'], decimal=5)

    @wraplog()
    def test_onnxt_runtime_global_average_pool(self):
        x = x = numpy.random.randn(1, 3, 5, 5).astype(numpy.float32)
        y = _global_average_pool(x).astype(numpy.float32)

        onx = OnnxGlobalAveragePool(
            'X', output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxGlobalAveragePool, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(y, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x}, got, OnnxGlobalAveragePool, model_def)

        x = numpy.array([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]]]).astype(numpy.float32)
        y = numpy.array([[[[5]]]]).astype(numpy.float32)
        onx = OnnxGlobalAveragePool(
            'X', output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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

    def test_onnxt_runtime_hard_sigmoid(self):
        self.common_test_onnxt_runtime_unary(
            OnnxHardSigmoid, lambda x: numpy.maximum(
                0, numpy.minimum(1, x * 0.2 + 0.5)))

    @wraplog()
    def test_onnxt_runtime_hardmax(self):
        def hardmax(x, axis=-1):
            x_argmax = numpy.argmax(x, axis=axis)
            y = numpy.zeros_like(x)
            numpy.put_along_axis(y, numpy.expand_dims(x_argmax, axis=axis),
                                 1, axis=axis)
            return y

        self.common_test_onnxt_runtime_unary(OnnxHardmax, hardmax)

    @wraplog()
    def test_onnxt_runtime_hardswish(self):

        def hardswish(x):
            alfa = 1. / 6
            beta = 0.5
            return x * numpy.maximum(0, numpy.minimum(1, alfa * x + beta))

        self.common_test_onnxt_runtime_unary(OnnxHardSwish, hardswish)

    @wraplog()
    def test_onnxt_runtime_identity(self):
        self.common_test_onnxt_runtime_unary(OnnxIdentity, lambda x: x)

    @wraplog()
    def test_onnxt_runtime_isnan(self):
        self.common_test_onnxt_runtime_unary(OnnxIsNaN, numpy.isnan)

    @wraplog()
    def test_onnxt_runtime_leaky_relu(self):
        self.common_test_onnxt_runtime_unary(
            OnnxLeakyRelu, lambda x: numpy.where(x > 0, x, x * 0.01))

    @wraplog()
    def test_onnxt_runtime_leaky_relu_fct(self):
        x = numpy.random.randn(3, 4, 7).astype(numpy.float32)
        x1 = _leaky_relu(x, 0.77)
        _leaky_relu_inplace(x, 0.77)
        self.assertEqualArray(x, x1)

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
    def test_onnxt_runtime_logsoftmax(self):
        def log_softmax(*args, **kwargs):
            return numpy.log(softmax(*args, **kwargs))

        self.common_test_onnxt_runtime_unary(OnnxLogSoftmax, log_softmax)

    @wraplog()
    def test_onnxt_runtime_lp_normalization(self):
        onx = OnnxLpNormalization('X', output_names=['Y'], p=2, axis=1,
                                  op_version=TARGET_OPSET)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        exp = numpy.array([[0.4472136, 0.8944272],
                           [0.6, -0.8]], dtype=numpy.float32)
        self.assertEqualArray(got['Y'], exp)
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxLpNormalization, model_def)

        onx = OnnxLpNormalization('X', output_names=['Y'], p=2, axis=0,
                                  op_version=TARGET_OPSET)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxLpNormalization, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxMaxPool, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        self.assertEqual(got['Y'].dtype, X.dtype)
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxMaxPool, model_def)

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxMaxPool, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxMaxPool, model_def)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
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
                          op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx(
            {'X': X}, target_opset=TARGET_OPSET)
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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceMean, model_def)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray((idi + X) / 2, got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxMean, model_def)
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
    def test_onnxt_runtime_neg(self):
        self.common_test_onnxt_runtime_unary(OnnxNeg, numpy.negative)

    @wraplog()
    def test_onnxt_runtime_not(self):
        self.common_test_onnxt_runtime_unary(OnnxNot, numpy.logical_not)

    @wraplog()
    def test_onnxt_runtime_or(self):
        self.common_test_onnxt_runtime_binary(
            OnnxOr, numpy.logical_or, dtype=numpy.bool_)

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxPad, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'data': data, 'pads': pads})
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'data': data, 'pads': pads}, got,
            OnnxPad, model_def)

        data = numpy.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                           dtype=numpy.float32)
        pads = numpy.array([0, 2, 0, 0], dtype=numpy.int64)
        constant_value = numpy.array([0.0], dtype=numpy.float32)
        exp = numpy.array([[1.0, 1.2, 1.0, 1.2],
                           [2.3, 3.4, 2.3, 3.4],
                           [4.5, 5.7, 4.5, 5.7]], dtype=numpy.float32)
        onx = OnnxPad(
            'data', 'pads', output_names=['Y'],
            mode='reflect', op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxPad, model_def)
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
            mode='edge', op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'data': data, 'pads': pads})
        self.assertEqualArray(exp, got['Y'])

        for mode in ('edge', 'reflect'):
            onx = OnnxPad(
                'data', 'pads', output_names=['Y'],
                mode=mode, op_version=TARGET_OPSET)
            model_def = onx.to_onnx({'data': data, 'pads': pads},
                                    target_opset=TARGET_OPSET)

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
    def test_onnxt_runtime_prelu(self):
        x = numpy.random.randn(1, 3, 4, 5).astype(numpy.float32)
        slope = numpy.array([3]).astype(numpy.float32)
        onx = OnnxPRelu(
            'x', 'slope', output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'x': x, 'slope': slope},
                                outputs={'Y': x},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        exp = numpy.where(x > 0, x, x * slope)
        got = oinf.run({'x': x, 'slope': slope})
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxPRelu)

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
        x_zero_point = numpy.array(132, dtype=numpy.uint8)

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
                               op_version=TARGET_OPSET)
        inputs = {'x': x, 'x_scale': x_scale, 'x_zero_point': x_zero_point,
                  'w': w, 'w_scale': w_scale, 'w_zero_point': w_zero_point,
                  'y_scale': y_scale, 'y_zero_point': y_zero_point}
        model_def = node.to_onnx(inputs,
                                 target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxQLinearConv, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run(inputs)
        self.assertEqualArray(output, got['y'])
        self.common_expected_shapes_types(
            oinf, inputs, got, OnnxQLinearConv, model_def)
        python_tested.append(OnnxQLinearConv)

    @wraplog()
    def test_onnxt_runtime_qlinear_conv_test0(self):
        x_scale = numpy.float32(0.00369204697)
        x_zero_point = numpy.uint8(132)
        x = numpy.array(
            [[255, 174, 162, 25, 203, 168, 58],
             [15, 59, 237, 95, 129, 0, 64],
             [56, 242, 153, 221, 168, 12, 166],
             [232, 178, 186, 195, 237, 162, 237],
             [188, 39, 124, 77, 80, 102, 43],
             [127, 230, 21, 83, 41, 40, 134],
             [255, 154, 92, 141, 42, 148, 247], ],
            dtype=numpy.uint8).reshape((1, 1, 7, 7))

        w_scale = numpy.array([0.00172794575], dtype=numpy.float32)
        w_zero_point = numpy.array([255], dtype=numpy.uint8)
        w = numpy.array([0], dtype=numpy.uint8).reshape((1, 1, 1, 1))

        y_scale = numpy.float32(0.00162681262)
        y_zero_point = numpy.uint8(123)
        y = numpy.array(
            [[0, 81, 93, 230, 52, 87, 197],
             [240, 196, 18, 160, 126, 255, 191],
             [199, 13, 102, 34, 87, 243, 89],
             [23, 77, 69, 60, 18, 93, 18],
             [67, 216, 131, 178, 175, 153, 212],
             [128, 25, 234, 172, 214, 215, 121],
             [0, 101, 163, 114, 213, 107, 8], ],
            dtype=numpy.uint8).reshape((1, 1, 7, 7))

        test_qlinear_conv(
            QuantizedTensor(x, x_scale, x_zero_point), (1, 1, 7, 7),
            QuantizedTensor(w, w_scale, w_zero_point), (1, 1, 1, 1),
            None,
            QuantizedTensor(y, y_scale, y_zero_point), (1, 1, 7, 7))

    @wraplog()
    def test_onnxt_runtime_qlinear_conv_2dtest(self):
        x = QuantizedTensor(numpy.array([
            0.45246148109436035, 0.15498268604278564, 0.11199361085891724, -0.39421093463897705,
            0.2626858949661255, 0.13414543867111206, -
            0.27184486389160156, -0.43028733134269714,
            -0.26825493574142456, 0.3893144130706787, -
            0.13631996512413025, -0.009590476751327515,
            -0.48771554231643677, -0.25256502628326416, -
            0.2812897562980652, 0.4043201804161072,
            0.07795023918151855, 0.326981782913208, 0.13114392757415771, -0.4416425824165344,
            0.12446999549865723, 0.36739975214004517, 0.1698915958404541, 0.2008744478225708,
            0.23339951038360596, 0.38613730669021606, 0.11117297410964966, 0.3877097964286804,
            0.20812749862670898, -0.34297940135002136, -
            0.029246658086776733, -0.20483523607254028,
            -0.19244328141212463, -0.11104947328567505, -
            0.32830488681793213, -0.01800677180290222,
            0.3618946671485901, -0.40949052572250366, -
            0.18248388171195984, -0.3349453806877136,
            -0.34091079235076904, 0.006497859954833984, 0.4537564516067505, 0.08006560802459717,
            -0.14788749814033508, 0.034442365169525146, -
            0.33322954177856445, 0.06049239635467529,
            0.42619407176971436], dtype=numpy.float32))
        w = QuantizedTensor(numpy.array(
            [-0.4406261742115021], dtype=numpy.float32))
        y = QuantizedTensor(numpy.array([
            -0.19936637580394745, -0.06828942894935608, -
            0.04934731498360634, 0.17369966208934784,
            -0.11574628204107285, -0.05910799279808998, 0.1197819635272026, 0.18959586322307587,
            0.1182001456618309, -0.17154212296009064, 0.06006614491343498, 0.0042258151806890965,
            0.21490024030208588, 0.11128675937652588, 0.12394362688064575, -0.17815405130386353,
            -0.034346915781497955, -0.14407673478126526, -
            0.05778544768691063, 0.19459928572177887,
            -0.05484473705291748, -0.16188594698905945, -
            0.07485868036746979, -0.08851054310798645,
            -0.10284193605184555, -0.17014220356941223, -
            0.04898572340607643, -0.17083507776260376,
            -0.09170642495155334, 0.1511256992816925, 0.012886842712759972, 0.09025576710700989,
            0.08479554951190948, 0.0489313043653965, 0.14465972781181335, 0.007934254594147205,
            -0.15946026146411896, 0.1804322451353073, 0.08040717244148254, 0.1475857049226761,
            0.15021422505378723, -0.0028631272725760937, -
            0.19993697106838226, -0.03527900204062462,
            0.06516310572624207, -0.015176207758486271, 0.14682966470718384, -0.02665453404188156,
            -0.18779225647449493], dtype=numpy.float32))
        test_qlinear_conv(x, (1, 1, 7, 7), w, (1, 1, 1, 1),
                          None, y, (1, 1, 7, 7),
                          opset=10)

    @wraplog()
    def test_onnxt_runtime_qlinear_conv_3dtest(self):
        x = QuantizedTensor(numpy.array([
            0.010772407054901123, -0.43806642293930054, 0.455391526222229, -0.28657248616218567,
            0.45676887035369873, -0.0320507287979126, 0.4229400157928467, -0.18730869889259338,
            -0.45851585268974304, 0.042054951190948486, -
            0.13332295417785645, -0.25374430418014526,
            -0.23845627903938293, 0.12214112281799316, -
            0.1778157651424408, 0.1891845464706421,
            0.37962496280670166, -0.033982306718826294, 0.12737131118774414, -0.040284961462020874,
            0.46427029371261597, -0.22687292098999023, 0.17398333549499512, -0.3014046251773834,
            -0.4043419063091278, -0.33206477761268616, 0.04655301570892334, -0.4947906732559204,
            0.0755157470703125, 0.1173025369644165, 0.47043120861053467, 0.4824737310409546,
            -0.37734976410865784, -0.056491583585739136, -
            0.10790631175041199, 0.043476223945617676,
            0.24469023942947388, -0.4100031852722168, 0.0616222620010376, 0.2296960949897766,
            0.27883386611938477, 0.08150351047515869, 0.2453773021697998, 0.08250969648361206,
            -0.1471814215183258, -0.43011274933815, 0.027180075645446777, 0.3605625033378601,
            0.24954384565353394, -0.22505927085876465, -
            0.36272895336151123, -0.47674262523651123,
            0.11275297403335571, 0.49773406982421875, 0.2686365246772766, 0.025525271892547607,
            -0.3037869930267334, 0.41126757860183716, 0.36149072647094727, 0.00883406400680542,
            -0.07959523797035217, 0.3601323366165161, 0.17322391271591187, -0.012007325887680054], dtype=numpy.float32))
        w = QuantizedTensor(numpy.array(
            [0.32824617624282837], dtype=numpy.float32))
        y = QuantizedTensor(numpy.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0035360013134777546, 0.14948052167892456, 0.0,
            0.0, -0.15050607919692993, -0.043762750923633575, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -0.12386361509561539, -0.03541983291506767, 0.0,
            0.0, 0.09152615070343018, 0.08054415881633759, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=numpy.float32))
        test_qlinear_conv(x, (1, 1, 4, 4, 4), w, (1, 1, 1, 1, 1),
                          None, y, (1, 1, 4, 4, 4),
                          opset=10,
                          pads=[2, 2, 2, 2, 2, 2],
                          strides=[2, 2, 2])

    @wraplog()
    def test_onnxt_runtime_qlinear_conv_2dtest_with_bias(self):
        x = QuantizedTensor(numpy.array([
            6, 81, 214, 151, 234, 42, 50, 89, 30, 91, 125, 141, 52, 31, 58, 224, 84, 251, 67, 137,
            223, 119, 79, 220, 249, 75, 131, 246, 113, 56, 54, 197, 110, 142, 126, 171, 53, 228,
            240, 83, 229, 218, 185, 9, 80, 116, 176, 193, 175, 253], dtype=numpy.uint8),
            0.01, 135)
        w = QuantizedTensor(numpy.array([
            234, 229, 13, 187, 98, 161, 246, 188, 252, 107, 49, 72, 53, 212, 175, 47, 21, 14, 86,
            230, 16, 177, 82, 166, 75, 220, 169, 119, 34, 205, 27, 9, 44, 74, 40, 8, 28, 139, 240,
            106, 63, 2, 255, 156, 128, 222, 73, 51, 66, 48, 81, 247, 180, 91, 206, 239, 190, 146,
            227, 235, 10, 130, 95, 232, 121, 133, 231, 162, 108, 105, 254, 143], dtype=numpy.uint8),
            0.15, 110)
        y = QuantizedTensor(numpy.array([
            67, 81, 66, 75, 71, 101, 20, 8, 44, 94, 83, 73, 133, 125, 54, 144, 165, 56, 53, 88,
            130, 118, 170, 168, 140, 109, 103, 80, 122, 142, 129, 100, 39, 61, 141, 133, 59, 155,
            68, 129, 74, 132, 83, 143, 146, 152, 81, 127, 82, 112, 131, 64, 82, 68, 93, 149, 146,
            137, 201, 118, 112, 183, 171, 144, 85, 122, 86, 63, 163, 245, 95, 152, 126, 80, 82,
            49, 136, 160, 187, 147, 29, 20, 135, 174, 126, 124, 36, 56, 0, 83, 134, 171, 119, 109,
            85, 155, 157, 167, 194, 130], dtype=numpy.uint8), 0.75, 121)
        b = QuantizedBiasTensor(
            numpy.array([-1123, 3212, 1723, -621], dtype=numpy.int32),
            x.scale_ * w.scale_)
        test_qlinear_conv(x, (1, 2, 5, 5), w, (4, 2, 3, 3),
                          b, y, (1, 4, 5, 5),
                          opset=10,
                          pads=[1, 1, 1, 1])

    @wraplog()
    def test_onnxt_runtime_qlinear_conv_2dtest_with_group(self):
        x = QuantizedTensor(numpy.array([
            98, 166, 219, 195, 46, 97, 27, 211, 239, 1, 28, 208, 143, 144, 215, 252, 79, 5, 154,
            56, 122, 191, 94, 25, 221, 48, 37, 182, 68, 245, 210, 206, 183, 22, 163, 104, 242,
            112, 161, 66, 181, 235, 117, 75, 236, 61, 115, 36, 120, 253, 165, 214, 159, 132, 11,
            201, 30, 249, 89, 171, 186, 67, 225, 197, 135, 142, 241, 169, 170, 164, 178, 58, 50,
            51, 200, 43, 199, 126, 222, 123, 227, 42, 3, 21, 124, 220, 24, 47, 63, 110], dtype=numpy.uint8),
            0.01, 135)
        w = QuantizedTensor(numpy.array([
            220, 111, 73, 254, 235, 151, 6, 156, 129, 204, 234, 198, 44, 89, 202, 82, 118, 189,
            71, 120, 123, 121, 110, 83, 173, 248, 108, 229, 124, 68, 85, 239, 133, 213, 112, 122,
            170, 231, 225, 195, 192, 9, 232, 97, 160, 227, 67, 137], dtype=numpy.uint8),
            0.15, 110)
        y = QuantizedTensor(numpy.array([
            113, 128, 70, 64, 125, 162, 80, 189, 112, 147, 121, 111, 96, 68, 94, 101, 77, 88, 223,
            128, 163, 194, 138, 164, 122, 109, 117, 91, 72, 121, 134, 155, 127, 125, 98, 128], dtype=numpy.uint8),
            0.75, 121)
        b = QuantizedBiasTensor(
            numpy.array([-1853, 598, -17854, 14592, 42, -366],
                        dtype=numpy.int32),
            x.scale_ * w.scale_)
        test_qlinear_conv(x, (1, 6, 3, 5), w, (6, 2, 2, 2),
                          b, y, (1, 6, 2, 3),
                          opset=10,
                          pads=[0, 0, 1, 1],
                          group=3,
                          strides=[2, 2])

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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxQuantizeLinear, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxQuantizeLinear, model_def)

        X = numpy.array([0, 2, 4, 1000, -254, -1000]).astype(numpy.float32)
        y_scale = numpy.array([2], dtype=numpy.float32)
        y_zero_point = numpy.array([128], dtype=numpy.uint8)
        exp = numpy.array([128, 129, 130, 255, 1, 0]).astype(numpy.uint8)
        onx = OnnxQuantizeLinear(
            'X', y_scale, y_zero_point, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['Y'])
        python_tested.append(OnnxQuantizeLinear)

    @wraplog()
    def test_onnxt_runtime_range(self):
        starts = numpy.array([0], dtype=numpy.float32)
        ends = numpy.array([10], dtype=numpy.float32)
        steps = numpy.array([4], dtype=numpy.float32)
        onx = OnnxRange(
            'starts', 'ends', steps, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'starts': starts, 'ends': ends},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxRange, model_def)
        oinf = OnnxInference(model_def)
        exp = numpy.array([0, 4, 8], dtype=numpy.float32)
        got = oinf.run({'starts': starts, 'ends': ends})
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'starts': starts, 'ends': ends}, got, OnnxRange, model_def)
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
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceL1, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l1(X, axis=1, keepdims=0),
                              got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceL1, model_def)

        onx = OnnxReduceL1('X', output_names=['Y'], axes=1,
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l1(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceL1('X', output_names=['Y'], axes=1, keepdims=1,
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceL1, model_def)
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
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceL2, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l2(X, axis=1, keepdims=0),
                              got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got, OnnxReduceL2,
            model_def)

        onx = OnnxReduceL2('X', output_names=['Y'], axes=1,
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceL2, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l2(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceL2('X', output_names=['Y'], axes=1, keepdims=1,
                           op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceL2, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(reduce_l2(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceL2)

    @wraplog()
    def test_onnxt_runtime_reduce_log_sum(self):
        X = numpy.array([[2, 1], [4, 1]], dtype=float)

        onx = OnnxReduceLogSum('X', output_names=['Y'], keepdims=0,
                               op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(X))
        self.assertEqualArray(res, got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceLogSum, model_def)
        python_tested.append(OnnxReduceLogSum)

    @wraplog()
    def test_onnxt_runtime_reduce_log_sum_exp(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceLogSumExp('X', output_names=['Y'], keepdims=0,
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X)))
        self.assertEqualArray(res, got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceLogSumExp, model_def)

        onx = OnnxReduceLogSumExp('X', output_names=['Y'], axes=[1],
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1))
        self.assertEqualArray(res, got['Y'].ravel())

        onx = OnnxReduceLogSumExp(
            'X', output_names=['Y'], axes=[1], keepdims=1,
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.log(numpy.sum(numpy.exp(X), axis=1))
        self.assertEqualArray(res, got['Y'], decimal=5)
        python_tested.append(OnnxReduceLogSumExp)

    @wraplog()
    def test_onnxt_runtime_reduce_max(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceMax('X', output_names=['Y'], keepdims=0,
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceMax, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, keepdims=False, axis=None),  # pylint: disable=E1101,E1123
                              got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceMax, model_def)

        onx = OnnxReduceMax('X', output_names=['Y'], axes=[1],
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceMax, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.maximum.reduce(X, axis=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

        onx = OnnxReduceMax('X', output_names=['Y'], axes=[1], keepdims=1,
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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
                             op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceMean, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X), got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceMean, model_def)

        onx = OnnxReduceMean('X', output_names=['Y'], axes=1,
                             op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceMean, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.mean(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceMean('X', output_names=['Y'], axes=1, keepdims=1,
                             op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceMean, model_def)
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
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, keepdims=False, axis=None),  # pylint: disable=E1101,E1123
                              got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceMin, model_def)

        onx = OnnxReduceMin('X', output_names=['Y'], axes=[1],
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.minimum.reduce(X, axis=1).ravel(),  # pylint: disable=E1101
                              got['Y'].ravel())

        onx = OnnxReduceMin('X', output_names=['Y'], axes=[1], keepdims=1,
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
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
                             op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X), got['Y'], decimal=5)

        onx = OnnxReduceProd('X', output_names=['Y'], axes=1,
                             op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X, axis=1).ravel(),
                              got['Y'].ravel())
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxReduceProd, model_def)

        onx = OnnxReduceProd('X', output_names=['Y'], axes=1, keepdims=1,
                             op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.prod(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceProd)

    @wraplog()
    def test_onnxt_runtime_reduce_sum(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        # opset=13, 14, ...
        for opset in (10, 11, 12, 13, 14, 15, TARGET_OPSET):
            if onnx_opset_version() < opset:
                continue
            if opset < 13:
                cl = OnnxReduceSum_11 if opset >= 11 else OnnxReduceSum_1
                onx = cl('X', output_names=['Y'], keepdims=0,
                         op_version=opset)
                model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxReduceSum, model_def)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X.astype(numpy.float32)})
                self.assertEqual(list(sorted(got)), ['Y'])
                self.assertEqualArray(numpy.sum(X), got['Y'], decimal=5)
                name = oinf.sequence_[0].ops_.__class__.__name__
                if opset < 11:
                    self.assertEqual(name, 'ReduceSum_1')
                else:
                    self.assertEqual(name, 'ReduceSum_11')
                self.common_expected_shapes_types(
                    oinf, {'X': X.astype(numpy.float32)}, got,
                    OnnxReduceSum, model_def)

            onx = OnnxReduceSumApi11('X', output_names=['Y'], axes=1,
                                     op_version=opset)
            model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                    target_opset=opset)
            self._check_shape_inference(OnnxReduceSum, model_def)
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

        for opset in (11, 12, 13, 14):  # opset=13, 14, ...
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
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        res = numpy.sum(X, axis=1)
        self.assertEqualArray(res, got['Y'], decimal=5)
        python_tested.append(OnnxReduceSum)

    @wraplog()
    def test_onnxt_runtime_reduce_sum_noop_with_empty_axes(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        for opset in range(13, TARGET_OPSET + 1):
            if onnx_opset_version() < opset:
                continue
            cl = OnnxReduceSum_13
            onx = cl('X', numpy.array([0], dtype=numpy.int64),
                     output_names=['Y'], keepdims=0,
                     op_version=opset, noop_with_empty_axes=1)
            model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                    target_opset=opset)
            self._check_shape_inference(OnnxReduceSum, model_def)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': X.astype(numpy.float32)})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(numpy.sum(X, axis=0), got['Y'], decimal=5)
            name = oinf.sequence_[0].ops_.__class__.__name__
            self.assertEqual(name, 'ReduceSum_13')
            self.common_expected_shapes_types(
                oinf, {'X': X.astype(numpy.float32)}, got,
                OnnxReduceSum, model_def)

        for opset in range(13, TARGET_OPSET + 1):
            if onnx_opset_version() < opset:
                continue
            cl = OnnxReduceSum_13
            onx = cl('X', numpy.array([], dtype=numpy.int64),
                     output_names=['Y'], keepdims=0,
                     op_version=opset, noop_with_empty_axes=1)
            model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                    target_opset=opset)
            oinf = OnnxInference(model_def)
            got = oinf.run({'X': X.astype(numpy.float32)})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(X, got['Y'], decimal=5)

    @wraplog()
    def test_onnxt_runtime_reduce_sum_square(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], keepdims=0,
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReduceSumSquare, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X)), got['Y'], decimal=5)
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReduceSumSquare, model_def)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1,
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1, keepdims=1,
                                  op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())
        python_tested.append(OnnxReduceSumSquare)

    @wraplog()
    def test_onnxt_runtime_reduce_sum_noop(self):
        X = numpy.array([], dtype=float).reshape((2, 0))

        # opset=13, 14, ...
        for opset in (13, 14, 15, TARGET_OPSET):
            if onnx_opset_version() < opset:
                continue

            node = onnx.helper.make_node(
                'ReduceSum', inputs=['X'], outputs=['Y'],
                keepdims=0, noop_with_empty_axes=0)
            oX = onnx.helper.make_tensor_value_info(
                'X', onnx.TensorProto.FLOAT, [None, None])  # pylint: disable=E1101
            oY = onnx.helper.make_tensor_value_info(
                'Y', onnx.TensorProto.FLOAT, [None, None])  # pylint: disable=E1101

            graph_def = onnx.helper.make_graph(
                [node], 'test-model', [oX], [oY])
            model_def = onnx.helper.make_model(
                graph_def, producer_name='mlprodict', ir_version=7,
                producer_version='0.1',
                opset_imports=[onnx.helper.make_operatorsetid('', opset)])

            oinf = OnnxInference(model_def)
            got = oinf.run({'X': X})
            self.assertEqual(list(sorted(got)), ['Y'])
            self.assertEqualArray(numpy.sum(X), got['Y'], decimal=5)
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
                          op_version=TARGET_OPSET)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxReshape, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = X.reshape(sh.tolist())
        self.assertEqualArray(exp, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X.astype(numpy.float32)}, got,
            OnnxReshape, model_def)
        python_tested.append(OnnxReshape)

    @wraplog()
    def test_onnxt_runtime_scatter_elements1(self):
        for opset in [11, TARGET_OPSET]:
            if opset > TARGET_OPSET:
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
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': data})
                self.assertEqualArray(output, got['Y'])
                self.common_expected_shapes_types(
                    oinf, {'X': data}, got, OnnxScatterElements, model_def)

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
        for opset in [11, TARGET_OPSET]:
            if opset > TARGET_OPSET:
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
    def test_onnxt_runtime_selu(self):
        alpha = 1.67326319217681884765625
        gamma = 1.05070102214813232421875
        self.common_test_onnxt_runtime_unary(
            OnnxSelu, lambda x: numpy.where(
                x > 0, x * gamma, numpy.exp(x) * alpha - alpha))

    @wraplog()
    def test_onnxt_runtime_sequence_at(self):
        x = numpy.random.randn(20, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        onx = OnnxSequenceAt(
            OnnxSequenceConstruct(
                'X', 'X', 'X',
                op_version=TARGET_OPSET),
            numpy.array(1, dtype=numpy.int64),
            op_version=TARGET_OPSET,
            output_names=['Y'])

        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        output = got['Y']
        self.assertEqualArray(x, output)
        python_tested.append(OnnxSequenceAt)

    @wraplog()
    def test_onnxt_runtime_sequence_construct(self):
        x = numpy.random.randn(20, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        onx = OnnxSequenceConstruct(
            'X', 'X', 'X', output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        output = got['Y']
        self.assertEqualArray(len(output), 3)
        for i in range(0, len(output)):  # pylint: disable=C0200
            self.assertEqualArray(x, output[i])
        python_tested.append(OnnxSequenceConstruct)

    @wraplog()
    def test_onnxt_runtime_shape(self):
        x = numpy.random.randn(20, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        y = x.shape
        onx = OnnxShape('X', output_names=['Y'],
                        op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxShape, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqualArray(y, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x}, got, OnnxShape, model_def)
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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxSize, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': x})
        self.assertEqualArray(y, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': x}, got, OnnxSize, model_def)
        python_tested.append(OnnxSize)

    @wraplog()
    def test_onnxt_runtime_slice(self):
        for opset in range(9, TARGET_OPSET + 1):
            if opset > TARGET_OPSET:
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
                self._check_shape_inference(OnnxSlice, model_def)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': x})
                self.assertEqualArray(y, got['Y'])
                self.common_expected_shapes_types(
                    oinf, {'X': x}, got, OnnxSlice, model_def)

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
                self._check_shape_inference(OnnxSlice, model_def)
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
                self._check_shape_inference(OnnxSlice, model_def)
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
        # opset=13, 14, ...
        for opset in [13, 14, 15, TARGET_OPSET]:
            if opset > TARGET_OPSET:
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
        # opset=13, 14, ...
        for opset in [10, 11, 12, 13, 14, 15, TARGET_OPSET]:
            if opset > TARGET_OPSET:
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
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': x})
                self.assertEqualArray(y[0], got['Y1'])
                self.assertEqualArray(y[1], got['Y2'])
                self.assertEqualArray(y[2], got['Y3'])
                self.common_expected_shapes_types(
                    oinf, {'X': x}, got, OnnxSplit, model_def)

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
        # opset=13, 14, ...
        for opset in [10, 11, 12, 13, 14, 15, TARGET_OPSET]:
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                x = numpy.random.randn(20, 1).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                y = numpy.squeeze(x)
                onx = OnnxSqueezeApi11(
                    'X', axes=[1], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxSqueeze, model_def)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': x})
                self.assertEqualArray(y, got['Y'])
                self.common_expected_shapes_types(
                    oinf, {'X': x}, got, OnnxSqueeze, model_def)

                x = numpy.random.randn(1, 20).astype(  # pylint: disable=E1101
                    numpy.float32)  # pylint: disable=E1101
                y = numpy.squeeze(x)
                onx = OnnxSqueezeApi11(
                    'X', axes=[0], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxSqueeze, model_def)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxSqueeze)

    @wraplog()
    def test_onnxt_runtime_softmax(self):
        self.common_test_onnxt_runtime_unary(OnnxSoftmax, softmax)

    @wraplog()
    def test_softmax_cross_entropy_loss(self):

        def _make_model(node, opset=15):
            ginputs = [
                onnx.helper.make_tensor_value_info(
                    name, (TensorProto.FLOAT if i % 2 == 0 else TensorProto.INT64), [])
                for i, name in enumerate(node.input)]
            goutputs = [
                onnx.helper.make_tensor_value_info(o, TensorProto.FLOAT, [])
                for o in node.output]
            model_def = onnx.helper.make_model(
                opset_imports=[onnx.helper.make_operatorsetid('', opset)],
                graph=onnx.helper.make_graph(
                    name='test_softmax_cross_entropy_loss',
                    inputs=ginputs, outputs=goutputs,
                    nodes=[node]))
            return model_def

        reduction = 'mean'
        ignore_index = numpy.int64(-1)
        node = onnx.helper.make_node(
            'SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'],
            outputs=['z'], reduction=reduction, ignore_index=ignore_index)
        model_def = _make_model(node)

        N, C, dim1 = 3, 5, 6
        numpy.random.seed(0)
        x = numpy.random.rand(N, C, dim1).astype(numpy.float32)
        labels = numpy.random.randint(0, high=C, size=(N, dim1)).astype(numpy.int64)
        labels[0, 0] = -1
        weight = numpy.random.rand(C).astype(numpy.float32)

        outputs = softmaxcrossentropy(
            x, labels, weight=weight, reduction=reduction,
            ignore_index=ignore_index)

        oinf = OnnxInference(model_def)
        got = oinf.run({'x': x, 'y': labels, 'w': weight})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(outputs, got['z'])
        python_tested.append(OnnxSoftmaxCrossEntropyLoss)

    @wraplog()
    def test_softmax_cross_entropy_loss_multi_output(self):

        def _make_model(node, opset=15):
            ginputs = [
                onnx.helper.make_tensor_value_info(
                    name, (TensorProto.FLOAT if i % 2 == 0 else TensorProto.INT64), [])
                for i, name in enumerate(node.input)]
            goutputs = [
                onnx.helper.make_tensor_value_info(o, TensorProto.FLOAT, [])
                for o in node.output]
            model_def = onnx.helper.make_model(
                opset_imports=[onnx.helper.make_operatorsetid('', opset)],
                graph=onnx.helper.make_graph(
                    name='test_softmax_cross_entropy_loss',
                    inputs=ginputs, outputs=goutputs,
                    nodes=[node]))
            return model_def

        reduction = 'none'
        ignore_index = numpy.int64(-5)
        node = onnx.helper.make_node(
            'SoftmaxCrossEntropyLoss', inputs=['x', 'y'],
            outputs=['z', 'log_prob'], reduction=reduction, ignore_index=ignore_index)
        model_def = _make_model(node)

        N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
        numpy.random.seed(0)
        x = numpy.random.rand(N, C, dim1, dim2, dim3).astype(numpy.float32)
        labels = numpy.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(numpy.int64)
        labels[0][0][0][0] = -5

        outputs = softmaxcrossentropy(
            x, labels, reduction=reduction,
            ignore_index=ignore_index, get_log_prob=True)

        oinf = OnnxInference(model_def)
        got = oinf.run({'x': x, 'y': labels})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(outputs[0], got['z'])
        self.assertEqualArray(outputs[1], got['log_prob'])

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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxTopK, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
        self.assertEqual(got['Y'].size, 0)
        self.assertEqual(got['Yi'].size, 0)
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxTopK, model_def)

    @wraplog()
    def test_onnxt_runtime_topk(self):
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        # axis=1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64),
                       axis=1, output_names=['Y', 'Yi'],
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=TARGET_OPSET)
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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxTopK, model_def)
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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=TARGET_OPSET)
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
                       op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType(X.shape)),
                                         ('Yi', Int64TensorType(X.shape))],
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxTopK, model_def)
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
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxTranspose, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X, got['Y'])
        self.common_expected_shapes_types(
            oinf, {'X': X}, got, OnnxTranspose, model_def)

        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        onx = OnnxTranspose('X', perm=[1, 0], output_names=['Y'],
                            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self._check_shape_inference(OnnxTranspose, model_def)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X.T, got['Y'])
        python_tested.append(OnnxTranspose)

    @wraplog()
    def test_onnxt_runtime_unsqueeze(self):
        # opset=13, 14, ...
        for opset in [10, 11, 12, 13, 14, 15, TARGET_OPSET]:
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                x = numpy.random.randn(1, 3, 1, 5).astype(numpy.float32)
                y = numpy.expand_dims(x, axis=-2)
                onx = OnnxUnsqueezeApi11(
                    'X', axes=[-2], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxUnsqueeze, model_def)
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': x})
                self.assertEqualArray(y, got['Y'])
                self.common_expected_shapes_types(
                    oinf, {'X': x}, got, OnnxUnsqueeze, model_def)

                x = numpy.random.randn(3, 4, 5).astype(numpy.float32)
                y = numpy.expand_dims(x, axis=2)
                y = numpy.expand_dims(y, axis=4)
                y = numpy.expand_dims(y, axis=5)
                onx = OnnxUnsqueezeApi11(
                    'X', axes=[2, 4, 5], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx({'X': x.astype(numpy.float32)},
                                        target_opset=opset)
                self._check_shape_inference(OnnxUnsqueeze, model_def)
                got = OnnxInference(model_def).run({'X': x})
                self.assertEqualArray(y, got['Y'])
        python_tested.append(OnnxUnsqueeze)

    @wraplog()
    def test_onnxt_runtime_trilu(self):
        self.common_test_onnxt_runtime_unary(
            OnnxTrilu, lambda x: numpy.triu(x, 0))

    @wraplog()
    def test_onnxt_runtime_xor(self):
        self.common_test_onnxt_runtime_binary(
            OnnxXor, numpy.logical_xor, dtype=numpy.bool_)


if __name__ == "__main__":
    # Working
    # TestOnnxrtPythonRuntime().test_softmax_cross_entropy_loss_multi_output()
    unittest.main(verbosity=2)
