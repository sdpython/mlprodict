"""
@brief      test log(time=2s)
"""
import unittest
import warnings
from logging import getLogger
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_function,
    make_graph, make_tensor_value_info, make_opsetid)
from onnxruntime import InferenceSession
from pyquickhelper.pycode import (
    ExtTestCase, ignore_warnings, skipif_azure)
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.datasets import make_regression
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxMul, OnnxAdd)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx
from mlprodict import __max_supported_opset__ as TARGET_OPSET, get_ir_version


class TestOnnxrtOnnxRuntimeRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_add(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y1'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)

        model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y1'])
        self.assertEqualArray(idi + X, got['Y1'], decimal=6)

        oinf = OnnxInference(model_def, runtime='onnxruntime2')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y1'])
        self.assertEqualArray(idi + X, got['Y1'], decimal=6)

        oinf = OnnxInference(model_def, runtime='onnxruntime1', inplace=False)
        got = oinf.run({'X': X}, intermediate=True)
        self.assertEqual(list(sorted(got)), ['Ad_Addcst', 'X', 'Y1'])
        self.assertEqualArray(idi + X, got['Y1'], decimal=6)

    @ignore_warnings(DeprecationWarning)
    @skipif_azure("Failure on Mac")
    def test_onnxt_runtime_add_raise(self):
        idi = numpy.identity(2).astype(numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y2'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        self.assertRaise(lambda: OnnxInference(model_def, runtime='onnxruntime-1'),
                         ValueError)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_add1(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y3'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y3'])
        self.assertEqualArray(idi + X, got['Y3'], decimal=6)

    @ignore_warnings(DeprecationWarning)
    @skipif_azure("Failure on Mac")
    def test_onnxruntime_bug(self):
        rnd = numpy.random.randn(3, 20, 20).astype(numpy.float32)
        bni = (numpy.random.random((20, 20)).astype(  # pylint: disable=E1101
            numpy.float32) >= 0.7).astype(numpy.float32)
        mul = rnd * bni
        isn = any(numpy.isnan(mul.ravel()))
        self.assertFalse(isn)

        node = OnnxMul('X', bni, output_names=['Y4'],
                       op_version=TARGET_OPSET)
        onx = node.to_onnx({'X': rnd})
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(onx, runtime=rt)
                y = oinf.run({'X': rnd})['Y4']
                self.assertEqualArray(mul, y)

    @ignore_warnings(DeprecationWarning)
    @skipif_azure("Failure on Mac")
    def test_onnxruntime_knn_radius(self):
        def _get_reg_data(self, n, n_features, n_targets, n_informative=10):
            X, y = make_regression(  # pylint: disable=W0632
                n, n_features=n_features, random_state=0,
                n_targets=n_targets, n_informative=n_informative)
            return X, y

        def _fit_model(model, n_targets=1, label_int=False,
                       n_informative=10):
            X, y = _get_reg_data(20, 4, n_targets, n_informative)
            if label_int:
                y = y.astype(numpy.int64)
            model.fit(X, y)
            return model, X

        model, X = _fit_model(RadiusNeighborsRegressor())
        model_onnx = to_onnx(
            model, X[:1].astype(numpy.float32),
            target_opset=TARGET_OPSET,
            options={id(model): {'optim': 'cdist'}})
        oinf = OnnxInference(model_onnx, runtime='onnxruntime1')
        X = X[:7]
        got = oinf.run({'X': X.astype(numpy.float32)})['variable']
        exp = model.predict(X.astype(numpy.float32))
        if any(numpy.isnan(got.ravel())):
            # The model is unexpectedly producing nan values
            # sometimes.
            res = oinf.run({'X': X.astype(numpy.float32)}, intermediate=True)
            rows = ['--EXP--', str(exp), '--GOT--', str(got),
                    '--EVERY-OUTPUT--']
            for k, v in res.items():
                rows.append('-%s-' % k)
                rows.append(str(v))
            if any(map(numpy.isnan, res["variable"].ravel())):
                # raise AssertionError('\n'.join(rows))
                warnings.warn("Unexpected NaN values\n" + '\n'.join(rows))
                return
            # onnxruntime and mlprodict do not return the same
            # output
            warnings.warn('\n'.join(rows))
            return
        self.assertEqualArray(exp, got, decimal=4)

    def test_make_function(self):
        new_domain = 'custom'
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])

        linear_regression = make_function(
            new_domain,            # domain name
            'LinearRegression',     # function name
            ['X', 'A', 'B'],        # input names
            ['Y'],                  # output names
            [node1, node2],         # nodes
            opset_imports,          # opsets
            [])                     # attribute names

        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

        graph = make_graph(
            [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'],
                       domain=new_domain),
             make_node('Abs', ['Y1'], ['Y'])],
            'example',
            [X, A, B], [Y])

        onnx_model = make_model(
            graph, opset_imports=opset_imports,
            functions=[linear_regression])  # functions to add)

        X = numpy.array([[0, 1], [2, 3]], dtype=numpy.float32)
        A = numpy.array([[10, 11]], dtype=numpy.float32).T
        B = numpy.array([[1, -1]], dtype=numpy.float32)
        expected = X @ A + B

        with self.subTest(runtime='onnxruntime'):
            sess = InferenceSession(onnx_model.SerializeToString())
            got = sess.run(None, {'X': X, 'A': A, 'B': B})[0]
            self.assertEqualArray(expected, got)

        with self.subTest(runtime='onnxruntime1'):
            oinf = OnnxInference(onnx_model, runtime='onnxruntime1')
            got = oinf.run({'X': X, 'A': A, 'B': B})['Y']
            self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main()
