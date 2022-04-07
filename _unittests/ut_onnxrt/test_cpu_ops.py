"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
import onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxConv)
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt.ops_cpu.op_conv import Conv
from mlprodict.onnx_tools.onnx2py_helper import _var_as_dict
from mlprodict.onnxrt import OnnxInference
from mlprodict.testing.test_utils.tests_helper import fit_multilabel_classification_model
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.onnxrt.ops_cpu._op_helper import dtype_name
from mlprodict.onnxrt.ops_cpu.op_conv_helper import (
    im2col, im2col_indices, col2im_indices, im2col_recursive,
    im2col_naive_implementation)


class TestCpuOps(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_dtype_name(self):
        self.assertEqual(dtype_name(numpy.float32), "float32")
        self.assertEqual(dtype_name(numpy.float64), "float64")
        self.assertEqual(dtype_name(numpy.float16), "float16")
        self.assertEqual(dtype_name(numpy.int64), "int64")
        self.assertEqual(dtype_name(numpy.int32), "int32")
        self.assertEqual(dtype_name(numpy.uint32), "uint32")
        self.assertEqual(dtype_name(numpy.int8), "int8")
        self.assertEqual(dtype_name(numpy.uint8), "uint8")
        self.assertEqual(dtype_name(numpy.str_), "str")
        self.assertEqual(dtype_name(numpy.bool_), "bool")
        self.assertRaise(lambda: dtype_name(numpy.int16), ValueError)

    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_cpu_conv(self):

        x = numpy.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                           [5., 6., 7., 8., 9.],
                           [10., 11., 12., 13., 14.],
                           [15., 16., 17., 18., 19.],
                           [20., 21., 22., 23., 24.]]]]).astype(numpy.float32)
        W = numpy.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                           [1., 1., 1.],
                           [1., 1., 1.]]]]).astype(numpy.float32)

        node_with_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1],
            # dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )
        atts = _var_as_dict(node_with_padding)
        cv = Conv(node_with_padding, desc=atts)
        got = cv.run(x, W)[0]
        exp = numpy.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                             [33., 54., 63., 72., 51.],
                             [63., 99., 108., 117., 81.],
                             [93., 144., 153., 162., 111.],
                             [72., 111., 117., 123., 84.]]]]).astype(numpy.float32)
        self.assertEqualArray(exp, got)

    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_cpu_conv_init(self):
        x = numpy.random.rand(1, 96, 56, 56).astype(numpy.float32)
        W = numpy.random.rand(24, 96, 1, 1).astype(numpy.float32)

        onx = OnnxConv(
            'X', 'W', output_names=['Y'],
            auto_pad='NOTSET', group=1, dilations=[1, 1],
            kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32),
                                 'W': W.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        oinfrt = OnnxInference(model_def, runtime='onnxruntime1')
        for _ in range(0, 3):
            x = numpy.random.rand(1, 96, 56, 56).astype(numpy.float32)
            W = numpy.random.rand(24, 96, 1, 1).astype(numpy.float32)
            got = oinf.run({'X': x, 'W': W})
            gotrt = oinfrt.run({'X': x, 'W': W})
            diff = list(numpy.abs((gotrt['Y'] - got['Y']).ravel()))
            sdiff = list(sorted(diff))
            if sdiff[-1] > 3e-5:
                raise AssertionError("runtimes disagree {}".format(sdiff[-5:]))
            for ii in range(len(diff)):  # pylint: disable=C0200
                if numpy.isnan(diff[ii]):
                    raise AssertionError(
                        "runtimes disagree about nan {}: {} # {} ? {}".format(
                            ii, diff[ii], gotrt['Y'].ravel()[ii], got['Y'].ravel()[ii]))
            self.assertEqualArray(gotrt['Y'], got['Y'], decimal=5)

    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_cpu_conv_group(self):
        x = numpy.random.rand(1, 3, 3, 4).astype(numpy.float32)
        W = numpy.random.rand(9, 1, 3, 3).astype(numpy.float32)

        onx = OnnxConv(
            'X', 'W', output_names=['Y'],
            auto_pad='NOTSET', group=3, dilations=[1, 1],
            kernel_shape=[3, 3], strides=[1, 1],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(numpy.float32),
                                 'W': W.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        oinfrt = OnnxInference(model_def, runtime='onnxruntime1')
        d = oinf.sequence_[-1].ops_.atts_value
        self.assertIsInstance(d, dict)
        self.assertEqual(d['kernel_shape'].tolist(), [3, 3])

        xs = [
            numpy.random.rand(1, 3, 3, 4).astype(numpy.float32),
            numpy.array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0,
                         34.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0,
                         32.0, 35.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0,
                         30.0, 33.0, 36.0], dtype=numpy.float32).reshape((1, 3, 3, 4))]
        Ws = [
            numpy.random.rand(9, 1, 3, 3).astype(numpy.float32),
            numpy.array([1.0, 10.0, 19.0, 28.0, 37.0, 46.0, 55.0, 64.0,
                         73.0, 2.0, 11.0, 20.0, 29.0, 38.0, 47.0, 56.0, 65.0, 74.0,
                         3.0, 12.0, 21.0, 30.0, 39.0, 48.0, 57.0, 66.0, 75.0, 4.0,
                         13.0, 22.0, 31.0, 40.0, 49.0, 58.0, 67.0, 76.0, 5.0, 14.0,
                         23.0, 32.0, 41.0, 50.0, 59.0, 68.0, 77.0, 6.0, 15.0, 24.0,
                         33.0, 42.0, 51.0, 60.0, 69.0, 78.0, 7.0, 16.0, 25.0, 34.0,
                         43.0, 52.0, 61.0, 70.0, 79.0, 8.0, 17.0, 26.0, 35.0, 44.0,
                         53.0, 62.0, 71.0, 80.0, 9.0, 18.0, 27.0, 36.0, 45.0, 54.0,
                         63.0, 72.0, 81.0], dtype=numpy.float32).reshape((9, 1, 3, 3))]

        for x, W in zip(xs, Ws):
            x = numpy.asfortranarray(x)
            W = numpy.asfortranarray(W)
            got = oinf.run({'X': x, 'W': W})
            gotrt = oinfrt.run({'X': x, 'W': W})
            diff = list(numpy.abs((gotrt['Y'] - got['Y']).ravel()))
            sdiff = list(sorted(diff))
            if sdiff[-1] > 1e-5:
                raise AssertionError("runtimes disagree {}".format(sdiff[-5:]))
            for ii in range(len(diff)):  # pylint: disable=C0200
                if numpy.isnan(diff[ii]):
                    raise AssertionError(
                        "runtimes disagree about nan {}: {} # {} ? {}".format(
                            ii, diff[ii], gotrt['Y'].ravel()[ii], got['Y'].ravel()[ii]))
            self.assertEqualArray(gotrt['Y'], got['Y'], decimal=5)

    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_slice_bug(self):

        for opset in [9, 12, TARGET_OPSET]:
            if opset > TARGET_OPSET:
                continue
            model = OneVsRestClassifier(
                RandomForestClassifier(n_estimators=2, max_depth=3))
            model, X = fit_multilabel_classification_model(
                model, 3, is_int=False, n_features=5)
            model_onnx = to_onnx(
                model, X[:1], target_opset=opset,
                options={id(model): {'zipmap': False}})
            X = X[:7]
            for rt in ['python', 'onnxruntime1']:
                with self.subTest(opset=opset, rt=rt):
                    oinf = OnnxInference(model_onnx, runtime=rt)
                    got = oinf.run({'X': X})
                    exp = model.predict(X), model.predict_proba(X)
                    self.assertEqual(exp[1].shape[1], 3)
                    self.assertEqualArray(exp[0], got['label'])
                    self.assertEqualArray(exp[1], got['probabilities'])

    def test_im2col_indices(self):
        img = numpy.arange(35 * 3).reshape((1, 3, 5, 7)
                                           ).astype(numpy.float32) + 101
        res2 = im2col_indices(img, 3, 3, padding=0)
        self.assertEqual(res2.shape, (27, 15))
        img2 = col2im_indices(res2, x_shape=img.shape)
        self.assertEqual(img.shape, img2.shape)

        img = numpy.arange(35).reshape(
            (1, 1, 5, 7)).astype(numpy.float32) + 101
        res2 = im2col_indices(img, 3, 3, padding=0)
        self.assertEqual(res2.shape, (9, 15))
        img2 = col2im_indices(res2, x_shape=img.shape)
        self.assertEqual(img.shape, img2.shape)
        print(img.shape, res2.shape)

    def test_im2col(self):
        data = numpy.arange(5).astype(numpy.float32) + 10
        res = im2col(data, fill_value=0)
        self.assertEqual(res.shape, (5, 3))
        expected = numpy.array([[11, 10, 0], [12, 11, 10], [13, 12, 11],
                                [14, 13, 12], [0, 14, 13]], dtype=numpy.float32)
        expected = expected[:, ::-1]
        self.assertEqualArray(expected.astype(
            numpy.int16), res.astype(numpy.int16))

        data = numpy.arange(6).astype(numpy.float32) + 10
        res = im2col(data, kernel_shape=(5,), fill_value=0)
        self.assertEqual(res.shape, (6, 5))
        expected = numpy.array([[12, 11, 10, 0, 0], [13, 12, 11, 10, 0],
                                [14, 13, 12, 11, 10], [15, 14, 13, 12, 11],
                                [0, 15, 14, 13, 12], [0, 0, 15, 14, 13]],
                               dtype=numpy.int16)
        expected = expected[:, ::-1]
        self.assertEqualArray(expected.astype(
            numpy.int16), res.astype(numpy.int16))

    def test_im2col_double(self):
        data = numpy.arange(5).astype(numpy.float64) + 10
        res = im2col(data, fill_value=0)
        self.assertEqual(res.shape, (5, 3))
        expected = numpy.array([[11, 10, 0], [12, 11, 10], [13, 12, 11],
                                [14, 13, 12], [0, 14, 13]], dtype=numpy.float64)
        expected = expected[:, ::-1]
        self.assertEqualArray(expected, res)

        data = numpy.arange(6).astype(numpy.float64) + 10
        res = im2col(data, kernel_shape=(5,), fill_value=0)
        self.assertEqual(res.shape, (6, 5))
        expected = numpy.array([[12, 11, 10, 0, 0], [13, 12, 11, 10, 0],
                                [14, 13, 12, 11, 10], [15, 14, 13, 12, 11],
                                [0, 15, 14, 13, 12], [0, 0, 15, 14, 13]],
                               dtype=numpy.int64)
        expected = expected[:, ::-1]
        self.assertEqualArray(expected, res.astype(numpy.int64))

    def test_im2col_2d(self):
        data = (numpy.arange(9).astype(numpy.float64) + 10).reshape((3, 3))
        self.assertRaise(lambda: im2col(data, [6, 7]), TypeError)
        self.assertRaise(lambda: im2col(data, (3, 3, 3)), ValueError)
        res = im2col(data, (3, 3), fill_value=0)
        self.assertEqual(res.shape, (3, 3, 3, 3))
        data = (numpy.arange(25).astype(numpy.float64) + 10).reshape((5, 5))
        res = im2col(data, (5, 5), fill_value=0)
        self.assertEqual(res.shape, (5, 5, 5, 5))

    def test_im2col_2d_recursive(self):
        data = (numpy.arange(9).astype(numpy.float64) + 10).reshape((3, 3))
        res = im2col_recursive(data, (3, 3), fill_value=0, fall_back_dim=1)
        expected = im2col_naive_implementation(data, (3, 3), fill_value=0)
        self.assertEqualArray(expected, res)

        data = (numpy.arange(25).astype(numpy.float64) + 10).reshape((5, 5))
        res = im2col_recursive(data, (3, 3), fill_value=0, fall_back_dim=1)
        expected = im2col_naive_implementation(data, (3, 3), fill_value=0)
        self.assertEqualArray(expected, res)

        data = (numpy.arange(25).astype(numpy.float64) + 10).reshape((5, 5))
        res = im2col_recursive(data, (5, 5), fill_value=0, fall_back_dim=1)
        expected = im2col_naive_implementation(data, (5, 5), fill_value=0)
        self.assertEqualArray(expected, res)

        for i in range(0, 2):
            kernel_shape = [3, 3]
            kernel_shape[i] = 5
            kernel_shape = tuple(kernel_shape)
            data = (numpy.arange(25).astype(
                numpy.float64) + 10).reshape((5, 5))
            res = im2col_recursive(
                data, kernel_shape, fill_value=0, fall_back_dim=1)
            expected = im2col_naive_implementation(
                data, kernel_shape, fill_value=0)
            self.assertEqualArray(expected, res)

    def test_im2col_3d_recursive(self):
        data = (numpy.arange(27).astype(numpy.float64) + 10).reshape((3, 3, 3))
        res = im2col_recursive(data, (3, 3, 3), fill_value=0)
        expected = im2col_naive_implementation(data, (3, 3, 3), fill_value=0)
        self.assertEqualArray(expected, res)

        data = (numpy.arange(125).astype(
            numpy.float64) + 10).reshape((5, 5, 5))
        res = im2col_recursive(data, (3, 3, 3), fill_value=0)
        expected = im2col_naive_implementation(data, (3, 3, 3), fill_value=0)
        self.assertEqualArray(expected, res)

        for i in range(0, 3):
            kernel_shape = [3, 3, 3]
            kernel_shape[i] = 5
            kernel_shape = tuple(kernel_shape)
            data = (numpy.arange(125).astype(
                numpy.float64) + 10).reshape((5, 5, 5))
            res = im2col_recursive(data, kernel_shape, fill_value=0)
            expected = im2col_naive_implementation(
                data, kernel_shape, fill_value=0)
            self.assertEqualArray(expected, res)


if __name__ == "__main__":
    # TestCpuOps().test_im2col_3d()
    unittest.main()
