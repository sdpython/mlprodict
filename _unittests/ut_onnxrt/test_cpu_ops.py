"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
import onnx
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxConv)
from mlprodict.onnxrt.ops_cpu.op_conv import Conv
from mlprodict.onnxrt.onnx2py_helper import _var_as_dict
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
from mlprodict.onnxrt import OnnxInference


class TestCpuOps(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

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

    def test_cpu_conv_init(self):
        x = numpy.random.rand(1, 96, 56, 56).astype(numpy.float32)
        W = numpy.random.rand(24, 96, 1, 1).astype(numpy.float32)

        onx = OnnxConv(
            'X', W, output_names=['Y'],
            auto_pad='NOTSET', group=1, dilations=[1, 1],
            kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
            op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': x.astype(numpy.float32),
                                 'W': W.astype(numpy.float32)},
                                target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(model_def)
        oinfrt = OnnxInference(model_def, runtime='onnxruntime1')
        for _ in range(0, 3):
            x = numpy.random.rand(1, 96, 56, 56).astype(numpy.float32)
            W = numpy.random.rand(24, 96, 1, 1).astype(numpy.float32)
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


if __name__ == "__main__":
    unittest.main()
