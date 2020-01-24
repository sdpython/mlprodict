"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import onnx
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.ops_cpu.op_conv import Conv
from mlprodict.onnxrt.onnx2py_helper import _var_as_dict


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


if __name__ == "__main__":
    unittest.main()
