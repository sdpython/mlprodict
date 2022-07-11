# pylint: disable=W0201
"""
@brief      test log(time=5s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnx_tools.model_checker import check_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop
from mlprodict.onnx_tools.compress import compress_proto
from mlprodict.plotting.text_plot import onnx_simple_text_plot


class TestCompressOnnx(ExtTestCase):

    @ignore_warnings(RuntimeWarning)
    def test_simple_case(self):
        OnnxAdd, OnnxLog = loadop('Add', 'Log')
        opv = 5
        add = OnnxAdd('x', numpy.array([1], dtype=numpy.float32), op_version=opv)
        logx = OnnxLog(add, op_version=opv, output_names=['y'])
        onx = logx.to_onnx(numpy.float32, numpy.float32)
        check_onnx(onx)

        x = numpy.random.randn(3, 4).astype(numpy.float32)
        oinf = OnnxInference(onx)
        y = oinf.run({'x': x})['y']
        self.assertEqual(numpy.log(x + 1), y)

        # compression
        onx2 = compress_proto(onx)
        self.assertEqual(len(onx2.graph.node), 1)
        check_onnx(onx2)
        oinf2 = OnnxInference(onx2)
        y = oinf.run({'x': x})['y']
        self.assertEqual(numpy.log(x + 1), y)

        # text
        text = onnx_simple_text_plot(onx2, recursive=True)
        self.assertIn('expression=G1', text)
        self.assertIn('Log(out_add_0) -> y', text)

    @ignore_warnings(RuntimeWarning)
    def test_simple_case2(self):
        OnnxAdd, OnnxLog, OnnxAbs = loadop('Add', 'Log', 'Abs')
        opv = 5
        add = OnnxAdd('x', numpy.array([1], dtype=numpy.float32), op_version=opv)
        aaa = OnnxAbs(add, op_version=opv)
        logx = OnnxLog(aaa, op_version=opv, output_names=['y'])
        onx = logx.to_onnx(numpy.float32, numpy.float32)
        check_onnx(onx)

        x = numpy.random.randn(3, 4).astype(numpy.float32)
        oinf = OnnxInference(onx)
        y = oinf.run({'x': x})['y']
        self.assertEqual(numpy.log(numpy.abs(x + 1)), y)

        # compression
        onx2 = compress_proto(onx)
        self.assertEqual(len(onx2.graph.node), 1)
        check_onnx(onx2)
        oinf2 = OnnxInference(onx2)
        y = oinf.run({'x': x})['y']
        self.assertEqual(numpy.log(numpy.abs(x + 1)), y)

        # text
        text = onnx_simple_text_plot(onx2, recursive=True)
        self.assertIn('expression=G1', text)
        self.assertIn('Log(out_abs_0) -> y', text)


    @ignore_warnings(RuntimeWarning)
    def test_simple_case3(self):
        OnnxAdd, OnnxLog, OnnxAbs, OnnxExp = loadop('Add', 'Log', 'Abs', 'Exp')
        opv = 5
        add = OnnxAdd('x', numpy.array([1], dtype=numpy.float32), op_version=opv)
        eee = OnnxExp(add, op_version=opv)
        logx = OnnxLog(OnnxAbs(eee, op_version=opv),
                       op_version=opv, output_names=['y'])
        onx = logx.to_onnx(numpy.float32, numpy.float32)
        check_onnx(onx)

        x = numpy.random.randn(3, 4).astype(numpy.float32)
        expected = numpy.log(numpy.abs(numpy.exp(x + 1)))

        oinf = OnnxInference(onx)
        y = oinf.run({'x': x})['y']
        self.assertEqual(expected, y)

        # compression
        onx2 = compress_proto(onx)
        self.assertEqual(len(onx2.graph.node), 1)
        check_onnx(onx2)
        oinf2 = OnnxInference(onx2)
        y = oinf.run({'x': x})['y']
        self.assertEqual(expected, y)

        # text
        text = onnx_simple_text_plot(onx2, recursive=True)
        self.assertIn('expression=G1', text)
        self.assertIn('Log(out_abs_0) -> y', text)


    @ignore_warnings(RuntimeWarning)
    def test_simple_case4(self):
        OnnxAdd, OnnxLog, OnnxAbs, OnnxExp, OnnxSub = loadop(
            'Add', 'Log', 'Abs', 'Exp', 'Sub')
        opv = 5
        add = OnnxAdd('x', numpy.array([1], dtype=numpy.float32), op_version=opv)
        eee = OnnxExp(add, op_version=opv)
        bbb = OnnxSub(eee, 'c', op_version=opv)
        logx = OnnxLog(OnnxAbs(bbb, op_version=opv),
                       op_version=opv, output_names=['y'])
        onx = logx.to_onnx(numpy.float32, numpy.float32)
        check_onnx(onx)

        x = numpy.random.randn(3, 4).astype(numpy.float32)
        expected = numpy.log(numpy.abs(numpy.exp(x + 1) - x))

        oinf = OnnxInference(onx)
        y = oinf.run({'x': x, 'c': x})['y']
        self.assertEqual(expected, y)

        # compression
        onx2 = compress_proto(onx)
        self.assertEqual(len(onx2.graph.node), 1)
        check_onnx(onx2)
        oinf2 = OnnxInference(onx2)
        y = oinf.run({'x': x, 'c': x})['y']
        self.assertEqual(expected, y)

        # text
        text = onnx_simple_text_plot(onx2, recursive=True)
        self.assertIn('expression=G1', text)
        self.assertIn('Log(out_abs_0) -> y', text)


if __name__ == "__main__":
    # TestCompressOnnx().test_simple_case2()
    unittest.main(verbosity=2)
