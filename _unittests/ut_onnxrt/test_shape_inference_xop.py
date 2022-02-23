"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from onnx.shape_inference import infer_shapes
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxShapeInference
from mlprodict.onnxrt.ops_shape.shape_result import ShapeResult
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_variable import Variable


class TestOnnxShapeInferenceXop(ExtTestCase):

    opsets = list(range(10, TARGET_OPSET + 1))

    def check_infer_shapes(self, onx, out, rt):
        onnx_shapes = infer_shapes(onx)
        inferred = onnx_shapes.graph.value_info  # pylint: disable=
        for data in inferred:
            if data.name not in out:
                raise AssertionError("Name %r not found." % data.name)
            shape, dtype, sparse = OnnxShapeInference._get_shape(
                data)  # pylint: disable=W0212
            for i in range(len(shape)):
                if not isinstance(shape[i], str):
                    continue
                if shape[i].startswith('unk_'):
                    shape[i] = shape[i][4:]
            res = ShapeResult(data.name, shape, dtype, sparse)
            if res != out[data.name]:
                raise AssertionError(
                    "Unexpected differences for name %r:\nexp: %r\ngot: %r"
                    "\n-----\n%s" % (
                        data.name, res, out[data.name],
                        onnx_simple_text_plot(onx)))

    def test_onnx_shape_inference(self):
        OnnxAdd = loadop('Add')
        dtype = numpy.float32
        for opset in TestOnnxShapeInferenceXop.opsets:
            with self.subTest(opset=opset):
                cop = OnnxAdd('X', numpy.array(
                    [[1]], dtype=dtype), op_version=opset)
                cop4 = OnnxAdd(cop, numpy.array([[2]], dtype=dtype),
                               output_names=['Y'])
                vari = Variable('X', numpy.float32, [None, None])
                model_def = cop4.to_onnx([vari], run_shape=False)
                rt = OnnxShapeInference(model_def)
                out = rt.run()
                self.assertIn('X', out)
                self.assertIn('Y', out)
                y = out['Y']
                self.assertEqual(numpy.float32, y.dtype)
                self.assertEqual(['_0', '_1'], y.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
