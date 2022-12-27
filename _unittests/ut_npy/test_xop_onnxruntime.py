# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node,
    make_graph, make_tensor_value_info)
from onnx.shape_inference import infer_shapes
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
from onnxruntime import __version__ as ortver
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_variable import max_supported_opset


class TestXOps(ExtTestCase):

    def test_syntax_onnx(self):
        from onnxruntime import InferenceSession
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', 0, None)
        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])
        graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
        onnx_model = make_model(graph)
        del onnx_model.opset_import[:]
        opset = onnx_model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        new_onnx = infer_shapes(onnx_model)
        sess = InferenceSession(new_onnx.SerializeToString())
        x = numpy.array([[1]], dtype=numpy.float32)
        y = sess.run(None, {'X': x, 'A': x, 'B': x})
        self.assertEqualArray(y, numpy.array([[[2]]], dtype=numpy.float32))

    def test_topk_classic(self):
        opv = max_supported_opset()
        OnnxIdentity, OnnxTopK = loadop("Identity", "TopK")
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        # axis=1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64), axis=1,
                       op_version=opv)
        id1 = OnnxIdentity(onx[0], output_names=['Y'], op_version=opv)
        id2 = OnnxIdentity(onx[1], output_names=['Yi'], op_version=opv)
        model_def = id1.to_onnx(numpy.float32, other_outputs=[id2],
                                target_opset=opv)
        for rt in ['onnxruntime1', 'python']:
            with self.subTest(rt=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
                exp = numpy.array(
                    [[4., 3.], [5., 4.], [5., 2.]], dtype=numpy.float32)
                self.assertEqualArray(exp, got['Y'])
                exp = numpy.array([[4, 3], [4, 3], [3, 0]], dtype=numpy.int64)
                self.assertEqualArray(exp, got['Yi'])

    def test_topk_iter(self):
        opv = max_supported_opset()
        OnnxIdentity, OnnxTopK = loadop("Identity", "TopK")
        X = numpy.array([[0, 1, 2, 3, 4],
                         [1, -1, -2, 4, 5],
                         [2, -2, -3, 5, -4]],
                        dtype=numpy.float32)

        # axis=1, k=2
        onx = OnnxTopK('X', numpy.array([2], dtype=numpy.int64), axis=1,
                       op_version=opv)
        vals, inds = onx
        id1 = OnnxIdentity(vals, output_names=['Y'], op_version=opv)
        id2 = OnnxIdentity(inds, output_names=['Yi'], op_version=opv)
        model_def = id1.to_onnx(numpy.float32, other_outputs=[id2],
                                target_opset=opv)
        for rt in ['onnxruntime1', 'python']:
            with self.subTest(rt=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                got = oinf.run({'X': X})
                self.assertEqual(list(sorted(got)), ['Y', 'Yi'])
                exp = numpy.array(
                    [[4., 3.], [5., 4.], [5., 2.]], dtype=numpy.float32)
                self.assertEqualArray(exp, got['Y'])
                exp = numpy.array([[4, 3], [4, 3], [3, 0]], dtype=numpy.int64)
                self.assertEqualArray(exp, got['Yi'])

    def test_onnx_add_op_onnxruntime(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovf = ov + ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)

        opv = max_supported_opset()
        ov = OnnxAbs('X', op_version=opv)
        ovf = ov + ov
        last = OnnxIdentity(ovf, output_names=['Y'], op_version=opv)
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0,
                           target_opset=opv)

        oinf = OnnxInference(onx, runtime='onnxruntime1')
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) * 2, got['Y'])

    def test_onnx_add_op_onnxruntime_specific(self):
        OnnxAbs_13, OnnxIdentity_14 = loadop("Abs_13", "Identity_14")

        opv = max_supported_opset()
        ov = OnnxAbs_13('X')
        ovf = ov + ov
        last = OnnxIdentity_14(ovf, output_names=['Y'], op_version=opv)
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0,
                           target_opset=opv)

        oinf = OnnxInference(onx, runtime='onnxruntime1')
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) * 2, got['Y'])

    @unittest.skipIf(compare_module_version(ortver, '1.13.1') <= 0,
                     reason="opset not supported by onnxruntime")
    def test_reduce_mean_verbose(self):
        from onnxruntime import InferenceSession
        from mlprodict.npy.xop_opset import OnnxReduceMeanApi18
        OnnxTopK, OnnxGatherElements = loadop('TopK', 'GatherElements')
        topk = OnnxTopK('X', numpy.array([2], dtype=numpy.int64), axis=1)
        dist = OnnxGatherElements('W', topk[1], axis=1)
        result = OnnxReduceMeanApi18(dist * topk[0], axes=[1])
        X = numpy.array([[4, 5, 6], [7, 0, 1]], dtype=numpy.float32)
        W = numpy.array([[1, 0.5, 0.6], [0.5, 0.2, 0.3]], dtype=numpy.float32)
        onx = result.to_onnx(numpy.float32, numpy.float32)
        sess = OnnxInference(onx)
        name = sess.output_names[0]
        result1 = sess.run({'X': X, 'W': W})[name]
        sess2 = InferenceSession(onx.SerializeToString())
        result2 = sess2.run(None, {'X': X, 'W': W})[0]
        self.assertEqualArray(result1, result2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
