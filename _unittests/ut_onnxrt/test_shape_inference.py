"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from onnx.shape_inference import infer_shapes
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxSub, OnnxDiv, OnnxMul)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt import OnnxShapeInference
from mlprodict.onnxrt.ops_shape.shape_result import (
    ShapeResult, ShapeConstraint, ShapeConstraintList)
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.onnxrt.ops_shape.shape_result import ShapeInferenceException


class TestOnnxShapeInference(ExtTestCase):

    opsets = list(range(10, TARGET_OPSET + 1))

    def check_infer_shapes(self, onx, out, rt):
        onnx_shapes = infer_shapes(onx)
        inferred = onnx_shapes.graph.value_info  # pylint: disable=
        for data in inferred:
            if data.name not in out:
                raise AssertionError(f"Name {data.name!r} not found.")
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

    def test_shape_constraint(self):
        sh1 = ShapeConstraint('_1', {1, 2})
        sh2 = ShapeConstraint('_1', {1, 2})
        self.assertEqual(sh1, sh2)
        shl = ShapeConstraintList()
        shl.append(sh1)
        self.assertIn(sh1, shl)
        self.assertIn(sh2, shl)

    def test_onnx_shape_inference(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        for opset in TestOnnxShapeInference.opsets:
            with self.subTest(opset=opset):
                cop = OnnxAdd('X', numpy.array(
                    [[1]], dtype=dtype), op_version=opset)
                cop4 = OnnxAdd(cop, numpy.array([[2]], dtype=dtype), op_version=opset,
                               output_names=['Y'])
                model_def = cop4.to_onnx({'X': x}, target_opset=opset)
                rt = OnnxShapeInference(model_def)
                self.assertIn('OnnxShapeInference(', repr(rt))
                out = rt.run({'X': x})
                self.assertIn('X', out)
                self.assertIn('Y', out)
                self.assertIn('Ad_Addcst', out)
                self.assertEqual(len(out), 5)
                self.assertIn(
                    "'Ad_C0': ShapeResult('Ad_C0', ['_0', 2], dtype('float32')",
                    str(out))
                self.check_infer_shapes(model_def, rt.run(), rt)
                cons = rt.known_shapes_.get_all_constraints()
                self.assertEqual(len(cons), 1)
                self.assertEqual(list(cons), ['_1'])
                self.assertEqual(len(cons['_1']), 1)
                cst = cons['_1'][0]
                self.assertEqual(cst.name, '_1')
                self.assertEqual(cst.values, {'_0'})
                self.assertEqual(
                    rt.known_shapes_.names,
                    {'_0': ('', 'X', 0), '_1': ('', 'Y', 0)})

    def test_onnx_shape_inference_lr(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        for opset in TestOnnxShapeInference.opsets:
            with self.subTest(opset=opset):
                cop = OnnxAdd('X', numpy.array(
                    [[1, 1]], dtype=dtype), op_version=opset)
                cop4 = OnnxAdd(cop, numpy.array([[2]], dtype=dtype), op_version=opset,
                               output_names=['Y'])
                model_def = cop4.to_onnx({'X': x}, target_opset=opset)
                rt = OnnxShapeInference(model_def)
                self.assertIn('OnnxShapeInference(', repr(rt))
                out = rt.run({'X': x})
                self.assertIn('X', out)
                self.assertIn('Y', out)
                self.assertIn('Ad_Addcst', out)
                self.assertEqual(len(out), 5)
                self.assertIn(
                    "'Ad_C0': ShapeResult('Ad_C0', ['_0', 2], dtype('float32')",
                    str(out))
                self.check_infer_shapes(model_def, rt.run(), rt)
                cons = rt.known_shapes_.get_all_constraints()
                self.assertEqual(len(cons), 1)
                self.assertEqual(list(cons), ['_1'])
                self.assertEqual(len(cons['_1']), 1)
                cst = cons['_1'][0]
                self.assertEqual(cst.name, '_1')
                self.assertEqual(cst.values, {'_0'})
                self.assertEqual(
                    rt.known_shapes_.names,
                    {'_0': ('', 'X', 0), '_1': ('', 'Y', 0)})

    def test_onnx_shape_inference_missing(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        for clop in [OnnxAdd, OnnxSub, OnnxDiv, OnnxMul]:
            for opset in TestOnnxShapeInference.opsets[-1:]:
                with self.subTest(opset=opset, clop=clop):
                    cop = OnnxAdd('X', numpy.array(
                        [[1]], dtype=dtype), op_version=opset)
                    cop4 = clop(cop, numpy.array([[2, 4]], dtype=dtype),
                                op_version=opset,
                                output_names=['Y'])
                    model_def = cop4.to_onnx(
                        {'X': FloatTensorType([None, None])},
                        {'Y': FloatTensorType([None, None])},
                        target_opset=opset)
                    rt = OnnxShapeInference(model_def)
                    out = rt.run({'X': x})
                    self.assertIn('X', out)
                    self.assertIn('Y', out)
                    self.assertIn('Ad_Addcst', out)
                    self.assertEqual(len(out), 5)
                    self.assertIn(
                        "'Ad_C0': ShapeResult('Ad_C0', ['_0', '_1'], dtype('float32'))",
                        str(out))
                    out = rt.run()
                    self.assertIn(
                        "'Y': ShapeResult('Y', ['_2', '_3']", str(out))
                    self.check_infer_shapes(model_def, rt.run(), rt)
                    cons = rt.known_shapes_.get_all_constraints()
                    self.assertEqual(len(rt.known_shapes_.names), 4)
                    self.assertEqual(set(rt.known_shapes_.names),
                                     {'_0', '_1', '_2', '_3'})
                    self.assertEqual(len(cons), 3)
                    self.assertEqual(list(cons), ['_1', '_2', '_3'])
                    self.assertEqual(len(cons['_1']), 1)
                    cst = cons['_1'][0]
                    self.assertEqual(cst.name, '_1')
                    self.assertEqual(cst.values, {1, 2})
                    self.assertEqual(
                        rt.known_shapes_.names,
                        {'_0': ('', 'X', 0), '_1': ('', 'X', 1),
                         '_2': ('', 'Y', 0), '_3': ('', 'Y', 1)})
                    get = out.get()
                    self.assertEqual(get['Ad_C0'].shape, ['d0', {1, 2}])
                    self.assertEqual(get['Y'].shape, ['d0', 2])
                    self.assertEqual(get['X'].shape, ['d0', {1, 2}])
                    self.assertEqual(len(get['Ad_C0'].shape), 2)
                    self.assertIsInstance(get['Ad_C0'].shape[0], str)

    def test_onnx_shape_inference_exc(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))

        # case 1
        opset = TestOnnxShapeInference.opsets[-1]
        cop = OnnxAdd(
            'X', numpy.array([[10, 10, 10]], dtype=dtype), op_version=opset,
            output_names=['Y'])
        model_def = cop.to_onnx(
            {'X': x}, {'Y': FloatTensorType([None, None])},
            target_opset=opset)
        self.assertRaise(lambda: OnnxShapeInference(model_def),
                         ShapeInferenceException)

        # case 2
        opset = TestOnnxShapeInference.opsets[-1]
        cop = OnnxAdd(
            'X', numpy.array([[10, 10, 10, 10]], dtype=dtype).reshape((2, 2)),
            op_version=opset, output_names=['Y'])
        model_def = cop.to_onnx(
            {'X': x}, {'Y': FloatTensorType([None, 3])},
            target_opset=opset)
        self.assertRaise(lambda: OnnxShapeInference(model_def),
                         RuntimeError)

        # case 3
        opset = TestOnnxShapeInference.opsets[-1]
        cop = OnnxAdd(
            'X', numpy.array([[10, 10, 10, 10]], dtype=dtype).T,
            op_version=opset, output_names=['Y'])
        model_def = cop.to_onnx(
            {'X': x}, {'Y': FloatTensorType([None, 3])},
            target_opset=opset)
        self.assertRaise(lambda: OnnxShapeInference(model_def), RuntimeError)
        # out = rt.run()
        # print(out)

    def test_onnx_onnx_shape_inference(self):
        def get_dim(d):
            if hasattr(d, 'dim_param') and d.dim_param not in (None, ''):
                return d.dim_param
            return d.dim_value

        X = make_tensor_value_info('X', TensorProto.FLOAT, ['N', None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [1])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [1, None])
        B = numpy_helper.from_array(
            numpy.random.randn(1, 3).astype(numpy.float32),
            name='B')
        node1 = make_node('Sub', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['T'])
        node3 = make_node('Identity', ['T'], ['Y'])
        graph = make_graph([node1, node2, node3], 'lr', [X, A], [Y], [B])
        onnx_model = make_model(graph)

        onnx_shapes = infer_shapes(onnx_model)
        shapes = {}
        for v in onnx_shapes.graph.value_info:
            dims = [get_dim(d) for d in v.type.tensor_type.shape.dim]
            shapes[v.name] = dims
        self.assertEqual(shapes, {'XA': ['N', 'unk__0'], 'T': ['N', 3]})
        oinf = OnnxShapeInference(onnx_model)
        got = oinf.run()
        got.resolve()
        res = got.get()
        xa = res["XA"].shape
        t = res["T"].shape
        self.assertEqual(xa[1], {1, 3})
        self.assertEqual(xa[0], shapes["XA"][0])
        self.assertEqual(t, shapes["T"])
    

if __name__ == "__main__":
    unittest.main(verbosity=2)
