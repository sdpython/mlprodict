# pylint: disable=E0611
"""
@brief      test log(time=10s)
"""
import unittest
import numpy
from scipy.spatial.distance import squareform, pdist
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node,
    make_graph, make_tensor_value_info)
from onnx.shape_inference import infer_shapes
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_variable import Variable, max_supported_opset
from mlprodict.npy.xop import _GraphBuilder
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnx_tools.onnx2py_helper import get_dtype_shape


class TestXOps(ExtTestCase):

    def test_float32(self):
        self.assertEqual(numpy.float32, numpy.dtype('float32'))

    def test_impossible(self):
        cl = loadop("OnnxAdd")
        self.assertEqual(cl.__name__, "OnnxAdd")
        cl = loadop("OnnxCast")
        self.assertEqual(cl.__name__, "OnnxCast")
        cl = loadop("Cast_13")
        self.assertEqual(cl.__name__, "OnnxCast_13")
        cl = loadop("OnnxCast_13")
        self.assertEqual(cl.__name__, "OnnxCast_13")
        self.assertRaise(lambda: loadop("OnnxImpossible"), ValueError)
        self.assertRaise(lambda: loadop("OnnxImpossible_1"), ValueError)
        self.assertRaise(lambda: loadop("OnnxCast_9999"), ValueError)

    def test_onnx_abs(self):
        OnnxAbs = loadop("OnnxAbs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add(self):
        OnnxAdd = loadop("Add")
        ov = OnnxAdd('X', 'X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + x, got['Y'])

    def test_onnx_add_cst(self):
        OnnxAdd = loadop("OnnxAdd")
        ov = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + 1, got['Y'])

    def test_number2alpha(self):
        sel = [_GraphBuilder.number2alpha(i) for i in range(0, 100001)]
        sel2 = sel.copy()
        sel2.sort()
        self.assertEqual(sel, sel2)

    def test_onnx_add_sub_left(self):
        OnnxAdd, OnnxSub = loadop("OnnxAdd", "OnnxSub")
        self.assertEqual(OnnxAdd.operator_name, 'Add')
        self.assertEqual(OnnxSub.operator_name, 'Sub')
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub(ov, 'X', output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x, got['Y'])

    def test_onnx_add_sub_right(self):
        OnnxAdd, OnnxSub = loadop("OnnxAdd", "OnnxSub")
        self.assertEqual(OnnxAdd.operator_name, 'Add')
        self.assertEqual(OnnxSub.operator_name, 'Sub')
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub('X', ov, output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(-x, got['Y'])

    def test_onnx_transpose(self):
        OnnxTranspose = loadop("OnnxTranspose")
        ov = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertIn('perm', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.T, got['Y'])

    def test_onnx_transpose3(self):
        OnnxTranspose = loadop("OnnxTranspose")
        ov = OnnxTranspose('X', perm=[1, 0, 2], output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertIn('perm', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[[-2, 2]]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.transpose(x, axes=(1, 0, 2)), got['Y'])

    def test_onnx_cast(self):
        OnnxCast = loadop("OnnxCast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.int64, verbose=0)
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_dict(self):
        OnnxCast = loadop("OnnxCast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx({'X': numpy.float32}, {'Y': numpy.int64}, verbose=0)
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_var(self):
        OnnxCast = loadop("OnnxCast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx(Variable('X', numpy.float32),
                         Variable('Y', numpy.float32), verbose=0)
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_var_list(self):
        OnnxCast = loadop("OnnxCast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx([Variable('X', numpy.float32)],
                         [Variable('Y', numpy.float32)], verbose=0)
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_if(self):
        OnnxConstant, OnnxIf, OnnxGreater = loadop(
            "OnnxConstant", "OnnxIf", "OnnxGreater")
        bthen = OnnxConstant(
            value_floats=numpy.array([0], dtype=numpy.float32),
            output_names=['res_then'])
        bthen.set_onnx_name_prefix('then')

        belse = OnnxConstant(
            value_floats=numpy.array([1], dtype=numpy.float32),
            output_names=['res_else'])
        belse.set_onnx_name_prefix('else')

        bthen_body = bthen.to_onnx(
            [], [Variable('res_then', numpy.float32)])
        belse_body = belse.to_onnx(
            [], [Variable('res_else', numpy.float32)])

        onx = OnnxIf(
            OnnxGreater('X', numpy.array([0], dtype=numpy.float32)),
            output_names=['Z'],
            then_branch=bthen_body.graph,
            else_branch=belse_body.graph)

        x = numpy.array([1, 2], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': numpy.float32}, {'Z': numpy.float32})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(numpy.array([0.], dtype=numpy.float32),
                              got['Z'])

        x = numpy.array([-1, -2], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': numpy.float32}, {'Z': numpy.float32})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(
            numpy.array([1.], dtype=numpy.float32), got['Z'])

    def test_if2(self):
        OnnxAdd, OnnxSub, OnnxIf, OnnxGreater, OnnxReduceSum = loadop(
            "OnnxAdd", "OnnxSub", "OnnxIf", "OnnxGreater", "OnnxReduceSum")

        node = OnnxAdd('x1', 'x2', output_names=['absxythen'])
        then_body = node.to_onnx(
            [Variable('x1', numpy.float32),
             Variable('x2', numpy.float32)],
            {'absxythen': numpy.float32})
        node = OnnxSub('x1', 'x2', output_names=['absxyelse'])
        else_body = node.to_onnx(
            [Variable('x1', numpy.float32),
             Variable('x2', numpy.float32)],
            {'absxyelse': numpy.float32})
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(OnnxReduceSum('x1'), OnnxReduceSum('x2'))
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        output_names=['y'])
        model_def = ifnode.to_onnx(
            [Variable('x1', numpy.float32),
             Variable('x2', numpy.float32)],
            {'y': numpy.float32})
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn("out_red0 -> _greater;", dot)

    def test_onnx_abs_shape_variable(self):
        OnnxAbs = loadop("OnnxAbs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx([Variable('X', numpy.float32, [1, 2])],
                         [Variable('Y', numpy.float32, [1, 2])],
                         verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])
        self.assertIn("input: name='X'", onnx_simple_text_plot(onx))
        dtype, shape = get_dtype_shape(onx.graph.input[0])
        self.assertEqual(dtype, TensorProto.FLOAT)
        self.assertEqual(shape, (1, 2))
        dtype, shape = get_dtype_shape(onx.graph.output[0])
        self.assertEqual(dtype, TensorProto.FLOAT)
        self.assertEqual(shape, (1, 2))

    def test_onnx_abs_shape_variable_batch(self):
        OnnxAbs = loadop("OnnxAbs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx([Variable('X', numpy.float32, [None, 2])],
                         [Variable('Y', numpy.float32, [None, 2])],
                         verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])
        dtype, shape = get_dtype_shape(onx.graph.input[0])
        self.assertEqual(dtype, TensorProto.FLOAT)
        self.assertEqual(shape, (None, 2))
        dtype, shape = get_dtype_shape(onx.graph.output[0])
        self.assertEqual(dtype, TensorProto.FLOAT)
        self.assertEqual(shape, (None, 2))

    def test_onnx_abs_shape_numpy(self):
        OnnxAbs = loadop("OnnxAbs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([-2, 2], dtype=numpy.float32)
        onx = ov.to_onnx({'X': x}, {'Y': x}, verbose=0)
        oinf = OnnxInference(onx)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])
        dtype, shape = get_dtype_shape(onx.graph.input[0])
        self.assertEqual(dtype, TensorProto.FLOAT)
        self.assertEqual(shape, (2, ))
        dtype, shape = get_dtype_shape(onx.graph.output[0])
        self.assertEqual(dtype, TensorProto.FLOAT)
        self.assertEqual(shape, (2, ))

    def test_scan_pdist(self):
        (OnnxSub, OnnxIdentity, OnnxReduceSumSquare, OnnxScan,
         OnnxAdd) = loadop('Sub', 'Identity',
                           'ReduceSumSquare', 'Scan', 'Add')

        def onnx_squareform_pdist(X, dtype=None, op_version=None, **kwargs):
            diff = OnnxSub('next_in', 'next',
                           op_version=op_version)
            id_next = OnnxIdentity('next_in', output_names=['next_out'],
                                   op_version=op_version)
            flat = OnnxReduceSumSquare(diff, axes=[1], op_version=op_version,
                                       output_names=['scan_out'], keepdims=0)
            scan_body = id_next.to_onnx(
                [Variable('next_in', numpy.float32, (None, None)),  # tensor_type([None, None])),
                 Variable('next', numpy.float32, (None, ))],  # tensor_type([None]))]),
                outputs=[Variable('next_out', numpy.float32, (None, None)),  # ([None, None])),
                         Variable('scan_out', numpy.float32, (None, ))],  # tensor_type([None]))],
                other_outputs=[flat],
                target_opset=op_version)
            output_names = [o.name for o in scan_body.graph.output]
            self.assertEqual(['next_out', 'scan_out'], output_names)
            dtype, shape = get_dtype_shape(scan_body.graph.output[0])
            self.assertEqual(dtype, TensorProto.FLOAT)
            self.assertEqual(shape, (None, None))
            dtype, shape = get_dtype_shape(scan_body.graph.output[1])
            self.assertEqual(dtype, TensorProto.FLOAT)
            self.assertEqual(shape, (None, ))

            node = OnnxScan(X, X, output_names=['S1', 'S2'],
                            num_scan_inputs=1,
                            body=(scan_body.graph, [id_next, flat]),
                            op_version=op_version, **kwargs)
            return node[1]

        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('input', 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': numpy.float32},
            outputs=[Variable('cdist', numpy.float32)])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})
        self.assertEqual(list(res.keys()), ['cdist'])
        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        self.assertEqualArray(exp, res['cdist'])

        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((2, 3))
        res = sess.run({'input': x})
        self.assertEqual(list(res.keys()), ['cdist'])
        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        self.assertEqualArray(exp, res['cdist'])

    def test_syntax_python(self):

        class AA:
            def __init__(self):
                pass

            def __iter__(self):
                yield 3
                yield 4

        a, b = AA()
        self.assertEqual(a, 3)
        self.assertEqual(b, 4)

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

    def test_onnx_abs_undefined(self):
        OnnxAbs = loadop("OnnxAbs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])
        oinf = OnnxInference(onx, runtime='onnxruntime1')
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add_sub_left_undefined(self):
        OnnxAdd, OnnxSub = loadop("OnnxAdd", "OnnxSub")
        self.assertEqual(OnnxAdd.operator_name, 'Add')
        self.assertEqual(OnnxSub.operator_name, 'Sub')
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub(ov, 'X', output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(x, got['Y'])
        oinf = OnnxInference(onx, runtime='onnxruntime1')
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(x, got['Y'])

    def test_topk_classic(self):
        opv = max_supported_opset()
        OnnxIdentity, OnnxTopK = loadop("OnnxIdentity", "OnnxTopK")
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
        OnnxIdentity, OnnxTopK = loadop("OnnxIdentity", "OnnxTopK")
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

    def test_onnx_add_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovf = ov + ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) * 2, got['Y'])

    def test_onnx_sub_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovf = ov + ov - ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_mul_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovf = ov * ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) ** 2, got['Y'])

    def test_onnx_div_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovf = ov / (ov + ov)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a / (a + a), got['Y'])

    def test_onnx_pow_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovf = ov ** ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a ** a, got['Y'])

    def test_onnx_matmul_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovf = ov @ ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [-3, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a @ a, got['Y'])

    def test_onnx_greater_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = ov > ovi
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a > x, got['Y'])

    def test_onnx_less_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = ov < ovi
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a < x, got['Y'])

    def test_onnx_equal_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = ov == ovi
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a == x, got['Y'])

    def test_onnx_and_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = (ov == ovi).and_(ov > ovi)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a == -10, got['Y'])

    def test_onnx_or_op(self):
        OnnxAbs, OnnxIdentity = loadop("OnnxAbs", "OnnxIdentity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = (ov == ovi).or_(ov > ovi)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a >= x, got['Y'])

    def test_onnx_abs_op(self):
        OnnxIdentity = loadop("OnnxIdentity")
        ovi = OnnxIdentity('X')
        ovf = abs(ovi)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a, got['Y'])

    def test_onnx_not_op(self):
        OnnxIdentity = loadop("OnnxIdentity")
        ovi = OnnxIdentity('X')
        ovf = (abs(ovi) == ovi).not_()
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a != x, got['Y'])

    def test_onnx_mod_op(self):
        OnnxIdentity = loadop("OnnxIdentity")
        ovi = OnnxIdentity('X')
        ovf = ovi % numpy.array([10], dtype=numpy.int64)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.int64, numpy.int64, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x % 10, got['Y'])


if __name__ == "__main__":
    # TestXOps().test_topk_iter()
    unittest.main(verbosity=2)