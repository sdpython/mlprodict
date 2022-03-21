"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from scipy.spatial.distance import squareform, pdist
from onnx import TensorProto, ValueInfoProto
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnx_tools.onnx2py_helper import get_dtype_shape
from mlprodict.npy.xop import (
    loadop, OnnxLoadFactory, _GraphBuilder, _domain_to_class_name)
from mlprodict.npy.xop_auto import get_domain_list
from mlprodict.npy.xop_variable import (
    Variable, max_supported_opset,
    numpy_type_prototype, is_numpy_dtype,
    InputDetectedVariable, OutputDetectedVariable)
from mlprodict.npy.xop_opset import (
    OnnxReduceSumApi11, OnnxSplitApi11, OnnxSqueezeApi11,
    OnnxUnsqueezeApi11, OnnxReduceL2_typed, OnnxReshapeApi13)


class TestXOps(ExtTestCase):

    def test_private(self):
        v = _domain_to_class_name('ai.onnx')
        self.assertEqual(v, '')
        v = _domain_to_class_name('o')
        self.assertEqual(v, 'O')

    def test_private2(self):
        v = OnnxLoadFactory()
        self.assertIsInstance(v._loaded_classes, dict)

    def test_square_error_no_output_names(self):
        OnnxSub, OnnxMul = loadop('Sub', 'Mul')
        diff = OnnxSub('X', 'Y')
        error = OnnxMul(diff, diff)
        onx = error.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn("elem_type: 0", str(onx))
        X = numpy.array([4, 5], dtype=numpy.float32)
        Y = numpy.array([4.3, 5.7], dtype=numpy.float32)
        sess = OnnxInference(onx)
        name = sess.output_names[0]
        result = sess.run({'X': X, 'Y': Y})
        self.assertEqualArray((X - Y) ** 2, result[name])

    def test_float32(self):
        self.assertEqual(numpy.float32, numpy.dtype('float32'))

    def test_numpy_dtype(self):
        self.assertEqual(is_numpy_dtype(numpy.float32), True)
        self.assertEqual(is_numpy_dtype(numpy.dtype('float32')), True)
        self.assertEqual(is_numpy_dtype({}), False)

    def test_numpy_type_prototype(self):
        self.assertEqual(
            numpy_type_prototype(numpy.float32), TensorProto.FLOAT)
        self.assertEqual(
            numpy_type_prototype(numpy.dtype('float32')), TensorProto.FLOAT)
        self.assertRaise(lambda: numpy_type_prototype(5), TypeError)

    def test_get_domain_list(self):
        self.assertEqual(['', 'ai.onnx.ml', 'ai.onnx.preview.training'],
                         get_domain_list())

    def test_variable(self):
        var = Variable('X', numpy.float32)
        self.assertEqual(var.is_named('X'), True)
        self.assertEqual(var.name, 'X')
        self.assertEqual(var.dtype, numpy.float32)
        self.assertEqual(var.proto_type, TensorProto.FLOAT)
        self.assertRaise(lambda: Variable('X', 5), TypeError)
        self.assertRaise(lambda: var.is_named(4), TypeError)
        self.assertRaise(
            lambda: Variable('X', numpy.float32, added_dtype=5),
            TypeError)
        self.assertRaise(lambda: Variable('X', shape='t'), TypeError)
        self.assertRaise(lambda: Variable('X', added_shape='t'), TypeError)
        var = Variable('X', numpy.float32)
        r = repr(var)
        self.assertEqual(r, "Variable('X', dtype=<class 'numpy.float32'>)")
        var = Variable('X', added_dtype=numpy.float32)
        r = repr(var)
        self.assertEqual(
            r, "Variable('X', added_dtype=<class 'numpy.float32'>)")
        self.assertRaise(lambda: var == 'T', TypeError)
        var2 = var
        self.assertEqual(var == var2, True)
        self.assertEqual(var == Variable('Y'), False)
        self.assertEqual(var == Variable('X', numpy.float32), False)
        self.assertEqual(
            var == Variable('X', added_dtype=numpy.float32), True)

    def test_variable_from_pb(self):
        var = Variable('X', numpy.float32)
        info = var.make_value_info()
        self.assertIsInstance(info, ValueInfoProto)
        var2 = Variable.from_pb(info)
        self.assertEqual(var2.name, 'X')
        self.assertEqual(var2.dtype, numpy.float32)

    def test_detected_variable(self):
        var = Variable('X', numpy.float32)
        ivar = InputDetectedVariable(None, var)
        sivar = repr(ivar)
        self.assertIn("InputDetectedVariable(None, Variable('X',", sivar)
        ovar = OutputDetectedVariable(None, var, 0)
        sovar = repr(ovar)
        self.assertIn("OutputDetectedVariable(None, Variable('X',", sovar)

    def test_impossible(self):
        cl = loadop("Add")
        self.assertEqual(cl.__name__, "OnnxAdd")
        cl = loadop("Cast")
        self.assertEqual(cl.__name__, "OnnxCast")
        cl = loadop("Cast_13")
        self.assertEqual(cl.__name__, "OnnxCast_13")
        cl = loadop("Cast_13")
        self.assertEqual(cl.__name__, "OnnxCast_13")
        self.assertRaise(lambda: loadop("OnnxCast"), ValueError)
        self.assertRaise(lambda: loadop("Impossible"), ValueError)
        self.assertRaise(lambda: loadop("Impossible_1"), ValueError)
        self.assertRaise(lambda: loadop("Cast_9999"), ValueError)

    def test_onnx_abs(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_abs_z(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Z'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Z'])

    def test_onnx_abs_wz(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('W', output_names=['Z'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'W': x})
        self.assertEqualArray(numpy.abs(x), got['Z'])

    def test_onnx_abs_domain(self):
        OnnxAbs = loadop(("", "Abs"))
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_abs_domain_ai(self):
        OnnxAbs = loadop(("ai.onnx", "Abs"))
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add(self):
        OnnxAdd = loadop("Add")
        ov = OnnxAdd('X', 'X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + x, got['Y'])

    def test_onnx_add_cst(self):
        OnnxAdd = loadop("Add")
        ov = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
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
        OnnxAdd, OnnxSub = loadop("Add", "Sub")
        self.assertEqual(OnnxAdd.operator_name, 'Add')
        self.assertEqual(OnnxSub.operator_name, 'Sub')
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub(ov, 'X', output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x, got['Y'])

    def test_onnx_add_sub_right(self):
        OnnxAdd, OnnxSub = loadop("Add", "Sub")
        self.assertEqual(OnnxAdd.operator_name, 'Add')
        self.assertEqual(OnnxSub.operator_name, 'Sub')
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub('X', ov, output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(-x, got['Y'])

    def test_onnx_transpose(self):
        OnnxTranspose = loadop("Transpose")
        ov = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        self.assertIn('perm', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.T, got['Y'])

    def test_onnx_transpose3(self):
        OnnxTranspose = loadop("Transpose")
        ov = OnnxTranspose('X', perm=[1, 0, 2], output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        self.assertIn('perm', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[[-2, 2]]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.transpose(x, axes=(1, 0, 2)), got['Y'])

    def test_onnx_cast(self):
        OnnxCast = loadop("Cast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.int64, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_dict(self):
        OnnxCast = loadop("Cast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx({'X': numpy.float32}, {'Y': numpy.int64}, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_var(self):
        OnnxCast = loadop("Cast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx(Variable('X', numpy.float32),
                         Variable('Y', numpy.float32), verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_var_list(self):
        OnnxCast = loadop("Cast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx([Variable('X', numpy.float32)],
                         [Variable('Y', numpy.float32)], verbose=0)
        self.assertIn('to', str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2.1, 2.1]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_abs_shape_variable(self):
        OnnxAbs = loadop("Abs")
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
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx([Variable('X', numpy.float32, [None, 2])],
                         [Variable('Y', numpy.float32, [None, 2])],
                         verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
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
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([-2, 2], dtype=numpy.float32)
        onx = ov.to_onnx({'X': x}, {'Y': x}, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
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
        for rt in ['python', 'python_compiled']:
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
        text = str(vals)
        self.assertIn('[0]', text)
        text = repr(vals)
        self.assertNotEmpty(vals.get_output_result(0))
        self.assertIn('OnnxOperatorItem', text)
        id1 = OnnxIdentity(vals, output_names=['Y'], op_version=opv)
        id2 = OnnxIdentity(inds, output_names=['Yi'], op_version=opv)
        model_def = id1.to_onnx(numpy.float32, other_outputs=[id2],
                                target_opset=opv)
        for rt in ['python_compiled', 'python']:
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
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity", verbose=0)
        ov = OnnxAbs('X')
        ovf = ov + ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) * 2, got['Y'])

    def test_onnx_add_op_python_compiled(self):
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

        oinf = OnnxInference(onx, runtime='python_compiled')
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) * 2, got['Y'])

    def test_onnx_add_op_python_compiled_specific(self):
        OnnxAbs_13, OnnxIdentity_14 = loadop("Abs_13", "Identity_14")

        opv = max_supported_opset()
        ov = OnnxAbs_13('X')
        ovf = ov + ov
        last = OnnxIdentity_14(ovf, output_names=['Y'], op_version=opv)
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0,
                           target_opset=opv)

        oinf = OnnxInference(onx, runtime='python_compiled')
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) * 2, got['Y'])

    def test_onnx_sub_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovf = ov + ov - ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_mul_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovf = ov * ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x) ** 2, got['Y'])

    def test_onnx_div_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovf = ov / (ov + ov)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a / (a + a), got['Y'])

    def test_onnx_pow_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovf = ov ** ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a ** a, got['Y'])

    def test_onnx_matmul_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovf = ov @ ov
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [-3, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a @ a, got['Y'])

    def test_onnx_greater_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = ov > ovi
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a > x, got['Y'])

    def test_onnx_less_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = ov < ovi
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a < x, got['Y'])

    def test_onnx_equal_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = ov == ovi
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a == x, got['Y'])

    def test_onnx_and_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = (ov == ovi).and_(ov > ovi)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a == -10, got['Y'])

    def test_onnx_or_op(self):
        OnnxAbs, OnnxIdentity = loadop("Abs", "Identity")
        ov = OnnxAbs('X')
        ovi = OnnxIdentity('X')
        ovf = (ov == ovi).or_(ov > ovi)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a >= x, got['Y'])

    def test_onnx_abs_op(self):
        OnnxIdentity = loadop("Identity")
        ovi = OnnxIdentity('X')
        ovf = abs(ovi)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a, got['Y'])

    def test_onnx_not_op(self):
        OnnxIdentity = loadop("Identity")
        ovi = OnnxIdentity('X')
        ovf = (abs(ovi) == ovi).not_()
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(a != x, got['Y'])

    def test_onnx_mod_op(self):
        OnnxIdentity = loadop("Identity")
        ovi = OnnxIdentity('X')
        ovf = ovi % numpy.array([10], dtype=numpy.int64)
        last = OnnxIdentity(ovf, output_names=['Y'])
        onx = last.to_onnx(numpy.int64, numpy.int64, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x % 10, got['Y'])

    def test_onnx_ml_operator(self):
        OnnxNormalizer = loadop(('ai.onnx.ml', "Normalizer"))
        self.assertEqual(OnnxNormalizer.__name__,
                         'OnnxAiOnnxMlNormalizer')
        last = OnnxNormalizer('X', norm='L1', output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(x / a.sum(axis=1, keepdims=True), got['Y'])

    def test_onnx_ml_operator_shortcut(self):
        OnnxNormalizer = loadop("Normalizer")
        self.assertEqual(OnnxNormalizer.__name__,
                         'OnnxAiOnnxMlNormalizer')
        last = OnnxNormalizer('X', norm='L1', output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        a = numpy.abs(x)
        self.assertEqualArray(x / a.sum(axis=1, keepdims=True), got['Y'])

    def test_opset_reduce_sum(self):
        for opv in range(10, max_supported_opset() + 1):
            with self.subTest(opv=opv):
                node = OnnxReduceSumApi11(
                    'X', axes=numpy.array([1], dtype=numpy.int64),
                    op_version=opv, output_names=['Y'])
                onx = node.to_onnx(numpy.float32, numpy.float32,
                                   target_opset=opv)
                self.assertNotIn("elem_type: 0", str(onx))
                oinf = OnnxInference(onx)
                x = numpy.array([[4, 5], [5.5, -6]], dtype=numpy.float32)
                got = oinf.run({'X': x})
                self.assertEqualArray(x.sum(axis=1, keepdims=1), got['Y'])

    def test_opset_reduce_sum_no_axis(self):
        for opv in range(10, max_supported_opset() + 1):
            with self.subTest(opv=opv):
                node = OnnxReduceSumApi11(
                    'X', op_version=opv, output_names=['Y'])
                onx = node.to_onnx(numpy.float32, numpy.float32,
                                   target_opset=opv)
                self.assertNotIn("elem_type: 0", str(onx))
                oinf = OnnxInference(onx)
                x = numpy.array([[4, 5], [5.5, -6]], dtype=numpy.float32)
                got = oinf.run({'X': x})
                self.assertEqualArray(x.sum(), got['Y'])

    def test_opset_squeeze(self):
        for opv in range(10, max_supported_opset() + 1):
            with self.subTest(opv=opv):
                node = OnnxSqueezeApi11(
                    'X', axes=numpy.array([0], dtype=numpy.int64),
                    op_version=opv, output_names=['Y'])
                onx = node.to_onnx(numpy.float32, numpy.float32,
                                   target_opset=opv)
                self.assertNotIn("elem_type: 0", str(onx))
                oinf = OnnxInference(onx)
                x = numpy.array([[4, 5]], dtype=numpy.float32)
                got = oinf.run({'X': x})
                self.assertEqualArray(numpy.squeeze(x, axis=0), got['Y'])

    def test_opset_unsqueeze(self):
        for opv in range(10, max_supported_opset() + 1):
            with self.subTest(opv=opv):
                node = OnnxUnsqueezeApi11(
                    'X', axes=numpy.array([0], dtype=numpy.int64),
                    op_version=opv, output_names=['Y'])
                onx = node.to_onnx(numpy.float32, numpy.float32,
                                   target_opset=opv)
                self.assertNotIn("elem_type: 0", str(onx))
                oinf = OnnxInference(onx)
                x = numpy.array([4, 5], dtype=numpy.float32)
                got = oinf.run({'X': x})
                self.assertEqualArray(x[numpy.newaxis, :], got['Y'])

    def test_opset_reshape(self):
        for opv in range(10, max_supported_opset() + 1):
            with self.subTest(opv=opv):
                node = OnnxReshapeApi13(
                    'X', numpy.array([2, 1, 1], dtype=numpy.int64),
                    op_version=opv, output_names=['Y'])
                onx = node.to_onnx(numpy.float32, numpy.float32,
                                   target_opset=opv)
                self.assertNotIn("elem_type: 0", str(onx))
                oinf = OnnxInference(onx)
                x = numpy.array([4, 5], dtype=numpy.float32)
                got = oinf.run({'X': x})
                self.assertEqualArray(
                    x[:, numpy.newaxis, numpy.newaxis], got['Y'])

    def test_opset_reduce_l2_typed(self):
        for dtype in [numpy.float32, numpy.float64]:
            for opv in range(10, max_supported_opset() + 1):
                with self.subTest(opv=opv, dtype=dtype):
                    node = OnnxReduceL2_typed(
                        dtype, 'X', numpy.array([1], dtype=numpy.int64),
                        op_version=opv, output_names=['Y'])
                    onx = node.to_onnx(numpy.float32, numpy.float32,
                                       target_opset=opv)
                    self.assertNotIn("elem_type: 0", str(onx))
                    oinf = OnnxInference(onx)
                    x = numpy.array([[4, 5], [6.7, 7.8]], dtype=numpy.float32)
                    got = oinf.run({'X': x})
                    self.assertEqualArray(
                        (x ** 2).sum(axis=1, keepdims=1) ** 0.5, got['Y'])

    def test_opset_split(self):
        OnnxSub = loadop("Sub")
        for dtype in [numpy.float32, numpy.float64]:
            for opv in range(10, max_supported_opset() + 1):
                with self.subTest(opv=opv, dtype=dtype):
                    node_split = OnnxSplitApi11(
                        'X', split=numpy.array([1, 1], dtype=numpy.int64),
                        axis=1, op_version=opv)
                    node1 = node_split[0]
                    node2 = node_split[1]
                    node = OnnxSub(node1, node2, op_version=opv,
                                   output_names=['Y'])
                    onx = node.to_onnx(numpy.float32, numpy.float32,
                                       target_opset=opv)
                    oinf = OnnxInference(onx, runtime='python_compiled')
                    x = numpy.array([[4, 5], [6.7, 7.8]], dtype=numpy.float32)
                    x_copy = x.copy()
                    expected = (x[:, :1] - x[:, 1:]).copy()
                    got = oinf.run({'X': x})
                    self.assertEqualArray(expected, got['Y'])
                    self.assertEqualArray(x, x_copy)
                    oinf = OnnxInference(onx, runtime='python')
                    x = numpy.array([[4, 5], [6.7, 7.8]], dtype=numpy.float32)
                    got = oinf.run({'X': x})
                    self.assertEqualArray(expected, got['Y'])
                    # This not always hold, computation may happen in place.
                    # self.assertEqualArray(x, x_copy)

    def test_opset_split_no_split(self):
        OnnxSub = loadop("Sub")
        for dtype in [numpy.float32, numpy.float64]:
            for opv in range(10, max_supported_opset() + 1):
                with self.subTest(opv=opv, dtype=dtype):
                    node_split = OnnxSplitApi11(
                        'X', axis=1, op_version=opv)
                    node1 = node_split[0]
                    node2 = node_split[1]
                    node = OnnxSub(node1, node2, op_version=opv,
                                   output_names=['Y'])
                    onx = node.to_onnx(numpy.float32, numpy.float32,
                                       target_opset=opv)
                    oinf = OnnxInference(onx, runtime='python_compiled')
                    x = numpy.array([[4, 5], [6.7, 7.8]], dtype=numpy.float32)
                    x_copy = x.copy()
                    expected = (x[:, :1] - x[:, 1:]).copy()
                    got = oinf.run({'X': x})
                    self.assertEqualArray(expected, got['Y'])
                    self.assertEqualArray(x, x_copy)
                    oinf = OnnxInference(onx, runtime='python')
                    x = numpy.array([[4, 5], [6.7, 7.8]], dtype=numpy.float32)
                    got = oinf.run({'X': x})
                    self.assertEqualArray(expected, got['Y'])
                    # This not always hold, computation may happen in place.
                    # self.assertEqualArray(x, x_copy)

    def test_zif(self):
        OnnxConstant, OnnxIf, OnnxGreater = loadop(
            "Constant", "If", "Greater")
        bthen = OnnxConstant(
            value_floats=numpy.array([0], dtype=numpy.float32),
            output_names=['res_then'])

        belse = OnnxConstant(
            value_floats=numpy.array([1], dtype=numpy.float32),
            output_names=['res_else'])

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
        self.assertEqualArray(
            numpy.array([0.], dtype=numpy.float32), got['Z'])

        x = numpy.array([-1, -2], dtype=numpy.float32)
        model_def = onx.to_onnx({'X': numpy.float32}, {'Z': numpy.float32})
        got = OnnxInference(model_def).run({'X': x})
        self.assertEqualArray(
            numpy.array([1.], dtype=numpy.float32), got['Z'])

    def test_zif2(self):
        OnnxAdd, OnnxSub, OnnxIf, OnnxGreater, OnnxReduceSum = loadop(
            "Add", "Sub", "If", "Greater", "ReduceSum")

        node = OnnxAdd('x1', 'x2', output_names=['absxythen'])
        then_body = node.to_onnx(
            [Variable('x1', numpy.float32), Variable('x2', numpy.float32)],
            {'absxythen': numpy.float32})
        node = OnnxSub('x1', 'x2', output_names=['absxyelse'])
        else_body = node.to_onnx(
            [Variable('x1', numpy.float32), Variable('x2', numpy.float32)],
            {'absxyelse': numpy.float32})
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(OnnxReduceSum('x1'), OnnxReduceSum('x2'))
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        output_names=['y'])
        model_def = ifnode.to_onnx(
            [Variable('x1', numpy.float32), Variable('x2', numpy.float32)],
            {'y': numpy.float32})
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn("_greater -> out_gre_0;", dot)

    def test_onnx_astype(self):
        OnnxIdentity = loadop("Identity")
        ovi = OnnxIdentity('X')
        last = OnnxIdentity(ovi.astype(numpy.int64), output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.int64, verbose=0)
        self.assertNotIn("elem_type: 0", str(onx))
        oinf = OnnxInference(onx)
        x = numpy.array([[-2, 2.5], [0, 3]], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('xop')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestXOps().test_onnx_ml_operator()
    unittest.main(verbosity=2)
