# pylint: disable=E0611
"""
@brief      test log(time=5s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_variable import Variable
from mlprodict.npy.xop_ops import _GraphBuilder
from mlprodict.onnxrt import OnnxInference


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
        y = numpy.array([-1, -3], dtype=numpy.float32)
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


if __name__ == "__main__":
    unittest.main()
