"""
@brief      test log(time=4s)
"""
import unittest
import inspect
import ast
from textwrap import dedent
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_grammar import CodeNodeVisitor, translate_fct2onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxGrammarTranslate(ExtTestCase):

    def test_tree_job(self):

        def addition(x, y):
            z = x + numpy.abs(y)
            return x * z

        code = inspect.getsource(addition)
        node = ast.parse(dedent(code))
        stack = [(0, node)]

        while len(stack) > 0:
            ind, n = stack[-1]
            del stack[-1]
            att = {name: ch for name, ch in ast.iter_fields(n)}
            self.assertTrue(att is not None)
            for ch in ast.iter_child_nodes(n):
                stack.append((ind + 1, ch))
        self.assertEmpty(stack)

    def test_translation(self):

        def addition(x, y):
            z = x + numpy_dot(numpy_abs(y), x)  # pylint: disable=E0602
            return x * z

        code = inspect.getsource(addition)
        node = ast.parse(dedent(code))
        v = CodeNodeVisitor()
        v.visit(node)
        self.assertEqual(len(v.Rows), 21)
        st = v.print_tree()
        self.assertIn('BinOp', st)
        st = CodeNodeVisitor.print_node(node)
        self.assertIn('body', st)
        code = v._translator._code_fct  # pylint: disable=W0212
        exp = ('FunctionDef',
               {'args': [('x', None), ('y', None)],
                'code': [('Assign',
                          {'name': 'z',
                           'args': [('BinOp',
                                     {'args': ['x',
                                               ('Call',
                                                {'args': ['numpy_dot',
                                                          ('Call',
                                                           {'args': ['numpy_abs', 'y'],
                                                            'col_offset': 22,
                                                            'lineno': 2,
                                                            'name': 'numpy_abs'}),
                                                          'x'],
                                                 'col_offset': 12,
                                                 'lineno': 2,
                                                 'name': 'numpy_dot'})],
                                      'col_offset': 8,
                                      'lineno': 2,
                                      'op': 'Add'})],
                           'col_offset': 4,
                           'lineno': 2}),
                         ('Return',
                          {'code': [('BinOp',
                                     {'args': ['x', 'z'],
                                      'col_offset': 11,
                                      'lineno': 3,
                                      'op': 'Mult'})],
                           'col_offset': 4,
                           'lineno': 3})],
                'col_offset': 0,
                'lineno': 1,
                'name': 'addition'})
        self.assertEqual(exp, code)

    def test_translate_nested(self):

        def addition(x, y):
            def add_nested(x, y):
                return x + y
            z = add_nested(x, numpy.abs(y))
            return x * z

        code = inspect.getsource(addition)
        v = CodeNodeVisitor()
        node = ast.parse(dedent(code))
        self.assertRaise(lambda: v.visit(node), RuntimeError, "Nested")

    def test_export(self):

        numpy_abs = numpy.abs

        def addition(x, y):
            z = x + numpy_abs(y)
            return x * z

        code = inspect.getsource(addition)
        node = ast.parse(dedent(code))
        v = CodeNodeVisitor()
        v.visit(node)
        onnx_code = v.export(context={'numpy_abs': numpy.abs})
        exp = dedent("""
            def addition(x, y):
                z = (
                    OnnxAdd(
                        x,
                        OnnxAbs(
                            y
                        )
                    )
                )
                return (
                    OnnxMul(
                        x,
                        z
                    )
                )
        """)
        self.assertEqual(exp.strip('\n '), onnx_code.strip('\n '))

    def test_export_error(self):

        numpy_abs = numpy.abs

        def addition(x, y):
            z = x + numpy_abs(y)
            return x * z

        code = inspect.getsource(addition)
        node = ast.parse(dedent(code))
        v = CodeNodeVisitor()
        v.visit(node)
        self.assertRaise(lambda: v.export(), RuntimeError, 'numpy_abs')
        self.assertRaise(lambda: v.export(), RuntimeError, 'line 2')

    def test_export_numpy(self):

        def addition(x, y):
            z = x + numpy.abs(y)
            return x * z

        code = inspect.getsource(addition)
        node = ast.parse(dedent(code))
        v = CodeNodeVisitor()
        v.visit(node)
        onnx_code = v.export(context={'numpy.abs': numpy.abs})
        exp = dedent("""
            def addition(x, y):
                z = (
                    OnnxAdd(
                        x,
                        OnnxAbs(
                            y
                        )
                    )
                )
                return (
                    OnnxMul(
                        x,
                        z
                    )
                )
        """)
        self.assertEqual(exp.strip('\n '), onnx_code.strip('\n '))

    def test_export_transpose(self):

        def trs(x, y):
            z = x + numpy.transpose(y, axes=[1, 0])
            return x * z

        code = inspect.getsource(trs)
        node = ast.parse(dedent(code))
        v = CodeNodeVisitor()
        v.visit(node)
        onnx_code = v.export(context={'numpy.transpose': numpy.transpose})
        exp = dedent("""
            def trs(x, y):
                z = (
                    OnnxAdd(
                        x,
                        OnnxTranspose(
                            y,
                            perm=[1, 0]
                        )
                    )
                )
                return (
                    OnnxMul(
                        x,
                        z
                    )
                )
        """)
        self.assertEqual(exp.strip('\n '), onnx_code.strip('\n '))

    def test_export_transpose_compile(self):

        def trs(x, y):
            z = x + numpy.transpose(y, axes=[1, 0])
            return x * z

        onnx_code = translate_fct2onnx(
            trs, context={'numpy.transpose': numpy.transpose},
            output_names=['Z'])
        exp = dedent("""
            def trs(x, y):
                z = (
                    OnnxAdd(
                        x,
                        OnnxTranspose(
                            y,
                            perm=[1, 0]
                        )
                    )
                )
                return OnnxIdentity(
                    OnnxMul(
                        x,
                        z
                    ),
                    output_names=['Z']
                )
        """)
        self.assertEqual(exp.strip('\n '), onnx_code.strip('\n '))

        fct = translate_fct2onnx(
            trs, context={'numpy.transpose': numpy.transpose},
            cpl=True, output_names=['Z'])
        self.assertTrue(callable(fct))

        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxTranspose, OnnxMul, OnnxIdentity
        )
        ctx = {'OnnxAdd': OnnxAdd,
               'OnnxTranspose': OnnxTranspose,
               'OnnxMul': OnnxMul, 'OnnxIdentity': OnnxIdentity}

        fct = translate_fct2onnx(
            trs, context={'numpy.transpose': numpy.transpose},
            cpl=True, context_cpl=ctx, output_names=['Z'])

        r = fct('x', 'y')
        self.assertIsInstance(r, OnnxIdentity)

        inputs = {'x': numpy.array([[1, 2]], dtype=numpy.float32),
                  'y': numpy.array([[-0.3, 0.4]], dtype=numpy.float32).T}

        expected = trs(inputs['x'], inputs['y'])

        onnx_g = r.to_onnx(inputs)

        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        self.assertEqualArray(expected, res['Z'])


if __name__ == "__main__":
    unittest.main()
