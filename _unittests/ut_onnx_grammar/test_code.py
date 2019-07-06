"""
@brief      test log(time=1s)
"""
import unittest
import inspect
import ast
from textwrap import dedent
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_grammar import CodeNodeVisitor


class TestCode(ExtTestCase):

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
                'name': 'addition',
                'code': [('Assign',
                          {'args': [('BinOp',
                                     {'args': ['x',
                                               ('Call',
                                                {'name': 'numpy_dot',
                                                 'args': ['numpy_dot',
                                                          ('Call',
                                                           {'name': 'numpy_abs',
                                                            'args': ['numpy_abs', 'y']}),
                                                          'x']})],
                                      'op': 'Add'})],
                           'Name': 'z'}),
                         ('Return',
                          {'code': [('BinOp',
                                     {'args': ['x', 'z'],
                                      'op': 'Mult'})]})]})
        self.assertEqual(exp, code)

    def test_translate_nested(self):

        def addition(x, y):
            def add_nested(x, y):
                return x + y
            z = add_nested(x, numpy.abs(y))
            return x * z

        code = inspect.getsource(addition)
        node = ast.parse(dedent(code))
        v = CodeNodeVisitor()
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
                    OnnxMult(
                        x,
                        z
                    )
                )
        """)
        self.assertEqual(exp.strip('\n '), onnx_code.strip('\n '))


if __name__ == "__main__":
    unittest.main()
