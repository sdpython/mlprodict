"""
@brief      test log(time=4s)
"""
import unittest
import ast
import inspect
from textwrap import dedent
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx_grammar import CodeNodeVisitor


class TestOnnxGrammarBug(ExtTestCase):

    def test_bug1(self):

        def norm2(x, y):
            delta = x - y
            n = delta ** 2
            return n

        code = dedent(inspect.getsource(norm2))
        node = ast.parse(code)
        v = CodeNodeVisitor()
        v.visit(node)
        rows = []
        for r in v.Rows:
            rows.append(
                f"{'    ' * r['indent']}{r['type']}: {r['str']}")
        final = "\n".join(rows)
        self.assertIn("Assign:", final)


if __name__ == "__main__":
    unittest.main()
