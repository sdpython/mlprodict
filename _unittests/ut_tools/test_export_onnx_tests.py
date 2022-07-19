# pylint: disable=W0201
"""
@brief      test log(time=40s)
"""
import unittest
import collections
import inspect
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import numpy
from onnx import numpy_helper
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info, make_opsetid, make_function)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx_export import export2python
from mlprodict.testing.verify_code import verify_code
from mlprodict.tools.code_helper import print_code
from mlprodict.testing.onnx_backend import enumerate_onnx_tests
from mlprodict.onnx_tools.model_checker import check_onnx


class TestExportOnnx(ExtTestCase):

    def verify(self, content, more_context=None, limit_left=10):
        try:
            left, __ = verify_code(content, exc=False)
        except (SyntaxError, AttributeError) as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {'numpy_helper': numpy_helper,
               'make_model': make_model,
               'make_node': make_node,
               'set_model_props': set_model_props,
               'make_tensor': make_tensor,
               'make_graph': make_graph,
               'make_function': make_function,
               'make_tensor_value_info': make_tensor_value_info,
               'print': print, 'sorted': sorted,
               'make_opsetid': make_opsetid,
               'collections': collections, 'inspect': inspect}
        if more_context is not None:
            loc.update(more_context)
            glo.update(more_context)
        out, err = StringIO(), StringIO()
        if limit_left is not None and len(left) >= limit_left:
            raise AssertionError(
                f"Too many unknown symbols ({len(left)}): {left!r} in\n{content}")

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_export_all(self):

        class LocalDomain:
            def __init__(self, domain, version):
                self.domain = domain
                self.version = version

        context = {'mlprodict1': LocalDomain('mlprodict', 1)}
        for i in range(0, 17):
            op = LocalDomain('', i)
            op.ReduceSum = numpy.sum
            op.Identity = lambda i: i
            op.Constant = lambda value: numpy_helper.to_array(value)
            context['opset%d' % i] = op

        for te in enumerate_onnx_tests('node'):
            with self.subTest(name=te.name):
                if te.name in {'test_if_opt',
                               'test_loop11',
                               'test_loop13_seq',
                               'test_loop16_seq_none',
                               'test_range_float_type_positive_delta_expanded',
                               'test_range_int32_type_negative_delta_expanded',
                               'test_scan9_sum',
                               'test_scan_sum',
                               'test_sequence_map_add_1_sequence_1_tensor',
                               'test_sequence_map_add_1_sequence_1_tensor_expanded',
                               'test_sequence_map_add_2_sequences',
                               'test_sequence_map_add_2_sequences_expanded',
                               'test_sequence_map_extract_shapes',
                               'test_sequence_map_extract_shapes_expanded',
                               'test_sequence_map_identity_1_sequence',
                               'test_sequence_map_identity_1_sequence_1_tensor',
                               'test_sequence_map_identity_1_sequence_1_tensor_expanded',
                               'test_sequence_map_identity_1_sequence_expanded',
                               'test_sequence_map_identity_2_sequences',
                               'test_sequence_map_identity_2_sequences_expanded',
                               }:
                    continue
                check_onnx(te.onnx_model)
                try:
                    new_onnx = export2python(te.onnx_model, name="TEST")
                except Exception as e:
                    raise AssertionError(
                        "Unable to convert test %r and model\n%s" % (
                            te.name, te.onnx_model)) from e
                _, loc = self.verify(
                    new_onnx, more_context=context, limit_left=None)
                self.assertIn('main', loc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
