"""
@brief      test log(time=10s)
"""
import os
import unittest
from numpy import array, float32, int64
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_graph,
    make_tensor_value_info, __file__ as onnx_file)
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.testing.onnx_backend import (
    enumerate_onnx_tests, assert_almost_equal_string)
from mlprodict.onnxrt.onnx_micro_runtime import OnnxMicroRuntime


class TestOnnxBackEndMicro(ExtTestCase):

    @staticmethod
    def load_fct(obj):
        return OnnxMicroRuntime(obj)

    @staticmethod
    def run_fct(obj, *inputs):
        names = obj.input_names
        if len(names) < len(inputs):
            raise AssertionError(
                "Got %d inputs but expecting %d." % (
                    len(inputs), len(names)))
        feeds = {names[i]: inputs[i] for i in range(len(inputs))}
        got = obj.run(feeds)

        names = obj.output_names
        return [got[n] for n in names]

    def test_enumerate_onnx_tests_run_one(self):
        done = 0
        for te in enumerate_onnx_tests('node', lambda folder: folder == 'test_abs'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEndMicro.load_fct, TestOnnxBackEndMicro.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_tests_run(self):

        self.assertRaise(lambda: list(
            enumerate_onnx_tests('NNN')), FileNotFoundError)
        missed = []
        failed = []
        mismatch = []
        for te in enumerate_onnx_tests('node'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            try:
                te.run(TestOnnxBackEndMicro.load_fct,
                       TestOnnxBackEndMicro.run_fct)
            except NotImplementedError as e:
                missed.append((te, e))
                continue
            except (IndexError, RuntimeError, TypeError, ValueError,
                    AttributeError, KeyError) as e:
                failed.append((te, e))
                continue
            except AssertionError as e:
                mismatch.append((te, e))
                continue

        if __name__ == '__main__':
            path = os.path.dirname(onnx_file)
            print(len(missed), len(failed), len(mismatch))
            for t in failed:
                print("failed",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))
            for t in mismatch:
                print("mismatch",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))
            for t in missed:
                print("missed",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))


if __name__ == "__main__":
    # TestOnnxBackEnd().test_cast_FLOAT_to_STRING()
    unittest.main()
