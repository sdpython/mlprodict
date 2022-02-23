"""
@brief      test log(time=10s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.onnx_backend import enumerate_onnx_tests
from mlprodict.onnxrt import OnnxInference


class TestOnnxBackEnd(ExtTestCase):

    @staticmethod
    def load_fct(obj, runtime='python'):
        return OnnxInference(obj, runtime)

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
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
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
                te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            except NotImplementedError as e:
                missed.append((te, e))
                continue
            except (IndexError, RuntimeError, TypeError, ValueError,
                    AttributeError) as e:
                failed.append((te, e))
                continue
            except AssertionError as e:
                mismatch.append((te, e))
                continue

        if __name__ == '__main__':
            print(len(missed), len(failed), len(mismatch))
            for t in missed:
                print("missed", t[0])
            for t in failed:
                print("failed", t[0])
            for t in mismatch:
                print("mismatch", t[0])


if __name__ == "__main__":
    unittest.main()
