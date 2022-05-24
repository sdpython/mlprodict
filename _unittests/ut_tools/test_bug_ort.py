"""
@brief      test log(time=14s)
"""
import os
import unittest
import numpy
from onnx import load
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper.version_helper import compare_module_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.text_plot import onnx_simple_text_plot


def get_ort_version():
    import onnxruntime
    return onnxruntime.__version__


print(get_ort_version())


class TestBugOrt(ExtTestCase):

    @unittest.skipIf(compare_module_version(get_ort_version(), '1.12') <= 0,
                     reason="see https://github.com/microsoft/onnxruntime/issues/11614")
    def test_weird_behaviour(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        onx1 = os.path.join(data, "dft_last_axis.onnxruntime1.output.onnx")
        onx2 = os.path.join(data, "dft_last_axis.error.ort.exec.onnx")
        inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                  'fft_length': numpy.array([5], dtype=numpy.int64),
                  'onesided': numpy.array([0], dtype=numpy.int64),
                  'inverse': numpy.array([0], dtype=numpy.int64),
                  'normalize': numpy.array([0], dtype=numpy.int64)}
        # with open("debug1.txt", "w") as f:
        #     with open(onx1, "rb") as g:
        #         f.write(onnx_simple_text_plot(load(g), recursive=True))
        # with open("debug2.txt", "w") as f:
        #     with open(onx2, "rb") as g:
        #         f.write(onnx_simple_text_plot(load(g), recursive=True))
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt, case='no-unused'):
                oinf1 = OnnxInference(onx1, runtime=rt)
                res1 = oinf1.run(inputs)

            with self.subTest(runtime=rt, case='with-unused'):
                oinf2 = OnnxInference(onx2, runtime=rt)
                res2 = oinf2.run(inputs)
                self.assertEqualArray(res1["output"], res1["output"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
