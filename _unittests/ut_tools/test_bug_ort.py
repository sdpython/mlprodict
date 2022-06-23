# pylint: disable=W0703,W0632
"""
@brief      test log(time=14s)
"""
import os
import unittest
import numpy
from onnx import load
from onnx.shape_inference import infer_shapes
from onnx.checker import check_model
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.texthelper.version_helper import compare_module_version
from pyquickhelper.texthelper.edit_text_diff import (
    diff2html, edit_distance_text)
from mlprodict.onnx_tools.model_checker import check_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.npy.xop import loadop
from mlprodict.onnx_tools.onnx_manipulations import change_subgraph_io_type_shape


def get_ort_version():
    import onnxruntime
    return onnxruntime.__version__


class TestBugOrt(ExtTestCase):

    def common_test_weird_behaviour(self, onx1, onx2, temp, inputs, output):
        rows_base = onnx_simple_text_plot(
            onx1, recursive=True, indent=False).split('\n')
        rows_new = onnx_simple_text_plot(
            onx2, recursive=True, indent=False).split('\n')
        _, aligned, final = edit_distance_text(rows_base, rows_new)
        ht = diff2html(rows_base, rows_new, aligned, final,
                       two_columns=True)
        with open(os.path.join(temp, "diff.html"), 'w', encoding='utf-8') as f:
            f.write(ht)

        # very long
        rows_base = str(onx1).split('\n')
        rows_new = str(onx2).split('\n')
        _, aligned, final = edit_distance_text(rows_base, rows_new)
        ht = diff2html(rows_base, rows_new, aligned, final,
                       two_columns=True)
        with open(os.path.join(temp, "diff.json.html"), 'w', encoding='utf-8') as f:
            f.write(ht)

        err = {}
        try:
            # : ValidationError: Field 'shape' of type is required but missing.
            check_onnx(onx1)
        except Exception as e:
            err['check', 1] = e
        try:
            check_onnx(onx2)
        except Exception as e:
            err['check', 2] = e
        try:
            infer_shapes(onx1, check_type=True, strict_mode=True)
        except Exception as e:
            err['shape', 1] = e
        try:
            infer_shapes(onx2, check_type=True, strict_mode=True)
        except Exception as e:
            err['shape', 2] = e

        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt, case='no-unused'):
                oinf1 = OnnxInference(onx1.SerializeToString(), runtime=rt)
                res1 = oinf1.run(inputs)
            with self.subTest(runtime=rt, case='with-unused'):
                oinf2 = OnnxInference(onx2.SerializeToString(), runtime=rt)
                res2 = oinf2.run(inputs)
                self.assertEqualArray(res1[output], res2[output])
        return err

    @unittest.skipIf(compare_module_version(get_ort_version(), '1.12') <= 0,
                     reason="see https://github.com/microsoft/onnxruntime/issues/11614")
    def test_weird_behaviour1(self):
        inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                  'fft_length': numpy.array([5], dtype=numpy.int64),
                  'onesided': numpy.array([0], dtype=numpy.int64),
                  'inverse': numpy.array([0], dtype=numpy.int64),
                  'normalize': numpy.array([0], dtype=numpy.int64)}
        temp = get_temp_folder(__file__, "temp_weird_behaviour1")
        data = os.path.join(os.path.dirname(__file__), "data")
        onx1 = os.path.join(data, "dft_last_axis.onnxruntime1.output.onnx")
        onx2 = os.path.join(data, "dft_last_axis.error.ort.exec.onnx")
        err = self.common_test_weird_behaviour(
            load(onx1), load(onx2), temp, inputs, 'output')
        self.assertLess(len(err), 2)

    def test_weird_behaviour2(self):
        inputs = {'X': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32)}
        OnnxAbs = loadop('Abs')
        temp = get_temp_folder(__file__, "temp_weird_behaviour2")
        onx1 = OnnxAbs('X', output_names=['Y']).to_onnx(
            numpy.float32, numpy.float32)
        onx2 = OnnxAbs('X', output_names=['Y']).to_onnx(
            numpy.float32, numpy.float32)
        onx2 = change_subgraph_io_type_shape(
            onx2, shape_changes={'X': [], 'Y': []})
        err = self.common_test_weird_behaviour(onx1, onx2, temp, inputs, 'Y')
        self.assertLess(len(err), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
