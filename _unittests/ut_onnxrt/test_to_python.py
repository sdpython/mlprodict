"""
@brief      test log(time=2s)
"""
import os
import unittest
from logging import getLogger
import textwrap
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.loghelper import run_script
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxTranspose
)
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx


class TestToPython(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_code_add_except(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        try:
            oinf.to_python()
        except ValueError:
            pass

    def auto_test_script(self, filename, test_code, test_out):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        test_code = textwrap.dedent(test_code)
        content += "\n" + test_code
        filename += ".test.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        out, err = run_script(filename, wait=True)
        if err is not None:
            err = err.strip("\n\r\t")
        if err is not None and len(err) > 0:
            raise RuntimeError(
                "Execution of '{}' failed.\n--OUT--\n{}\n--ERR--\n{}\n---".format(
                    filename, out, err))
        out = out.strip("\n\r")
        if test_out is not None:
            self.assertEqual(test_out, out)
        return out, err

    def test_code_add_transpose(self):
        idi = numpy.identity(2)
        onx = OnnxTranspose(OnnxAdd('X', idi), output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def, runtime='python')
        res = oinf.to_python(inline=False)
        self.assertNotEmpty(res)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 2)
        self.assertIn('onnx_pyrt_Ad_Addcst.pkl', res)
        self.assertIn('onnx_pyrt_main.py', res)
        cd = res['onnx_pyrt_main.py']
        self.assertIn('def pyrt_Add(X, Ad_Addcst):', cd)
        self.assertIn('def run(self, X):', cd)
        # inline
        temp = get_temp_folder(__file__, "temp_code_add_transpose")
        res = oinf.to_python(inline=True, dest=temp)
        self.assertNotEmpty(res)
        name = os.path.join(temp, 'onnx_pyrt_main.py')
        self.assertExists(name)
        # test code
        test_code = """
            X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
            oinf = OnnxPythonInference()
            Y = oinf.run(X)
            print(Y)
            """
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        exp = oinf.run({'X': X})['Y']
        sexp = str(exp)
        self.auto_test_script(name, test_code, sexp)


if __name__ == "__main__":
    unittest.main()
