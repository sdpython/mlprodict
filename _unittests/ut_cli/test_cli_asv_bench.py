"""
@brief      test tree node (time=30s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliAsvBench(ExtTestCase):

    def test_cli_asv_bench(self):
        st = BufferedPrint()
        main(args=["asv_bench", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    def test_cli_asv_bench_model(self):
        temp = get_temp_folder(__file__, "temp_asv_bench")
        st = BufferedPrint()
        main(args=["asv_bench", "-l", temp,
                   "-o", '10', '-m',
                   "LogisticRegression,LinearRegression",
                   '-v', '2'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('Linear', res)
        name = "bench_LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10.py"
        self.assertExists(os.path.join(temp, 'benches', name))
        self.assertExists(os.path.join(temp, 'asv.conf.json'))
        self.assertExists(os.path.join(temp, 'tools', 'flask_serve.py'))


if __name__ == "__main__":
    # TestCliAsvBench().test_cli_validate_model_csv_bug()
    unittest.main()
