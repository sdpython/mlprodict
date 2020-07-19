"""
@brief      test tree node (time=7s)
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
                   '-v', '2', '--flat', '1',
                   '--matrix', '{"onnxruntime":["1.1.1","1.1.2"]}'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('Lin', res)
        name = "bench_LogReg_liblinear_b_cl_solverliblinear_onnx.py"
        self.assertExists(os.path.join(temp, 'benches', name))
        self.assertExists(os.path.join(temp, 'asv.conf.json'))
        self.assertExists(os.path.join(temp, 'tools', 'flask_serve.py'))
        conf = os.path.join(temp, 'asv.conf.json')
        with open(conf, "r") as f:
            content = f.read()
        self.assertIn('"1.1.1"', content)

    def test_cli_asv_bench_model2(self):
        temp = get_temp_folder(__file__, "temp_asv_bench2")
        st = BufferedPrint()
        main(args=["asv_bench", "-l", temp,
                   "-o", '10', '-m',
                   "LogisticRegression,LinearRegression",
                   '-v', '0', '--flat', '1',
                   '--matrix', '{"onnxruntime":["1.1.1","1.1.2"]}',
                   '--dtype', '64'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('Lin', res)
        name = "bench_LogReg_liblinear_b_cl_64_solverliblinear_onnx.py"
        self.assertExists(os.path.join(temp, 'benches', name))
        self.assertExists(os.path.join(temp, 'asv.conf.json'))
        self.assertExists(os.path.join(temp, 'tools', 'flask_serve.py'))
        conf = os.path.join(temp, 'asv.conf.json')
        with open(conf, "r") as f:
            content = f.read()
        self.assertIn('"1.1.1"', content)


if __name__ == "__main__":
    unittest.main()
