"""
@brief      test tree node (time=15s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliValidateBug(ExtTestCase):

    def test_cli_validate_model_rfbug(self):
        temp = get_temp_folder(__file__, "temp_validate_model_rfbug")
        out1 = os.path.join(temp, "raw.xlsx")
        out2 = os.path.join(temp, "sum.xlsx")
        gr = os.path.join(temp, 'gr.png')
        st = BufferedPrint()
        main(args=["validate_runtime", "--out_raw", out1,
                   "--out_summary", out2,
                   '-o', '11', '-op', '11', '-v', '2', '-b', '1',
                   '--runtime', 'python_compiled,onnxruntime1',
                   '--models', 'RandomForestRegressor', '--n_features', '4',
                   '--out_graph', gr, '--dtype', '32'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('RandomForestRegressor', res)
        self.assertIn('time_kwargs', res)
        self.assertExists(out1)
        self.assertExists(out2)
        self.assertExists(gr)

    def test_cli_validate_model_rfbug_410(self):
        temp = get_temp_folder(__file__, "temp_validate_model_rfbug410")
        out1 = os.path.join(temp, "raw.xlsx")
        out2 = os.path.join(temp, "sum.xlsx")
        gr = os.path.join(temp, 'gr.png')
        st = BufferedPrint()
        main(args=["validate_runtime", "--out_raw", out1,
                   "--out_summary", out2,
                   '-o', '11', '-op', '11', '-v', '2', '-b', '1',
                   '--runtime', 'python_compiled,onnxruntime1',
                   '--models', 'RandomForestRegressor', '--n_features', '4,10',
                   '--out_graph', gr, '--dtype', '32'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('RandomForestRegressor', res)
        self.assertIn('time_kwargs', res)
        self.assertExists(out1)
        self.assertExists(out2)
        self.assertExists(gr)


if __name__ == "__main__":
    unittest.main()
