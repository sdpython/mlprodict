"""
@brief      test tree node (time=30s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliValidateProcess(ExtTestCase):

    def test_cli_validate_model_process_csv(self):
        temp = get_temp_folder(__file__, "temp_validate_model_process_csv")
        out1 = os.path.join(temp, "raw.csv")
        out2 = os.path.join(temp, "sum.csv")
        st = BufferedPrint()
        main(args=["validate_runtime", "--out_raw", out1,
                   "--out_summary", out2, "--models",
                   "LogisticRegression,LinearRegression",
                   '-o', '10', '-op', '10', '-v', '1', '-b', '1',
                   '-se', '1'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('Linear', res)
        self.assertExists(out1)
        self.assertExists(out2)


if __name__ == "__main__":
    unittest.main()
