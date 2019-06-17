"""
@brief      test tree node (time=4s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliValidate(ExtTestCase):

    def test_cli_validate(self):
        st = BufferedPrint()
        main(args=["validate_runtime", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    def test_cli_validate_model(self):
        temp = get_temp_folder(__file__, "temp_validate_model")
        out1 = os.path.join(temp, "raw.xlsx")
        out2 = os.path.join(temp, "sum.xlsx")
        st = BufferedPrint()
        main(args=["validate_runtime", "--out_raw", out1,
                   "--out_summary", out2, "--models",
                   "LogisticRegression,LinearRegression"],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('Linear', res)
        self.assertExists(out1)
        self.assertExists(out2)


if __name__ == "__main__":
    unittest.main()
