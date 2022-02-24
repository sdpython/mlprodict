"""
@brief      test tree node (time=42s)
"""
import os
import sys
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import (
    ExtTestCase, get_temp_folder, ignore_warnings)
from mlprodict.__main__ import main


class TestCliValidateBenchDoc(ExtTestCase):

    @ignore_warnings(UserWarning)
    def test_cli_validate_bench_doc_help(self):
        st = BufferedPrint()
        main(args=["benchmark_doc", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    @ignore_warnings(UserWarning)
    def test_cli_validate_bench_doc(self):
        temp = get_temp_folder(__file__, "temp_bench_doc")
        out1 = os.path.join(temp, "raw.xlsx")
        out2 = os.path.join(temp, "sum.csv")
        st = BufferedPrint()
        main(args=["benchmark_doc", "-o", out1, "-ou", out2, "-w",
                   "LinearRegression", '-d', temp,
                   '-r', 'python_compiled'],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('Linear', res)
        self.assertExists(out1)
        self.assertExists(out2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
