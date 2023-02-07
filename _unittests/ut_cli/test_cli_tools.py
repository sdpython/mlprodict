"""
@brief      test tree node (time=4s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliTools(ExtTestCase):

    def test_cli_tools(self):
        st = BufferedPrint()
        main(args=["replace_initializer", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)


if __name__ == "__main__":
    unittest.main()
