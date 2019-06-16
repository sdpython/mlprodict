"""
@brief      test tree node (time=4s)
"""
import unittest
from pyquickhelper.loghelper import BufferedPrint
from mlprodict.__main__ import main


class TestCliValidate(unittest.TestCase):

    def test_cli_validate(self):
        st = BufferedPrint()
        main(args=["validate_runtime", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)


if __name__ == "__main__":
    unittest.main()
