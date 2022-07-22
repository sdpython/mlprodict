"""
@brief      test tree node (time=23s)
"""
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase
from mlprodict.__main__ import main


class TestCliDynamicDoc(ExtTestCase):

    def test_cli_onnx_code_help(self):
        st = BufferedPrint()
        main(args=["dynamic_doc", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("Generates", res)

    def test_cli_onnx_code(self):
        st = BufferedPrint()
        main(args=["dynamic_doc", '--verbose', '1'], fLOG=st.fprint)
        res = str(st)
        if len(res) > 0:
            self.assertIn("Abs", res)


if __name__ == "__main__":
    unittest.main()
