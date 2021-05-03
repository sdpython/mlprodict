"""
@brief      test tree node (time=4s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliEinsum(ExtTestCase):

    def test_cli_einsum(self):
        st = BufferedPrint()
        main(args=["einsum_test", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    def test_cli_excel(self):
        temp = get_temp_folder(__file__, "temp_cli_excel")
        name = os.path.join(temp, "res.xlsx")
        st = BufferedPrint()
        main(args=["einsum_test", "--equation", "abc,cd->ad",
                   "--output", name, "--shape", "5",
                   "--verbose", "0"], fLOG=st.fprint)
        self.assertExists(name)
        res = str(st)
        self.assertIn("wrote", res)

    def test_cli_csv(self):
        temp = get_temp_folder(__file__, "temp_cli_csv")
        name = os.path.join(temp, "res.csv")
        st = BufferedPrint()
        main(args=["einsum_test", "--equation", "abc,cd->ad",
                   "--output", name, "--shape", "(5,5,5);(5,5)",
                   "--verbose", "0"], fLOG=st.fprint)
        self.assertExists(name)
        res = str(st)
        self.assertIn("wrote", res)

    def test_cli_csv_n(self):
        temp = get_temp_folder(__file__, "temp_cli_csvn")
        name = os.path.join(temp, "res.csv")
        st = BufferedPrint()
        main(args=["einsum_test", "--equation", "abc,cd->ad",
                   "--output", name, "--shape", "5,5",
                   "--verbose", "0"], fLOG=st.fprint)
        self.assertExists(name)
        res = str(st)
        self.assertIn("wrote", res)

    def test_cli_csv_rt(self):
        temp = get_temp_folder(__file__, "temp_cli_csv_rt")
        name = os.path.join(temp, "res.csv")
        st = BufferedPrint()
        main(args=["einsum_test", "--equation", "abc,cd->ad",
                   "--output", name, "--shape", "(5,5,5);(5,5)",
                   "--verbose", "0", "--runtime", "onnxruntime"],
             fLOG=st.fprint)
        self.assertExists(name)
        res = str(st)
        self.assertIn("wrote", res)

    def test_cli_csv_perm(self):
        temp = get_temp_folder(__file__, "temp_cli_csv_perm")
        name = os.path.join(temp, "res.csv")
        st = BufferedPrint()
        main(args=["einsum_test", "--equation", "abc,cd->ad",
                   "--output", name, "--shape", "(5,5,5);(5,5)",
                   "--verbose", "0", "--perm", "1"], fLOG=st.fprint)
        self.assertExists(name)
        res = str(st)
        self.assertIn("wrote", res)


if __name__ == "__main__":
    unittest.main()
