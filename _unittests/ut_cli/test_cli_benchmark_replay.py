"""
@brief      test tree node (time=120s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliBenchmarkReplay(ExtTestCase):

    def test_cli_benchmark_replay_help(self):
        st = BufferedPrint()
        main(args=["benchmark_replay", "--help"],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('benchmark_replay', res)

    def test_cli_benchmark_replay(self):
        temp = get_temp_folder(__file__, "temp_benchmark_replay")
        out1 = os.path.join(temp, "raw.csv")
        st = BufferedPrint()
        out1 = os.path.join(temp, "raw.csv")
        st = BufferedPrint()
        main(args=["validate_runtime", "--n_features", "4", "-nu", "3",
                   "-re", "3", "-o", "11", "-op", "11", "-v", "2", "--out_raw",
                   out1, "-b", "0",
                   "--runtime", "python_compiled",
                   "--models", "KMeans", "--dtype", "32",
                   "--dump_all", '1', '--dump_folder', temp],
             fLOG=st.fprint)
        out = os.path.join(temp, "res.xlsx")
        main(args=["benchmark_replay", "--folder", temp, "--out", out, '--verbose', '2'],
             fLOG=st.fprint)
        res = str(st)
        self.assertExists(out)
        self.assertIn("'folder'", res)


if __name__ == "__main__":
    unittest.main()
