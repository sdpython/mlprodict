"""
@brief      test tree node (time=30s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, skipif_circleci
from mlprodict.__main__ import main


class TestCliValidateDump(ExtTestCase):

    @skipif_circleci('too long')
    def test_cli_validate_model_dump(self):
        fLOG(OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_validate_model_dump")
        out1 = os.path.join(temp, "raw.csv")
        out2 = os.path.join(temp, "sum.csv")
        st = BufferedPrint()
        args = ["validate_runtime", "--out_raw", out1,
                "--out_summary", out2, "--models",
                "LinearRegression", '-r', "python",
                '-o', '10', '-op', '10', '-v', '1', '-b', '1',
                '-dum', '1', '-du', temp, '-n', '20,100,500']
        cmd = "python -m mlprodict " + " ".join(args)
        fLOG(cmd)
        main(args=args, fLOG=fLOG if __name__ == "__main__" else st.fprint)
        names = os.listdir(temp)
        names = [_ for _ in names if "dump-i-" in _]
        self.assertNotEmpty(names)


if __name__ == "__main__":
    unittest.main()
