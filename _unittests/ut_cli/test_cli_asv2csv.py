"""
@brief      test tree node (time=30s)
"""
import os
import unittest
import pandas
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.filehelper.compression_helper import unzip_files
from mlprodict.__main__ import main


class TestCliAsvBench(ExtTestCase):

    data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

    def test_cli_asv_bench(self):
        st = BufferedPrint()
        main(args=["asv2csv", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("outfile", res)

    def test_cli_asv2csv(self):
        temp = get_temp_folder(__file__, "temp_asv2csv")
        file_zip = os.path.join(TestCliAsvBench.data, 'results.zip')
        unzip_files(file_zip, temp)
        data = os.path.join(temp, 'results')

        out = os.path.join(temp, "data.csv")
        st = BufferedPrint()
        main(args=["asv2csv", "-f", data, "-o", out], fLOG=st.fprint)
        self.assertExists(out)
        df = pandas.read_csv(out)
        self.assertEqual(df.shape, (168, 66))
        out = os.path.join(temp, "data<date>.csv")
        main(args=["asv2csv", "-f", data, "-o", out], fLOG=st.fprint)
        main(args=["asv2csv", "-f", data], fLOG=st.fprint)


if __name__ == "__main__":
    # TestCliAsvBench().test_cli_validate_model_csv_bug()
    unittest.main()
