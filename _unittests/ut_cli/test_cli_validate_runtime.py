"""
@brief      test tree node (time=30s)
"""
import os
import unittest
import pandas
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.onnxrt.validate.validate_summary import (
    merge_benchmark, summary_report)
from mlprodict.__main__ import main


class TestCliValidateRuntime(ExtTestCase):

    def test_cli_validate_kmeans(self):
        temp = get_temp_folder(__file__, "temp_validate_runtime_kmeans")
        out1 = os.path.join(temp, "raw.csv")
        out2 = os.path.join(temp, "sum.csv")
        gr = os.path.join(temp, 'gr.png')
        st = BufferedPrint()
        main(args=["validate_runtime", "--n_features", "4,50", "-nu", "3",
                   "-re", "3", "-o", "11", "-op", "11", "-v", "2", "--out_raw",
                   out1, "--out_summary", out2, "-b", "1",
                   "--runtime", "python_compiled,onnxruntime1",
                   "--models", "KMeans", "--out_graph", gr, "--dtype", "32"],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn('KMeans', res)
        self.assertExists(out1)
        self.assertExists(out2)
        self.assertExists(gr)
        df1 = pandas.read_csv(out1)
        merged = merge_benchmark({'r1-': df1, 'r2-': df1.copy()},
                                 baseline='r1-onnxruntime1')
        add_cols = list(
            sorted(c for c in merged.columns if c.endswith('-base')))
        suma = summary_report(merged, add_cols=add_cols)
        self.assertEqual(merged.shape[0], suma.shape[0])
        self.assertIn('N=10-base', suma.columns)
        outdf = os.path.join(temp, "merged.xlsx")
        suma.to_excel(outdf, index=False)


if __name__ == "__main__":
    unittest.main()
