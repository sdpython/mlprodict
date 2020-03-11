"""
@brief      test log(time=2s)
"""
import unittest
import os
from pandas import read_csv
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate.validate_summary import (
    merge_benchmark, summary_report)


class TestBenchmarkTools(ExtTestCase):

    def common_test_benchmark_merge(self, f1, f2):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data")
        df1 = read_csv(os.path.join(data, f1))
        df2 = read_csv(os.path.join(data, f2))
        self.assertRaise(lambda: merge_benchmark({'1.1.2-': df1, 'master-': df2},
                                                 column='rtu'),
                         ValueError)
        merged = merge_benchmark({'1.1.2-': df1, 'master-': df2},
                                 baseline="1.1.2-onnxruntime1")
        self.assertIn('time-ratio-N=10-base', merged.columns)
        rt = merged['runtime']
        values = set(rt)
        self.assertEqual({'master-python_compiled',
                          'master-onnxruntime1',
                          '1.1.2-onnxruntime1',
                          '1.1.2-python_compiled'},
                         values)
        add_cols = list(
            sorted(c for c in merged.columns if c.endswith('-base')))
        suma = summary_report(merged, add_cols=add_cols)
        self.assertIn('N=10-base', suma.columns)
        suma.to_excel(f1 + ".xlsx", index=False)

    def test_benchmark_merge(self):
        self.common_test_benchmark_merge("data.csv", "data2.csv")


if __name__ == "__main__":
    unittest.main()
