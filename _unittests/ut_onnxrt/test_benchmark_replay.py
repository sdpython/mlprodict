"""
@brief      test log(time=2s)
"""
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.onnxrt.validate import (
    enumerate_benchmark_replay, enumerate_validated_operator_opsets
)


class TestBenchmarkReplay(ExtTestCase):

    def test_benchmark_replay(self):
        temp = get_temp_folder(__file__, "temp_bct")
        self.assertRaise(lambda: list(
            enumerate_benchmark_replay(temp, runtime='python')),
            FileNotFoundError)
        res = list(enumerate_validated_operator_opsets(
            0, fLOG=None, models={"LogisticRegression"}, opset_min=14,  # opset=13, 14, ...
            opset_max=14, benchmark=False, store_models=True, dump_all=True,
            dump_folder=temp, filter_exp=lambda m, p: (
                "64" not in p and "b-cl" in p and "dec" not in p)))
        self.assertNotEmpty(res)
        res.clear()
        rows = list(enumerate_benchmark_replay(
            temp, runtime='python', verbose=0))
        df = pandas.DataFrame(rows)
        self.assertEqual(df.shape, (4, 35))
        self.assertIn('1000-skl-details', df.columns)
        self.assertIn('1000-skl', df.columns)

    def test_benchmark_replay_onnxruntime(self):
        temp = get_temp_folder(__file__, "temp_bct_ort")
        self.assertRaise(lambda: list(
            enumerate_benchmark_replay(temp, runtime='onnxruntime1')),
            FileNotFoundError)
        res = list(enumerate_validated_operator_opsets(
            0, fLOG=None, models={"LogisticRegression"}, opset_min=11,
            opset_max=11, benchmark=False, store_models=True, dump_all=True,
            dump_folder=temp, filter_exp=lambda m, p: (
                "64" not in p and "b-cl" in p and "dec" not in p)))
        self.assertNotEmpty(res)
        res.clear()
        rows = list(enumerate_benchmark_replay(
            temp, runtime='onnxruntime', verbose=0))
        df = pandas.DataFrame(rows)
        self.assertEqual(df.shape, (4, 35))
        self.assertIn('1000-skl-details', df.columns)
        self.assertIn('1000-skl', df.columns)


if __name__ == "__main__":
    unittest.main()
