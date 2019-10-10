"""
@brief      test log(time=2s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.asv_benchmark import create_asv_benchmark
from mlprodict.asv_benchmark.create_asv import _format_dict


class TestCreateAsvBenchmark(ExtTestCase):

    def test_format_dict(self):
        di = dict(a=0, b='t', c=[0, 1], d=['r', [5]], e={'t': 'f'})
        for i, k in enumerate('azer'):
            di['azer%d' % i] = k + 'azer' * i
        st = _format_dict(di, indent=4)
        self.assertIn("a=0, azer0='a', azer1='zazer', azer2='eazerazer'", st)
        self.assertIn(
            "azer3='razerazerazer', b='t', c=[0, 1], d=['r', [5]]", st)
        self.assertIn(", e={'t': 'f'}", st)

    def test_create_asv_benchmark_flat(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_flat")
        created = create_asv_benchmark(
            location=temp, models={'LogisticRegression', 'LinearRegression'},
            verbose=5, fLOG=fLOG, flat=True)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'bench_LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        self.assertIn(
            "from sklearn.linear_model import LogisticRegression", content)

    def test_create_asv_benchmark_noflat(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_noflat")
        created = create_asv_benchmark(
            location=temp, models={'LogisticRegression', 'LinearRegression'},
            verbose=5, fLOG=fLOG, flat=False)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'linear_model', 'LogisticRegression',
            'bench_LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        self.assertIn(
            "from sklearn.linear_model import LogisticRegression", content)

    def test_create_asv_benchmark_noflat_ext(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_noflat__ext")
        created = create_asv_benchmark(
            location=temp, models={
                'LogisticRegression', 'BernoulliNB', 'XGBRegressor'},
            verbose=5, fLOG=fLOG, flat=False)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'linear_model', 'LogisticRegression',
            'bench_LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10.py')
        self.assertExists(name)

        name = os.path.join(
            temp, 'benches', 'naive_bayes', 'BernoulliNB',
            'bench_BernoulliNB_b_cl_default_10.py')
        self.assertExists(name)

        name = os.path.join(
            temp, 'benches', '_externals', 'XGBRegressor',
            'bench_XGBRegressor_b_reg_64_default_10.py')
        self.assertExists(name)


if __name__ == "__main__":
    unittest.main()
