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
            verbose=5, fLOG=fLOG, flat=True, execute=True)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'bench_LogisticRegression_liblinear_solverliblinear_onnx.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogisticRegression_liblinear_solverliblinear_onnx_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        if 'LogisticRegression' in content:
            self.assertIn(
                "from sklearn.linear_model.logistic import LogisticRegression", content)
        self.assertIn("par_optimonnx = True", content)
        self.assertIn("par_scenario = ", content)
        self.assertIn("par_problem = ", content)

    def test_create_asv_benchmark_noflat(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_noflat")
        created = create_asv_benchmark(
            location=temp, models={'LogisticRegression', 'LinearRegression'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'linear_model', 'LogisticRegression',
            'bench_LogisticRegression_liblinear_solverliblinear_onnx.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogisticRegression_liblinear_solverliblinear_onnx_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        if 'LogisticRegression' in content:
            self.assertIn(
                "from sklearn.linear_model.logistic import LogisticRegression", content)
        self.assertIn("par_optimonnx = True", content)

    def test_create_asv_benchmark_noflat_ext(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_noflat__ext")
        created = create_asv_benchmark(
            location=temp, models={
                'LogisticRegression', 'BernoulliNB', 'XGBRegressor', 'LGBMRegressor'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'linear_model', 'LogisticRegression',
            'bench_LogisticRegression_liblinear_solverliblinear.py')
        self.assertExists(name)

        name = os.path.join(
            temp, 'benches', 'naive_bayes', 'BernoulliNB',
            'bench_BernoulliNB_default.py')
        self.assertExists(name)

        name = os.path.join(
            temp, 'benches', '_externals', 'XGBRegressor',
            'bench_XGBRegressor_default.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("from xgboost import XGBRegressor", content)

        name = os.path.join(
            temp, 'benches', '_externals', 'LGBMRegressor',
            'bench_LGBMRegressor_default_n_estimators5.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("from lightgbm import LGBMRegressor", content)

    def test_create_asv_benchmark_noflat_vc(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_noflat_vc")
        created = create_asv_benchmark(
            location=temp, models={'VotingClassifier'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)

        names = os.listdir(os.path.join(
            temp, 'benches', 'ensemble', 'VotingClassifier'))
        names = [name for name in names if '__init__' not in name]
        full_name = os.path.join(
            temp, 'benches', 'ensemble', 'VotingClassifier', names[0])
        self.assertExists(full_name)
        with open(full_name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class VotingClassifier_", content)
        if 'LogisticRegression' in content:
            self.assertIn("LogisticRegression(", content)
        self.assertIn(
            "from sklearn.ensemble.voting import VotingClassifier", content)
        if "LogisticRegression" in content:
            self.assertIn(
                "from sklearn.linear_model import LogisticRegression", content)

    def test_create_asv_benchmark_text(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_text")
        created = create_asv_benchmark(
            location=temp, models={'HashingVectorizer'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)

        names = os.listdir(os.path.join(
            temp, 'benches', 'feature_extraction', 'HashingVectorizer'))
        names = [name for name in names if '__init__' not in name]
        full_name = os.path.join(
            temp, 'benches', 'feature_extraction', 'HashingVectorizer', names[0])
        self.assertExists(full_name)
        with open(full_name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class HashingVectorizer_", content)
        self.assertIn(
            "from sklearn.feature_extraction.text import HashingVectorizer", content)

    def test_create_asv_benchmark_calibrated(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_calibrated")
        created = create_asv_benchmark(
            location=temp, models={'CalibratedClassifierCV'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)

        names = os.listdir(os.path.join(
            temp, 'benches', 'calibration', 'CalibratedClassifierCV'))
        names = [name for name in names if '__init__' not in name]
        full_name = os.path.join(
            temp, 'benches', 'calibration', 'CalibratedClassifierCV', names[0])
        self.assertExists(full_name)
        with open(full_name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class CalibratedClassifierCV_", content)
        self.assertIn(
            "from sklearn.calibration import CalibratedClassifierCV", content)
        if 'SGDclassifier' in content:
            self.assertIn(
                "from sklearn.linear_model import SGDClassifier", content)

    def test_create_asv_benchmark_knnr(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_knnr")
        created = create_asv_benchmark(
            location=temp, models={'KNeighborsRegressor'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)
        self.assertNotExists(os.path.join(
            temp, "benches", "neighbors", "KNeighborsRegressor",
            "bench_KNeighborsRegressor_cdist_algorithmbrute_cdist.py"))
        self.assertNotExists(os.path.join(
            temp, "benches", "neighbors", "KNeighborsRegressor",
            "bench_KNeighborsRegressor_cdist_algorithmbrute.py"))

        full_name = os.path.join(
            temp, "benches", "neighbors", "KNeighborsRegressor",
            "bench_KNeighborsRegressor_default_algorithmbrute.py")
        self.assertExists(full_name)
        with open(full_name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class KNeighborsRegressor_", content)
        self.assertIn("[{}, 'cdist'],", content)

    def test_create_asv_benchmark_gpr(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_gpr")
        created = create_asv_benchmark(
            location=temp, models={'GaussianProcessRegressor'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)


if __name__ == "__main__":
    # TestCreateAsvBenchmark().test_create_asv_benchmark_gpr()
    unittest.main()
