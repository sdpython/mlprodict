"""
@brief      test log(time=15s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.asv_benchmark import create_asv_benchmark
from mlprodict.asv_benchmark._create_asv_helper import _format_dict
import mlprodict


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
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_flat")
        created = create_asv_benchmark(
            location=temp, models={'LogisticRegression', 'LinearRegression'},
            verbose=5, fLOG=fLOG, flat=True, execute=True)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'bench_LogReg_liblinear_b_cl_solverliblinear_onnx.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogReg_liblinear_b_cl_solverliblinear_onnx_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        if 'LogisticRegression' in content:
            if ("from sklearn.linear_model.logistic import LogisticRegression" not in content and
                    "from sklearn.linear_model import LogisticRegression" not in content):
                raise AssertionError(
                    "Unable to find 'import LogisticRegression in \n{}".format(content))
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
            'bench_LogReg_liblinear_b_cl_solverliblinear_onnx.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogReg_liblinear_b_cl_solverliblinear_onnx_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        if 'LogisticRegression' in content:
            if ("from sklearn.linear_model.logistic import LogisticRegression" not in content and
                    "from sklearn.linear_model import LogisticRegression" not in content):
                raise AssertionError(
                    "Unable to find 'import LogisticRegression in \n{}".format(content))
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
            'bench_LogReg_liblinear_b_cl_solverliblinear.py')
        self.assertExists(name)

        name = os.path.join(
            temp, 'benches', 'naive_bayes', 'BernoulliNB',
            'bench_BernoulliNB_default_b_cl.py')
        self.assertExists(name)

        name = os.path.join(
            temp, 'benches', '_externals', 'XGBRegressor',
            'bench_XGBReg_default_b_reg_nest100.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("from xgboost import XGBRegressor", content)

        name = os.path.join(
            temp, 'benches', '_externals', 'LGBMRegressor',
            'bench_LGBMReg_default_b_reg_nest100.py')
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
        self.assertIn("class VotingClas_", content)
        if 'LogisticRegression' in content:
            if ("from sklearn.linear_model.logistic import LogisticRegression" not in content and
                    "from sklearn.linear_model import LogisticRegression" not in content):
                raise AssertionError(
                    "Unable to find 'import LogisticRegression in \n{}".format(content))
        if 'VotingClassifier' in content:
            if ("from sklearn.ensemble.voting import VotingClassifier" not in content and
                    "from sklearn.ensemble import VotingClassifier" not in content):
                raise AssertionError(
                    "Unable to find 'import LogisticRegression in \n{}".format(content))

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
        self.assertIn("class CalibratedClasCV_", content)
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

        full_name = os.path.join(
            temp, "benches", "neighbors", "KNeighborsRegressor",
            "bench_KNNReg_default_k3_b_reg_algorithmbrute_n_neighbors3.py")
        self.assertExists(full_name)
        with open(full_name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class KNNReg_", content)
        self.assertIn("['cdist'],", content)

    def test_create_asv_benchmark_gpr(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_gpr")
        created = create_asv_benchmark(
            location=temp, models={'GaussianProcessRegressor'},
            verbose=5, fLOG=fLOG, flat=False, execute=True)
        self.assertGreater(len(created), 2)


if __name__ == "__main__":
    unittest.main()
