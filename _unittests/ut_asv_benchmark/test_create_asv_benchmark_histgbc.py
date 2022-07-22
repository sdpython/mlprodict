"""
@brief      test log(time=16s)
"""
import os
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.texthelper.version_helper import compare_module_version
import sklearn
from mlprodict.asv_benchmark import create_asv_benchmark
import mlprodict


class TestCreateAsvBenchmarkHistGBC(ExtTestCase):

    def test_create_asv_benchmark_hist_gbc(self):
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_hist_gbc")
        created = create_asv_benchmark(
            location=temp, verbose=0,
            runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True,
            models={'HistGradientBoostingClassifier'})
        self.assertNotEmpty(created)

        verif = False
        allnames = []
        for path, _, files in os.walk(os.path.join(temp, 'benches')):
            for zoo in files:
                if '__init__' in zoo:
                    continue
                fullname = os.path.join(path, zoo)
                if "_hist_gradient_boosting" in fullname:
                    raise AssertionError(fullname)
                with open(fullname, 'r', encoding='utf-8') as f:
                    content = f.read()
                if (zoo.endswith("bench_HGBClas_default_b_cl_mxit100.py") and
                        compare_module_version(sklearn.__version__, "0.21") >= 0):
                    if "random_state=42" not in content:
                        raise AssertionError(content)
                    if "from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import" not in content:
                        raise AssertionError(content)
                    if "par_full_test_name = 'bench" not in content:
                        raise AssertionError(content)
                    verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))


if __name__ == "__main__":
    unittest.main()
