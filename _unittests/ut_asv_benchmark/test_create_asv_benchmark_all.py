"""
@brief      test log(time=2s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.asv_benchmark import create_asv_benchmark


class TestCreateAsvBenchmarkAll(ExtTestCase):

    def test_create_asv_benchmark_all(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_all")
        created = create_asv_benchmark(
            location=temp, verbose=1, fLOG=fLOG,
            skip_models={
                'DictVectorizer', 'FeatureHasher',  # 'CountVectorizer'
            }, runtime=('scikit-learn', 'python', 'onnxruntime1'), exc=False)
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


if __name__ == "__main__":
    unittest.main()
