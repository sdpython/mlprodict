"""
@brief      test log(time=20s)
"""
import os
import unittest
import re
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.loghelper.run_cmd import run_script
from mlprodict.asv_benchmark import create_asv_benchmark
import mlprodict


class TestCreateAsvBenchmarkAll(ExtTestCase):

    def test_create_asv_benchmark_all(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_all")
        created = create_asv_benchmark(
            location=temp, verbose=1, fLOG=fLOG,
            skip_models={
                'DictVectorizer', 'FeatureHasher',  # 'CountVectorizer'
            }, runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True)
        self.assertGreater(len(created), 2)

        name = os.path.join(
            temp, 'benches', 'linear_model', 'LogisticRegression',
            'bench_LogisticRegression_liblinear_b_cl_solverliblinear_onnx.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogisticRegression_liblinear_b_cl_solverliblinear_onnx_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        try:
            self.assertIn(
                "from sklearn.linear_model._logistic import LogisticRegression", content)
        except AssertionError:
            self.assertIn(
                "from sklearn.linear_model.logistic import LogisticRegression", content)

        if __name__ == "__main__":
            fLOG("[] checks setup_cache")
            reg = re.compile("class ([a-zA-Z0-9_]+)[(]")
            checked = []
            folder = os.path.join(temp, 'benches')
            subsets_test = [
                'Stacking',
                'ovariance',
                'bench_LogisticRegression_liblinear',
                'Latent'
            ]
            for path, _, files in os.walk(folder):
                for zoo in files:
                    if '__init__' in zoo:
                        continue
                    if 'chain' in zoo.lower():
                        continue
                    if not any(map(lambda x,z=zoo: x in z, subsets_test)):
                        continue
                    checked.append(zoo)
                    fLOG("process '{}'".format(zoo))
                    fullname = os.path.join(path, zoo)
                    with open(fullname, 'r', encoding='utf-8') as f:
                        content = f.read()
                    names = reg.findall(content)
                    name = names[0]
                    content += "\n\ncl = %s()\ncl.setup_cache()\n" % name
                    with open(fullname, 'w', encoding='utf-8') as f:
                        f.write(content)
                    __, err = run_script(fullname, wait=True)
                    lines = [_ for _ in err.split('\n') if _ and _[0] != ' ']
                    lines = [_ for _ in lines if "Warning" not in _]
                    err = "\n".join(lines).strip(' \n\r')
                    if len(err) > 0:
                        raise RuntimeError(
                            "Issue with '{}'\n{}".format(fullname, err))
            if len(checked) == 0:
                raise AssertionError("Nothing found in '{}'.".format(folder))


if __name__ == "__main__":
    unittest.main()
