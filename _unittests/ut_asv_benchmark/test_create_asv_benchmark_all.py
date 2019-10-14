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


class TestCreateAsvBenchmarkAll(ExtTestCase):

    def test_create_asv_benchmark_all(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
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
            'bench_LogisticRegression_liblinear_solverliblinear_onnx.py')
        self.assertExists(name)
        with open(name, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "class LogisticRegression_liblinear_solverliblinear_onnx_benchClassifier(", content)
        self.assertIn("solver='liblinear'", content)
        self.assertIn("return onnx_optimisations(onx)", content)
        self.assertIn(
            "from sklearn.linear_model.logistic import LogisticRegression", content)

        if __name__ == "__main__":
            fLOG("[] checks setup_cache")
            reg = re.compile("class ([a-zA-Z0-9_]+)[(]")
            for path, _, files in os.walk(os.path.join(temp, 'benches')):
                for zoo in files:
                    if 'Stacking' not in zoo:
                        continue
                    if '__init__' in zoo:
                        continue
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


if __name__ == "__main__":
    unittest.main()
