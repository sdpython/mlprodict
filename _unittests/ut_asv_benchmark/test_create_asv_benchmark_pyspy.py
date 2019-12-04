"""
@brief      test log(time=3s)
"""
import os
import unittest
import re
from onnx.defs import onnx_opset_version
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.texthelper.version_helper import compare_module_version
import sklearn
from mlprodict.asv_benchmark import create_asv_benchmark
import mlprodict


class TestCreateAsvBenchmarkPySpy(ExtTestCase):

    def test_create_asv_benchmark_pyspy(self):
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_pyspy")
        created = create_asv_benchmark(
            location=temp, verbose=0,
            opset_min=onnx_opset_version(),
            runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True,
            models={'DecisionTreeClassifier'},
            add_pyspy=True)
        self.assertNotEmpty(created)

        reg = re.compile("class ([a-zA-Z0-9_]+)[(]")
        ops = onnx_opset_version()
        verif = False
        allnames = []
        for path, _, files in os.walk(os.path.join(temp, 'pyspy')):
            for zoo in files:
                if '__init__' in zoo:
                    continue
                allnames.append(zoo)
                fullname = os.path.join(path, zoo)
                with open(fullname, 'r', encoding='utf-8') as f:
                    content = f.read()
                if (zoo.endswith("bench_DecisionTreeClassifier_default_b_cl_1_4_%d_float_.py" % ops) and
                        compare_module_version(sklearn.__version__, "0.21") >= 0):
                    if "setup_profile" not in content:
                        raise AssertionError(content)
                    verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))


if __name__ == "__main__":
    unittest.main()
