"""
@brief      test log(time=3s)
"""
import os
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.texthelper.version_helper import compare_module_version
import sklearn
from mlprodict.asv_benchmark import create_asv_benchmark
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
import mlprodict


class TestCreateAsvBenchmarkPySpy(ExtTestCase):

    def test_create_asv_benchmark_pyspy(self):
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_pyspy")
        created = create_asv_benchmark(
            location=temp, verbose=0,
            runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True,
            models={'DecisionTreeClassifier'},
            add_pyspy=True)
        self.assertNotEmpty(created)

        ops = get_opset_number_from_onnx()
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
                if (zoo.endswith("bench_DecisionTreeClas_default_b_cl_1_4_%d_float_nozipmap.py" % ops) and
                        compare_module_version(sklearn.__version__, "0.21") >= 0):
                    if "setup_profile" not in content:
                        raise AssertionError(content)
                    verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))

    def test_create_asv_benchmark_pyspy_knn(self):
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_pyspy_knn")
        created = create_asv_benchmark(
            location=temp, verbose=0,
            runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True,
            models={'KNeighborsClassifier'},
            add_pyspy=True)
        self.assertNotEmpty(created)

        verif = False
        target_opset = get_opset_number_from_onnx()
        allnames = []
        for path, _, files in os.walk(os.path.join(temp, 'pyspy')):
            for zoo in files:
                if '__init__' in zoo:
                    continue
                allnames.append(zoo)
                fullname = os.path.join(path, zoo)
                with open(fullname, 'r', encoding='utf-8') as f:
                    content = f.read()
                if (zoo.endswith(
                        "bench_KNNClas_default_k3_b_cl_64_algorithmbrute_n_neighbors3"
                        "_10000_20_%d_double_optcdist-zm0.py" % target_opset) and
                        compare_module_version(sklearn.__version__, "0.21") >= 0):
                    if "setup_profile" not in content:
                        raise AssertionError(content)
                    verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))

    def test_create_asv_benchmark_pyspy_compiled(self):
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_pyspy_compiled")
        created = create_asv_benchmark(
            location=temp, verbose=0,
            runtime=('python', 'python_compiled'),
            exc=False, execute=True,
            models={'AdaBoostRegressor'},
            add_pyspy=True)
        self.assertNotEmpty(created)

        ops = get_opset_number_from_onnx()
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
                if (zoo.endswith("bench_AdaBoostReg_default_b_reg_nest10_1_4_%d_float_.py" % ops) and
                        compare_module_version(sklearn.__version__, "0.21") >= 0):
                    if "setup_profile" not in content:
                        raise AssertionError(content)
                    verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))


if __name__ == "__main__":
    unittest.main()
