"""
@brief      test log(time=6s)
"""
import os
import unittest
import re
from onnx.defs import onnx_opset_version
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.loghelper.run_cmd import run_script
from pyquickhelper.texthelper.version_helper import compare_module_version
import sklearn
from mlprodict.asv_benchmark import create_asv_benchmark
import mlprodict


class TestCreateAsvBenchmarkII(ExtTestCase):

    def test_create_asv_benchmark_iterative_imputer(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_ii")
        created = create_asv_benchmark(
            location=temp, verbose=1, fLOG=fLOG,
            opset_min=onnx_opset_version(),
            runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True, models={'IterativeImputer'})
        self.assertNotEmpty(created)

        reg = re.compile("class ([a-zA-Z0-9_]+)[(]")
        verif = False
        allnames = []
        for path, _, files in os.walk(os.path.join(temp, 'benches')):
            for zoo in files:
                if '__init__' in zoo:
                    continue
                fLOG("process '{}'".format(zoo))
                fullname = os.path.join(path, zoo)
                with open(fullname, 'r', encoding='utf-8') as f:
                    content = f.read()
                names = reg.findall(content)
                name = names[0]
                content += "\n\ncl = %s()\ncl.setup_cache()\n" % name
                allnames.append(fullname)
                with open(fullname, 'w', encoding='utf-8') as f:
                    f.write(content)
                __, err = run_script(fullname, wait=True)
                lines = [_ for _ in err.split('\n') if _ and _[0] != ' ']
                lines = [_ for _ in lines if "Warning" not in _]
                lines = [
                    _ for _ in lines if "No module named 'mlprodict'" not in _]
                lines = [_ for _ in lines if "Traceback " not in _]
                err = "\n".join(lines).strip(' \n\r')
                if len(err) > 0:
                    raise RuntimeError(
                        "Issue with '{}'\n{}".format(fullname, err))
                if (zoo.endswith("bench_IterativeImputer_default_num_tr.py") and
                        compare_module_version(sklearn.__version__, "0.22") >= 0):
                    if "random_state=42" not in content:
                        raise AssertionError(content)
                    else:
                        verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))


if __name__ == "__main__":
    unittest.main()
