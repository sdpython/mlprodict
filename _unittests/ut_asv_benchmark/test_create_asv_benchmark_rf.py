"""
@brief      test log(time=6s)
"""
import os
import unittest
import re
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.loghelper.run_cmd import run_script
from pyquickhelper.texthelper.version_helper import compare_module_version
import sklearn
from mlprodict.asv_benchmark import create_asv_benchmark
import mlprodict


class TestCreateAsvBenchmarkRF(ExtTestCase):

    def test_create_asv_benchmark_rf(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark_rf")
        created = create_asv_benchmark(
            location=temp, verbose=1, fLOG=fLOG,
            runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True, models={'RandomForestRegressor'})
        self.assertNotEmpty(created)

        reg = re.compile("class ([a-zA-Z0-9_]+)[(]")
        verif = False
        allnames = []
        for path, _, files in os.walk(os.path.join(temp, 'benches')):
            for zoo in files:
                if '__init__' in zoo:
                    continue
                fLOG(f"process '{zoo}'")
                fullname = os.path.join(path, zoo)
                with open(fullname, 'r', encoding='utf-8') as f:
                    content = f.read()
                names = reg.findall(content)
                name = names[0]
                content += f"\n\ncl = {name}()\ncl.setup_cache()\n"
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
                        f"Issue with '{fullname}'\n{err}")
                if (zoo.endswith("bench_RandomForestReg_default_b_reg_nest10.py") and
                        compare_module_version(sklearn.__version__, "0.21") >= 0):
                    if "random_state=42" not in content:
                        raise AssertionError(content)
                    else:
                        verif = True
        if not verif:
            raise AssertionError("Visited files\n{}".format(
                "\n".join(allnames)))


if __name__ == "__main__":
    unittest.main()
