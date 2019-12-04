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


class TestCreateAsvBenchmarkAllTinySame(ExtTestCase):

    def test_create_asv_benchmark_tiny_same(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        self.assertNotEmpty(mlprodict)
        temp = get_temp_folder(
            __file__, "temp_create_asv_benchmark_all_tiny_same")
        created = create_asv_benchmark(
            location=temp, verbose=1, fLOG=fLOG,
            opset_min=onnx_opset_version(),
            skip_models={
                'DictVectorizer', 'FeatureHasher',  # 'CountVectorizer'
            }, runtime=('scikit-learn', 'python', 'onnxruntime1'),
            exc=False, execute=True, models={
                'SelectFromModel', 'NMF', 'LatentDirichletAllocation'
            }, env='same')
        self.assertNotEmpty(created)

        reg = re.compile("class ([a-zA-Z0-9_]+)[(]")
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
                if (zoo.endswith("bench_NMF_default_num_tr_pos.py") and
                        compare_module_version(sklearn.__version__, "0.22") >= 0):
                    if ("from sklearn.decomposition.nmf import NMF" not in content and
                            "from sklearn.decomposition import NMF" not in content):
                        raise AssertionError(
                            "Unable to find 'import NMF' in\n{}".format(content))


if __name__ == "__main__":
    unittest.main()
