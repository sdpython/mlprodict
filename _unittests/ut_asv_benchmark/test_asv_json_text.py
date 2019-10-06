"""
@brief      test log(time=2s)
"""
import os
import json
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.filehelper.compression_helper import unzip_files
from mlprodict.asv_benchmark import export_asv_json


class TestAsvJsonText(ExtTestCase):

    data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

    def test_asv_json_simplify(self):
        temp = get_temp_folder(__file__, 'temp_asv_json_simplify')
        filenames = [
            os.path.join(TestAsvJsonText.data, 'benchmarks.json'),
            os.path.join(r"C:\temp\results", 'benchmarks.json'),
        ]
        for i, filename in enumerate(filenames):
            if not os.path.exists(filename):
                continue
            with open(filename, 'r', encoding='utf-8') as f:
                content = json.load(f)

            for _, v in content.items():
                if isinstance(v, dict) and 'code' in v:
                    v['code'] = ""

            res = os.path.join(temp, 'benchmarks_%d.json' % i)
            with open(res, 'w', encoding='utf-8') as f:
                json.dump(content, f)

            with open(res, "r", encoding='utf-8') as f:
                content = f.read()
            self.assertIn('{', content)

    def test_unzip_and_convert(self):
        file_zip = os.path.join(TestAsvJsonText.data, 'results.zip')
        temp = get_temp_folder(__file__, 'temp_unzip_and_convert')
        unzip_files(file_zip, temp)
        data = os.path.join(temp, 'results')
        exp = export_asv_json(data)
        self.assertIsInstance(exp, dict)


if __name__ == "__main__":
    unittest.main()
