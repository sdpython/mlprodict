"""
@brief      test log(time=2s)
"""
import os
import json
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.filehelper.compression_helper import unzip_files
from mlprodict.asv_benchmark import export_asv_json, create_asv_benchmark
from mlprodict.asv_benchmark.asv_exports import _figures2dict


class TestAsvJsonText(ExtTestCase):

    data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

    def test_update_obs(self):
        values = list(range(0, 18))
        coor = [["'skl'", "'pyrt'", "'ort'"], [
            '1', '100', '10000'], ['4', '20']]
        res = _figures2dict(values, coor)
        exp = {'M-skl-1-4': 0, 'M-skl-1-20': 1, 'M-skl-100-4': 2, 'M-skl-100-20': 3,
               'M-skl-10000-4': 4, 'M-skl-10000-20': 5, 'M-pyrt-1-4': 6, 'M-pyrt-1-20': 7,
               'M-pyrt-100-4': 8, 'M-pyrt-100-20': 9, 'M-pyrt-10000-4': 10,
               'M-pyrt-10000-20': 11, 'M-ort-1-4': 12, 'M-ort-1-20': 13, 'M-ort-100-4': 14,
               'M-ort-100-20': 15, 'M-ort-10000-4': 16, 'M-ort-10000-20': 17}
        self.assertEqual(res, exp)
        res = _figures2dict(values, coor, baseline='skl')
        self.assertIn('R-ort-10000-20', res)

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
        exp = export_asv_json(data, baseline="skl")
        self.assertIsInstance(exp, list)
        self.assertTrue(all(map(lambda x: isinstance(x, dict), exp)))
        cc = 0
        for e in exp:
            ms = [k for k in e if k.startswith("M-")]
            rs = [k for k in e if k.startswith("R-")]
            if len(ms) > 0 and len(rs) > 0:
                cc += 1
        if cc == 0:
            raise AssertionError("No rs")

    def test_unzip_and_convert2(self):
        file_zip = os.path.join(TestAsvJsonText.data, 'results2.zip')
        temp = get_temp_folder(__file__, 'temp_unzip_and_convert2')
        unzip_files(file_zip, temp)
        data = os.path.join(temp, 'results')
        exp = export_asv_json(data, baseline="skl")
        self.assertIsInstance(exp, list)
        self.assertTrue(all(map(lambda x: isinstance(x, dict), exp)))
        cc = 0
        for e in exp:
            ms = [k for k in e if k.startswith("M-")]
            rs = [k for k in e if k.startswith("R-")]
            if len(ms) > 0 and len(rs) > 0:
                cc += 1
        if cc == 0:
            raise AssertionError("No rs")
        df = export_asv_json(data, baseline="skl", as_df=True)
        df.to_excel(os.path.join(temp, "res.xlsx"))

    def test_unzip_and_convert_metadata(self):
        file_zip = os.path.join(TestAsvJsonText.data, 'results2.zip')
        temp = get_temp_folder(__file__, 'temp_unzip_and_convert_metadata')
        create_asv_benchmark(
            location=temp, models={'LogisticRegression', 'LinearRegression'})
        unzip_files(file_zip, temp)
        data = os.path.join(temp, 'results')
        conf = os.path.join(temp, 'asv.conf.json')
        exp = export_asv_json(data, baseline="skl", conf=conf)
        par_problem = []
        par_scenario = []
        for row in exp:
            if 'par_problem' in row:
                par_problem.append(row['par_problem'])
            if 'par_scenario' in row:
                par_scenario.append(row['par_scenario'])
        s = set(par_scenario)
        self.assertEqual(s, {'default', 'liblinear'})
        s = set(par_problem)
        self.assertEqual(s, {'m-cl', '~m-reg-64', 'b-cl',
                             'm-reg', 'b-reg', '~b-cl-64',
                             '~b-reg-64'})
        out = os.path.join(temp, "df.xlsx")
        df = pandas.DataFrame(exp)
        df.to_excel(out)


if __name__ == "__main__":
    # TestAsvJsonText().test_unzip_and_convert_metadata()
    unittest.main()
