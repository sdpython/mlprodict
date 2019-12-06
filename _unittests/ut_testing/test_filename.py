"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.filename_helper import (
    extract_information_from_filename,
    make_readable_title
)


class TestFilename(ExtTestCase):

    def test_extract(self):
        candidates = [
            'bench_DecisionTreeClassifier_default_b_cl_1_4_12_float_.py',
            'bench_DecisionTreeClassifier_default_b_cl_64_10_20_12_double_.py',
            'bench_DecisionTreeClassifier_default_b_cl_64_100_4_12_float_.py',
            'bench_AdaBoostClassifier_default_b_cl_1000_50_12_float__fct.svg',
            'bench_AdaBoostClassifier_default_m_cl_1_4_12_float__line.svg',
            'bench_LogisticRegression_liblinear_b_cl_solverliblinear_1_4_12_float_nozipmap_fct.svg',
        ]
        exps = [
            dict(model='DecisionTreeClassifier', scenario='default', problem='b_cl',
                 N=1, nf=4, opset=12, opt='float'),
            dict(model='DecisionTreeClassifier', scenario='default', problem='b_cl',
                 N=10, nf=20, opset=12, opt='double', double=True),
            dict(model='DecisionTreeClassifier', scenario='default', problem='b_cl',
                 N=100, nf=4, opset=12, opt='float', double=True),
            dict(model='AdaBoostClassifier', scenario='default', problem='b_cl',
                 N=1000, nf=50, opset=12, opt='float', profile='fct'),
            dict(model='AdaBoostClassifier', scenario='default', problem='m_cl',
                 N=1, nf=4, opset=12, opt='float', profile='line'),
            dict(model='LogisticRegression', scenario='liblinear', problem='b_cl',
                 N=1, nf=4, opset=12, opt='solverliblinear_float_nozipmap', profile='fct'),
        ]
        titles = [
            'DecisionTreeClassifier [b_cl] [default] N=1 nf=4 ops=12 [float]',
            'DecisionTreeClassifier [b_cl] [default] N=10 nf=20 ops=12 x64 [double]',
            'DecisionTreeClassifier [b_cl] [default] N=100 nf=4 ops=12 x64 [float]',
            'AdaBoostClassifier [b_cl] [default] N=1000 nf=50 ops=12 [float]',
            'AdaBoostClassifier [m_cl] [default] N=1 nf=4 ops=12 [float]',
            'LogisticRegression [b_cl] [liblinear] N=1 nf=4 ops=12 [solverliblinear_float_nozipmap]',
            'LogisticRegression [b_cl] [liblinear] N=1 nf=4 ops=12 [solverliblinear_float_nozipmap]',
        ]
        for exp, name, title in zip(exps, candidates, titles):
            d = extract_information_from_filename(name)
            self.assertEqual(exp, d)
            t = make_readable_title(d)
            self.assertEqual(title, t)


if __name__ == "__main__":
    unittest.main()
