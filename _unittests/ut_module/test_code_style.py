"""
@brief      test log(time=120s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import check_pep8, ExtTestCase


class TestCodeStyle(ExtTestCase):
    """Test style."""

    def test_style_src(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        src_ = os.path.normpath(os.path.join(thi, "..", "..", "mlprodict"))
        check_pep8(src_, fLOG=fLOG,
                   pylint_ignore=('C0103', 'C1801', 'R1705', 'W0108', 'W0613',
                                  'R1702', 'W0212', 'W0640', 'W0223', 'W0201',
                                  'W0622', 'C0123', 'W0107', 'R1728', 'C3001',
                                  'C0415', 'R1721', 'C0411', 'R1735', 'C2801',
                                  'C0208', 'C0325', 'W1514', 'C0209'),
                   skip=["R0401: Cyclic import",
                         '[E731] do not assign a lambda expression',
                         'gactions_num.py:',
                         'gactions.py'])

    def test_style_test(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        test = os.path.normpath(os.path.join(thi, "..", ))
        check_pep8(test, fLOG=fLOG, neg_pattern="temp_.*",
                   pylint_ignore=('C0103', 'C1801', 'R1705', 'W0108', 'W0613',
                                  'C0111', 'W0107', 'C0415', 'R1728', 'C0209',
                                  'R1721', 'C0302', 'C0411', 'R1735', 'W1514',
                                  'C0200', 'E1101', 'W0212', 'C3001', 'C2801',
                                  'R1720'),
                   skip=['if __name__ == "__main__":',
                         '[E731] do not assign a lambda expression'])


if __name__ == "__main__":
    unittest.main()
