"""
@brief      test log(time=4s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.pandashelper import df2rst
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import (
    enumerate_validated_operator_opsets, summary_report
)
from mlprodict.onnxrt.doc.doc_write_helper import (
    split_columns_subsets, build_key_split, filter_rows
)


class TestRtValidateOneVsRestClassifier(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_OneVsRestClassifier_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"OneVsRestClassifier"}, opset_min=9,
            opset_max=11, fLOG=myprint, benchmark=True,
            runtime='python', debug=debug,
            filter_exp=lambda m, p: True or 'm-cl' in p))
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[0])
        self.assertIn('onx_size', rows[-1])
        piv = summary_report(DataFrame(rows))
        self.assertGreater(piv.shape[0], 1)
        self.assertGreater(piv.shape[0], 2)
        common, subsets = split_columns_subsets(piv)
        rst = df2rst(piv, number_format=2,
                     replacements={'nan': '', 'ERR: 4convert': ''},
                     split_row=lambda index, dp=piv: build_key_split(
                         dp.loc[index, "name"], index),
                     split_col_common=common,
                     split_col_subsets=subsets,
                     filter_rows=filter_rows,
                     column_size={'problem': 25},
                     label_pattern=".. _lpy-{section}:")
        self.assertIn("opset9 | RT/SKL-N=1", rst)


if __name__ == "__main__":
    unittest.main()
