"""
@brief      test log(time=3s)
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
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
from mlprodict.onnxrt.doc.doc_write_helper import split_columns_subsets


class TestRtValidateKMeans(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_KMeans_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"KMeans"}, opset_min=11,
            opset_max=11, fLOG=myprint,
            runtime='python', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[-1])
        keys = set()
        for row in rows:
            keys.update(set(row))
        self.assertIn('onx_size', keys)
        piv = summary_report(DataFrame(rows))
        opset = [c for c in piv.columns if 'opset' in c]
        self.assertTrue('opset11' in opset or 'opset10' in opset)
        self.assertGreater(len(buffer), 1 if debug else 0)
        common, subsets = split_columns_subsets(piv)
        try:
            conv = df2rst(piv, split_col_common=common,  # pylint: disable=E1123
                          split_col_subsets=subsets)
            self.assertIn('| KMeans |', conv)
        except TypeError as e:
            if "got an unexpected keyword argument 'split_col_common'" in str(e):
                return
            raise e


if __name__ == "__main__":
    unittest.main()
