"""
@brief      test log(time=20s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from mlprodict.onnxrt.validate import validate_operator_opsets, summary_report


class TestOnnxrtValidateOnnxRuntime(ExtTestCase):

    def test_validate_sklearn_operators_all_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_all_onnxruntime")
        if False:  # pylint: disable=W0125
            rows = validate_operator_opsets(
                verbose, models={"KMeans"}, opset_min=11, fLOG=fLOG,
                runtime='onnxruntime', debug=True)
        else:
            rows = validate_operator_opsets(verbose, debug=None, fLOG=fLOG,
                                            runtime='onnxruntime', dump_folder=temp)
        self.assertGreater(len(rows), 1)
        df = DataFrame(rows)
        self.assertGreater(df.shape[1], 1)
        fLOG("output results")
        df.to_csv(os.path.join(temp, "sklearn_opsets_report.csv"), index=False)
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)
        piv = summary_report(df)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)


if __name__ == "__main__":
    unittest.main()
