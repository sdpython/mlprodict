"""
@brief      test log(time=14s)
"""
import os
import unittest
from logging import getLogger
from textwrap import dedent
from sklearn.linear_model import LinearRegression
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, sklearn_operators
from mlprodict.onnxrt.validate import sklearn__all__
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.doc_write_helper import enumerate_visual_onnx_representation_into_rst


class TestOnnxrtValidateDocumentation(ExtTestCase):

    def test_validate_sklearn_store_models(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        rows = list(enumerate_validated_operator_opsets(
            verbose=0, models={"LinearRegression"}, opset_min=10,
            store_models=True, fLOG=fLOG))

        self.assertNotEmpty(rows)
        self.assertIn('MODEL', rows[0])
        self.assertIn('ONNX', rows[0])
        self.assertIsInstance(rows[0]['MODEL'], LinearRegression)
        oinf = OnnxInference(rows[0]['ONNX'])
        dot = oinf.to_dot()
        self.assertIn('LinearRegressor', dot)

    def test_write_documentation_converters(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        this = os.path.abspath(os.path.dirname(__file__))
        whe = os.path.join(this, "..", "..", "_doc",
                           "sphinxdoc", "source", "skl_converters")
        index = os.path.join(whe, "index.rst")
        subs = []
        for sub in sorted(sklearn__all__):
            models = sklearn_operators(sub)
            if len(models) > 0:
                rows = []
                for row in enumerate_visual_onnx_representation_into_rst(sub):
                    self.assertIn("digraph", row)
                    rows.append(row)
                if len(rows) == 0:
                    continue
                rows = [".. _l-skl2onnx-%s:" % sub, "", "=" * len(sub),
                        sub, "=" * len(sub), "", ".. contents::",
                        "    :local:", ""] + rows
                rows.append('')
                dest = os.path.join(whe, "skl2onnx_%s.rst" % sub)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write("\n".join(rows))
                subs.append(sub)
                fLOG("wrote '{}' - {} scenarios.".format(sub, len(models)))

        self.assertGreater(len(subs), 2)

        with open(index, "w", encoding="utf-8") as f:
            f.write(dedent("""
            Visual Representation of scikit-learn models
            ============================================

            :epkg:`sklearn-onnx` converts many models from
            :epkg:`scikit-learn` into :epkg:`ONNX`. Every of
            them is a graph made of :epkg:`ONNX` mathematical functions
            (see :ref:`l-onnx-runtime-operators`,
            :epkg:`ONNX Operators`, :epkg:`ONNX ML Operators`).
            The following sections display a visual representation
            of each converted model. Every graph
            represents one ONNX graphs obtained after a model
            is fitted. The structure may change is the model is trained
            again.

            .. toctree::
                :maxdepth: 1

            """))
            for sub in subs:
                f.write("    skl2onnx_%s\n" % sub)
            f.write('')


if __name__ == "__main__":
    unittest.main()
