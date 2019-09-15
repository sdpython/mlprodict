"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
from io import StringIO
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxConvDataframe(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_pipeline_dataframe(self):
        text = """
                fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,color
                7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
                7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
                7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
                11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,red
                """.replace("                ", "")
        X_train = pandas.read_csv(StringIO(text))
        for c in X_train.columns:
            if c != 'color':
                X_train[c] = X_train[c].astype(numpy.float32)
        numeric_features = [c for c in X_train if c != 'color']

        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("color", Pipeline([
                    ('one', OneHotEncoder()),
                    ('select', ColumnTransformer(
                        [('sel1', 'passthrough', [0])]))
                ]), ['color']),
                ("others", "passthrough", numeric_features)
            ])),
        ])

        pipe.fit(X_train)
        model_onnx = to_onnx(pipe, X_train)
        oinf = OnnxInference(model_onnx)

        pred = pipe.transform(X_train)
        inputs = {c: X_train[c].values for c in X_train.columns}
        inputs = {c: v.reshape((v.shape[0], 1)) for c, v in inputs.items()}
        onxp = oinf.run(inputs)
        got = onxp['transformed_column']
        self.assertEqualArray(pred, got)


if __name__ == "__main__":
    unittest.main()
