"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
from io import StringIO
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skl2onnx.common.data_types import Int64TensorType
from mlprodict.onnx_conv import (
    to_onnx, guess_schema_from_data, get_inputs_from_data)
from mlprodict.onnxrt import OnnxInference


class TestOnnxConvDataframe(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_pipeline_dataframe_case1(self):
        self.case_test_pipeline_dataframe(1)

    def test_pipeline_dataframe_case2(self):
        self.case_test_pipeline_dataframe(2)

    def test_pipeline_dataframe_case3(self):
        self.case_test_pipeline_dataframe(3)

    def test_pipeline_dataframe_case4(self):
        self.case_test_pipeline_dataframe(4)

    def test_pipeline_dataframe_case4_cat(self):
        self.case_test_pipeline_dataframe(4, cat=True)

    def case_test_pipeline_dataframe(self, case, cat=False):
        text = """
                fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,color
                7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
                7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
                7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
                11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,white
                """.replace("                ", "")
        X_train = pandas.read_csv(StringIO(text))
        for c in X_train.columns:
            if c != 'color':
                X_train[c] = X_train[c].astype(  # pylint: disable=E1136,E1137
                    numpy.float32)
        numeric_features = [c for c in X_train if c != 'color']

        if case == 1:
            pipe = Pipeline([
                ("prep", ColumnTransformer([
                    ("color", Pipeline([
                        ('one', OneHotEncoder(sparse=False)),
                    ]), ['color']),
                    ("others", "passthrough", numeric_features)
                ])),
            ])
        elif case == 2:
            pipe = Pipeline([
                ("prep", ColumnTransformer([
                    ("color", Pipeline([
                        ('one', OneHotEncoder(sparse=False)),
                        ('select', ColumnTransformer(
                            [('sel1', "passthrough", [0])]))
                    ]), ['color']),
                    ("others", "passthrough", numeric_features)
                ])),
            ])
        elif case == 3:
            pipe = Pipeline([
                ("prep", ColumnTransformer([
                    ("colorord", OrdinalEncoder(), ['color']),
                    ("others", "passthrough", numeric_features)
                ])),
            ])
        elif case == 4:
            pipe = Pipeline([
                ("prep", ColumnTransformer([
                    ("color", Pipeline([
                        ('one', OneHotEncoder(sparse=False)),
                        ('select', ColumnTransformer(
                            [('sel1', "passthrough", [0])]))
                    ]), ['color']),
                    ("colorord", OrdinalEncoder(), ['color']),
                    ("others", "passthrough", numeric_features)
                ])),
            ])
        else:
            raise NotImplementedError()

        if cat:
            X_train['color'] = X_train['color'].astype(  # pylint: disable=E1136,E1137
                'category')
        schema = guess_schema_from_data(X_train)
        if isinstance(schema[-1][-1], Int64TensorType):
            raise AssertionError(
                "Issue with type of last column %r: %r." % (
                    schema[-1], X_train.dtypes[-1]))  # pylint: disable=E1101

        pipe.fit(X_train)
        model_onnx = to_onnx(pipe, X_train)
        try:
            oinf = OnnxInference(model_onnx)
        except RuntimeError as e:
            raise RuntimeError("Fails for case={}\n{}".format(
                case, e)) from e

        pred = pipe.transform(X_train)
        inputs = get_inputs_from_data(X_train)
        onxp = oinf.run(inputs)
        got = onxp['transformed_column']
        self.assertEqualArray(pred, got)


if __name__ == "__main__":
    unittest.main()
