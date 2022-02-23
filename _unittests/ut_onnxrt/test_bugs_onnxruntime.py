"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame, concat
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.data import load_audit
from mlprodict import __max_supported_opset__, get_ir_version


class TestBugsOnnxrtOnnxRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_gradient_boosting_regressor_pipeline(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        random_seed = 123
        df = load_audit()
        train, test = train_test_split(df, test_size=0.2,
                                       random_state=random_seed)
        target_feature = 'TARGET_Adjusted'
        y_train = train[target_feature]
        x_train = train.drop(target_feature, axis=1, inplace=False)
        y_test = test[target_feature]
        x_test = test.drop(target_feature, axis=1, inplace=False)
        cols = list(x_train.columns)
        numerical_cols = list(
            x_train._get_numeric_data().columns)  # pylint: disable=W0212
        categorical_cols = list(set(cols) - set(numerical_cols))

        n_trees = 50
        max_depth = 10
        predictor = Pipeline([
            ('prep', ColumnTransformer([
                            ('num_prep', StandardScaler(), numerical_cols),
                            ('cat_prep', OneHotEncoder(
                                handle_unknown='ignore'), categorical_cols)
            ])),

            ('model', GradientBoostingClassifier(
                learning_rate=0.01,
                random_state=random_seed,
                n_estimators=n_trees,
                max_depth=max_depth))
        ])

        predictor.fit(x_train, y_train)
        fLOG('accuracy: ' + str(predictor.score(x_test, y_test)))
        sklearn_predictions = DataFrame(
            predictor.predict(x_test), columns=['sklearn_prediction'])

        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                # also ints treated as floats otherwise onnx exception "all columns must be equal" is raised.
                if v in ('int64', 'float64'):
                    t = FloatTensorType([None, 1])
                else:
                    t = StringTensorType([None, 1])
                inputs.append((k, t))
            return inputs

        model_name = 'gbt_audit'
        inputs = convert_dataframe_schema(x_train)
        try:
            model_onnx = convert_sklearn(predictor, model_name, inputs)
        except Exception as e:
            raise AssertionError(
                "Unable to convert model %r (version=%r)." % (
                    predictor, skl2onnx_version)) from e

        data = {col[0]: x_test[col[0]].values.reshape(x_test.shape[0], 1)
                for col in inputs}
        for col in numerical_cols:
            data[col] = data[col].astype(numpy.float32)

        for runtime in ['python', 'python_compiled',
                        'onnxruntime1', 'onnxruntime2']:
            if runtime == 'onnxruntime2':
                # Type for text column are guessed wrong
                # (Float instead of text).
                continue

            if 'onnxruntime' in runtime:
                model_onnx.ir_version = get_ir_version(__max_supported_opset__)
            sess = OnnxInference(model_onnx, runtime=runtime)

            onnx_predictions = sess.run(data)
            onnx_predictions = DataFrame(
                {'onnx_prediction': onnx_predictions['output_label']})

            fLOG('Model accuracy in SKlearn = ' +
                 str(accuracy_score(y_test, sklearn_predictions.values)))
            fLOG('Model accuracy in ONNX = ' +
                 str(accuracy_score(y_test, onnx_predictions)))
            fLOG()
            fLOG('predicted class distribution from SKLearn')
            fLOG(sklearn_predictions['sklearn_prediction'].value_counts())
            fLOG()
            fLOG('predicted class distribution from ONNX')
            fLOG(onnx_predictions['onnx_prediction'].value_counts())
            fLOG()

            df = concat([sklearn_predictions, onnx_predictions], axis=1)
            df["diff"] = df["sklearn_prediction"] - df["onnx_prediction"]
            df["diff_abs"] = numpy.abs(df["diff"])
            total = df.sum()
            sum_diff = total["diff_abs"]
            if sum_diff != 0:
                raise AssertionError("Runtime: '{}', discrepencies: sum_diff={}"
                                     "".format(runtime, sum_diff))


if __name__ == "__main__":
    unittest.main()
