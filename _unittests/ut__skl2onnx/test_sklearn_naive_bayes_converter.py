"""
@brief      test tree node (time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.naive_bayes import (
    BernoulliNB, GaussianNB, MultinomialNB)
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, BooleanTensorType)
from mlprodict.testing.test_utils import (
    dump_data_and_model, fit_classification_model, TARGET_OPSET)


class TestNaiveBayesConverter(ExtTestCase):

    def test_model_multinomial_nb_binary_classification(self):
        model, X = fit_classification_model(
            MultinomialNB(), 2, pos_features=True)
        model_onnx = convert_sklearn(
            model, "multinomial naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32), model, model_onnx,
            basename="SklearnBinMultinomialNB-Dec4", verbose=False)

    def test_model_bernoulli_nb_binary_classification(self):
        model, X = fit_classification_model(
            BernoulliNB(), 2)
        model_onnx = convert_sklearn(
            model, "bernoulli naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, "SklearnBinBernoulliNB")

    def test_model_multinomial_nb_multiclass(self):
        model, X = fit_classification_model(
            MultinomialNB(), 5, pos_features=True)
        model_onnx = convert_sklearn(
            model, "multinomial naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclMultinomialNB-Dec4")

    def test_model_multinomial_nb_multiclass_params(self):
        model, X = fit_classification_model(
            MultinomialNB(alpha=0.5, fit_prior=False), 5, pos_features=True)
        model_onnx = convert_sklearn(
            model, "multinomial naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        pp = model.predict_proba(X)
        col = pp.shape[1]
        pps = numpy.sort(pp, axis=1)
        diff = pps[:, col - 1] - pps[:, col - 2]
        ind = diff >= 1e-4
        dump_data_and_model(
            X[ind], model, model_onnx,
            basename="SklearnMclMultinomialNBParams-Dec4")

    def test_model_bernoulli_nb_multiclass(self):
        model, X = fit_classification_model(
            BernoulliNB(), 4)
        model_onnx = convert_sklearn(
            model, "bernoulli naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclBernoulliNB")

    def test_model_bernoulli_nb_multiclass_params(self):
        model, X = fit_classification_model(
            BernoulliNB(alpha=0, binarize=1.0, fit_prior=False), 4)
        model_onnx = convert_sklearn(model, "bernoulli naive bayes",
                                     [("input", FloatTensorType([None, X.shape[1]]))],
                                     target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclBernoulliNBParams")

    def test_model_multinomial_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            MultinomialNB(), 2, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "multinomial naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinMultinomialNBInt-Dec4")

    def test_model_multinomial_nb_binary_classification_bool(self):
        model, X = fit_classification_model(
            MultinomialNB(), 2, is_bool=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "multinomial naive bayes",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinMultinomialNBBool-Dec4")

    def test_model_bernoulli_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            BernoulliNB(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model, "bernoulli naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinBernoulliNBInt")

    def test_model_bernoulli_nb_binary_classification_bool(self):
        model, X = fit_classification_model(
            BernoulliNB(), 2, is_bool=True)
        model_onnx = convert_sklearn(
            model, "bernoulli naive bayes",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinBernoulliNBBool")

    def test_model_multinomial_nb_multiclass_int(self):
        model, X = fit_classification_model(
            MultinomialNB(), 5, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "multinomial naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclMultinomialNBInt-Dec4")

    def test_model_bernoulli_nb_multiclass_int(self):
        model, X = fit_classification_model(
            BernoulliNB(), 4, is_int=True)
        model_onnx = convert_sklearn(
            model, "bernoulli naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclBernoulliNBInt-Dec4")

    def test_model_gaussian_nb_binary_classification(self):
        model, X = fit_classification_model(
            GaussianNB(), 2)
        model_onnx = convert_sklearn(
            model, "gaussian naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinGaussianNB")

    def test_model_gaussian_nb_multiclass(self):
        model, X = fit_classification_model(
            GaussianNB(), 4)
        model_onnx = convert_sklearn(
            model, "gaussian naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclGaussianNB")

    def test_model_gaussian_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            GaussianNB(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model, "gaussian naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinGaussianNBInt")

    def test_model_gaussian_nb_multiclass_int(self):
        model, X = fit_classification_model(
            GaussianNB(), 5, is_int=True)
        model_onnx = convert_sklearn(
            model, "gaussian naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclGaussianNBInt-Dec4")

    def test_model_gaussian_nb_multiclass_bool(self):
        model, X = fit_classification_model(
            GaussianNB(), 5, is_bool=True)
        model_onnx = convert_sklearn(
            model, "gaussian naive bayes",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclGaussianNBBool-Dec4")

    def test_model_complement_nb_binary_classification(self):
        model, X = fit_classification_model(
            ComplementNB(), 2, pos_features=True)
        model_onnx = convert_sklearn(
            model, "complement naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinComplementNB-Dec4")

    def test_model_complement_nb_multiclass(self):
        model, X = fit_classification_model(
            ComplementNB(), 4, pos_features=True)
        model_onnx = convert_sklearn(
            model, "complement naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclComplementNB-Dec4")

    def test_model_complement_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            ComplementNB(), 2, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "complement naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnBinComplementNBInt-Dec4")

    def test_model_complement_nb_multiclass_int(self):
        model, X = fit_classification_model(
            ComplementNB(), 5, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "complement naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclComplementNBInt-Dec4")

    def test_model_complement_nb_multiclass_bool(self):
        model, X = fit_classification_model(
            ComplementNB(), 5, is_bool=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "complement naive bayes",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnMclComplementNBBool-Dec4")

    def test_model_categorical_nb(self):
        model, X = fit_classification_model(
            CategoricalNB(), 3, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model, "categorical naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X[10:13], model, model_onnx, basename="SklearnCategoricalNB")

    def test_model_gaussian_nb_multi_class_nocl(self):
        model, X = fit_classification_model(
            GaussianNB(),
            2, label_string=True)
        model_onnx = convert_sklearn(
            model, "GaussianNB multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}},
            target_opset=TARGET_OPSET)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnGaussianNBMultiNoCl")


if __name__ == "__main__":
    unittest.main()
