"""
@file
@brief Helpers to manipulate :epkg:`scikit-learn` models.
"""
import numpy
from sklearn.base import (
    TransformerMixin, ClassifierMixin, RegressorMixin, BaseEstimator
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor


def enumerate_pipeline_models(pipe, coor=None, vs=None):
    """
    Enumerates all the models within a pipeline.

    @param      pipe        *scikit-learn* pipeline
    @param      coor        current coordinate
    @param      vs          subset of variables for the model, None for all
    @return                 iterator on models ``tuple(coordinate, model)``

    Example:

    .. runpython::
        :showcode:

        from sklearn.datasets import load_iris
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split
        from mlprodict.onnxrt.optim.sklearn_helper import enumerate_pipeline_models

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = make_pipeline(PCA(n_components=2),
                            LogisticRegression(solver="liblinear"))
        clr.fit(X_train, y_train)

        for a in enumerate_pipeline_models(clr):
            print(a)
    """
    if coor is None:
        coor = (0,)
    yield coor, pipe, vs
    if hasattr(pipe, 'transformer_and_mapper_list') and len(pipe.transformer_and_mapper_list):
        # azureml DataTransformer
        raise NotImplementedError("Unable to handle this specific case.")
    elif hasattr(pipe, 'mapper') and pipe.mapper:
        # azureml DataTransformer
        for couple in enumerate_pipeline_models(pipe.mapper, coor + (0,)):
            yield couple
    elif hasattr(pipe, 'built_features'):
        # sklearn_pandas.dataframe_mapper.DataFrameMapper
        for i, (columns, transformers, _) in enumerate(pipe.built_features):
            if isinstance(columns, str):
                columns = (columns,)
            if transformers is None:
                yield (coor + (i,)), None, columns
            else:
                for couple in enumerate_pipeline_models(transformers, coor + (i,), columns):
                    yield couple
    elif isinstance(pipe, Pipeline):
        for i, (_, model) in enumerate(pipe.steps):
            for couple in enumerate_pipeline_models(model, coor + (i,)):
                yield couple
    elif isinstance(pipe, ColumnTransformer):
        for i, (_, fitted_transformer, column) in enumerate(pipe.transformers):
            for couple in enumerate_pipeline_models(
                    fitted_transformer, coor + (i,), column):
                yield couple
    elif isinstance(pipe, FeatureUnion):
        for i, (_, model) in enumerate(pipe.transformer_list):
            for couple in enumerate_pipeline_models(model, coor + (i,)):
                yield couple
    elif isinstance(pipe, TransformedTargetRegressor):
        raise NotImplementedError(
            "Not yet implemented for TransformedTargetRegressor.")
    elif isinstance(pipe, (TransformerMixin, ClassifierMixin, RegressorMixin)):
        pass
    elif isinstance(pipe, BaseEstimator):
        pass
    else:
        raise TypeError(
            "pipe is not a scikit-learn object: {}\n{}".format(type(pipe), pipe))


def enumerate_fitted_arrays(model):
    """
    Enumerate all fitted arrays included in a
    :epkg:`scikit-learn` object.

    @param      model       :epkg:`scikit-learn` object
    @return                 enumerator

    One example:

    .. runpython::
        :showcode:

        from sklearn.datasets import load_iris
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split
        from mlprodict.onnxrt.optim.sklearn_helper import enumerate_fitted_arrays

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = make_pipeline(PCA(n_components=2),
                            LogisticRegression(solver="liblinear"))
        clr.fit(X_train, y_train)

        for a in enumerate_fitted_arrays(clr):
            print(a)
    """
    def enumerate__(obj):
        if isinstance(obj, (tuple, list)):
            for el in obj:
                for o in enumerate__(el):
                    yield (obj, el, o)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                for o in enumerate__(v):
                    yield (obj, k, v, o)
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                if k[-1] != '_' and k[0] != '_':
                    continue
                if isinstance(v, numpy.ndarray):
                    yield (obj, k, v)
                else:
                    for row in enumerate__(v):
                        yield row

    for row in enumerate_pipeline_models(model):
        coord = row[:-1]
        sub = row[1]
        last = row[2:]
        for sub_row in enumerate__(sub):
            yield coord + (sub, sub_row) + last


def pairwise_array_distances(l1, l2, metric='l1med'):
    """
    Computes pairwise distances between two lists of arrays
    *l1* and *l2*. The distance is 1e9 if shapes are not equal.

    @param      l1          first list of arrays
    @param      l2          second list of arrays
    @param      metric      metric to use, `'l1med'` compute
                            the average absolute error divided
                            by the ansolute median
    @return                 matrix
    """
    dist = numpy.full((len(l1), len(l2)), 1e9)
    for i, a1 in enumerate(l1):
        if not isinstance(a1, numpy.ndarray):
            continue
        for j, a2 in enumerate(l2):
            if not isinstance(a2, numpy.ndarray):
                continue
            if a1.shape != a2.shape:
                continue
            a = numpy.median(numpy.abs(a1))
            if a == 0:
                a = 1
            diff = numpy.sum(numpy.abs(a1 - a2)) / a
            dist[i, j] = diff / diff.size
    return dist
