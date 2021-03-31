# -*- encoding: utf-8 -*-
"""
@file
@brief Overloads a conversion function.
"""
import pprint
from collections import OrderedDict
import numpy
import pandas
try:
    from sklearn.metrics._scorer import _PredictScorer
except ImportError:  # pragma: no cover
    # scikit-learn < 0.22
    from sklearn.metrics.scorer import _PredictScorer
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType, DataType, guess_numpy_type,
    StringTensorType, Int64TensorType)
from skl2onnx import convert_sklearn
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx.algebra.type_helper import _guess_type
from .register_rewritten_converters import register_rewritten_operators
from .register import register_converters
from .scorers import CustomScorerTransform


def convert_scorer(fct, initial_types, name=None,
                   target_opset=None, options=None,
                   custom_conversion_functions=None,
                   custom_shape_calculators=None,
                   custom_parsers=None, white_op=None,
                   black_op=None, final_types=None):
    """
    Converts a scorer into :epkg:`ONNX` assuming
    there exists a converter associated to it.
    The function wraps the function into a custom
    transformer, then calls function *convert_sklearn*
    from :epkg:`sklearn-onnx`.

    @param  fct                         function to convert (or a scorer from
                                        :epkg:`scikit-learn`)
    @param  initial_types               types information
    @param  name                        name of the produced model
    @param  target_opset                to do it with a different target opset
    @param  options                     additional parameters for the conversion
    @param  custom_conversion_functions a dictionary for specifying the user customized
                                        conversion function, it takes precedence over
                                        registered converters
    @param  custom_shape_calculators    a dictionary for specifying the user
                                        customized shape calculator
                                        it takes precedence over registered
                                        shape calculators.
    @param  custom_parsers              parsers determine which outputs is expected
                                        for which particular task, default parsers are
                                        defined for classifiers, regressors, pipeline but
                                        they can be rewritten, *custom_parsers* is a dictionary
                                        ``{ type: fct_parser(scope, model, inputs,
                                        custom_parsers=None) }``
    @param      white_op                white list of ONNX nodes allowed
                                        while converting a pipeline, if empty,
                                        all are allowed
    @param      black_op                black list of ONNX nodes allowed
                                        while converting a pipeline, if empty,
                                        none are blacklisted
    @param      final_types             a python list. Works the same way as
                                        initial_types but not mandatory, it is used
                                        to overwrites the type (if type is not None)
                                        and the name of every output.
    @return                             :epkg:`ONNX` graph
    """
    if hasattr(fct, '_score_func'):
        kwargs = fct._kwargs
        fct = fct._score_func
    else:
        kwargs = None  # pragma: no cover
    if name is None:
        name = "mlprodict_fct_ONNX(%s)" % fct.__name__
    tr = CustomScorerTransform(fct.__name__, fct, kwargs)
    return convert_sklearn(
        tr, initial_types=initial_types,
        target_opset=target_opset, options=options,
        custom_conversion_functions=custom_conversion_functions,
        custom_shape_calculators=custom_shape_calculators,
        custom_parsers=custom_parsers, white_op=white_op,
        black_op=black_op, final_types=final_types)


def guess_initial_types(X, initial_types):
    """
    Guesses initial types from an array or a dataframe.

    @param      X               array or dataframe
    @param      initial_types   hints about X
    @return                     data types
    """
    if X is None and initial_types is None:
        raise NotImplementedError(  # pragma: no cover
            "Initial types must be specified.")
    elif initial_types is None:
        if isinstance(X, (numpy.ndarray, pandas.DataFrame)):
            X = X[:1]
        if isinstance(X, pandas.DataFrame):
            initial_types = []
            for c in X.columns:
                if isinstance(X[c].values[0], (str, numpy.str)):
                    g = StringTensorType()
                else:
                    g = _guess_type(X[c].values)
                g.shape = [None, 1]
                initial_types.append((c, g))
        else:
            gt = _guess_type(X)
            initial_types = [('X', gt)]
    return initial_types


def _replace_tensor_type(schema, tensor_type):
    res = []
    for name, ty in schema:
        cl = ty.__class__
        if cl in (FloatTensorType, DoubleTensorType) and cl != tensor_type:
            ty = tensor_type(ty.shape)
        res.append((name, ty))
    return res


def guess_schema_from_data(X, tensor_type=None, schema=None):
    """
    Guesses initial types from a dataset.

    @param      X               dataset (dataframe, array)
    @param      tensor_type     if not None, replaces every
                                *FloatTensorType* or *DoubleTensorType*
                                by this one
    @param      schema          known schema
    @return                     schema (list of typed and named columns)
    """
    init = guess_initial_types(X, schema)
    if tensor_type is not None:
        init = _replace_tensor_type(init, tensor_type)
    # Grouping column
    unique = set()
    for _, col in init:
        if len(col.shape) != 2:
            return init  # pragma: no cover
        if col.shape[0] is not None:
            return init  # pragma: no cover
        if len(unique) > 0 and col.__class__ not in unique:
            return init  # pragma: no cover
        unique.add(col.__class__)
    unique = list(unique)
    return [('X', unique[0]([None, sum(_[1].shape[1] for _ in init)]))]


def get_inputs_from_data(X, schema=None):
    """
    Produces input data for *onnx* runtime.

    @param  X       data
    @param  schema  schema if None, schema is guessed with
        @see fn guess_schema_from_data
    @return input data
    """
    def _cast_data(X, ct):
        if isinstance(ct, FloatTensorType):
            return X.astype(numpy.float32)
        if isinstance(ct, DoubleTensorType):
            return X.astype(numpy.float64)
        if isinstance(ct, StringTensorType):
            return X.astype(numpy.str)
        if isinstance(ct, Int64TensorType):
            return X.astype(numpy.int64)
        raise RuntimeError(  # pragma: no cover
            "Unexpected column type {} for type {}."
            "".format(ct, type(X)))

    if schema is None:
        schema = guess_schema_from_data(X)
    if isinstance(X, numpy.ndarray):
        if len(schema) != 1:
            raise RuntimeError(  # pragma: no cover
                "More than one column but input is an array.")
        return {schema[0][0]: _cast_data(X, schema[0][1])}
    elif isinstance(X, pandas.DataFrame):
        if len(schema) != X.shape[1]:
            raise RuntimeError(  # pragma: no cover
                "Mismatch between onnx columns {} and DataFrame columns {}"
                "".format(len(schema), X.shape[1]))
        return {sch[0]: _cast_data(X[c].values, sch[1]).reshape((-1, 1))
                for sch, c in zip(schema, X.columns)}
    else:
        raise TypeError(  # pragma: no cover
            "Unexpected type {}, expecting an array or a dataframe."
            "".format(type(X)))


def guess_schema_from_model(model, tensor_type=None, schema=None):
    """
    Guesses initial types from a model.

    @param      X               dataset (dataframe, array)
    @param      tensor_type     if not None, replaces every
                                *FloatTensorType* or *DoubleTensorType*
                                by this one
    @param      schema          known schema
    @return                     schema (list of typed and named columns)
    """
    if schema is not None:
        try:
            guessed = guess_schema_from_model(model)
        except NotImplementedError:  # pragma: no cover
            return _replace_tensor_type(schema, tensor_type)
        if len(guessed) != len(schema):
            raise RuntimeError(  # pragma: no cover
                "Given schema and guessed schema are not the same:\nGOT: {}\n-----\nGOT:\n{}".format(
                    schema, guessed))
        return _replace_tensor_type(schema, tensor_type)

    if hasattr(model, 'coef_'):
        # linear model
        init = [('X', FloatTensorType([None, model.coef_.shape[1]]))]
        return _replace_tensor_type(init, tensor_type)
    elif hasattr(model, 'dump_model'):
        dumped = model.dump_model()
        if isinstance(dumped, dict) and 'feature_names' in dumped:
            names = dumped['feature_names']
            init = [(name, FloatTensorType([None, 1])) for name in names]
            return _replace_tensor_type(init, tensor_type)

    data = pprint.pformat(model.__dict__)
    dirs = pprint.pformat(dir(model))
    if hasattr(model, 'dump_model'):  # pragma: no cover
        dumped = model.dump_model()
        keys = list(sorted(dumped))
        last = pprint.pformat([keys, dumped])
        if len(last) >= 200000:
            last = last[:200000] + "\n..."
    else:
        last = ""
    raise NotImplementedError(  # pragma: no cover
        "Unable to guess schema for model {}\n{}\n----\n{}\n------\n{}".format(
            model.__class__, data, dirs, last))


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None, options=None, rewrite_ops=False,
            white_op=None, black_op=None, final_types=None):
    """
    Converts a model using on :epkg:`sklearn-onnx`.

    @param      model           model to convert or a function
                                wrapped into :epkg:`_PredictScorer` with
                                function :epkg:`make_scorer`
    @param      X               training set (at least one row),
                                can be None, it is used to infered the
                                input types (*initial_types*)
    @param      initial_types   if *X* is None, then *initial_types* must be
                                defined
    @param      name            name of the produced model
    @param      target_opset    to do it with a different target opset
    @param      options         additional parameters for the conversion
    @param      rewrite_ops     rewrites some existing converters,
                                the changes are permanent
    @param      white_op        white list of ONNX nodes allowed
                                while converting a pipeline, if empty,
                                all are allowed
    @param      black_op        black list of ONNX nodes allowed
                                while converting a pipeline, if empty,
                                none are blacklisted
    @param      final_types     a python list. Works the same way as
                                initial_types but not mandatory, it is used
                                to overwrites the type (if type is not None)
                                and the name of every output.
    @return                     converted model

    The function rewrites function *to_onnx* from :epkg:`sklearn-onnx`
    but may changes a few converters if *rewrite_ops* is True.
    For example, :epkg:`ONNX` only supports *TreeEnsembleRegressor*
    for float but not for double. It becomes available
    if ``rewrite_ops=True``.

    .. faqref::
        :title: How to deal with a dataframe as input?

        Each column of the dataframe is considered as an named input.
        The first step is to make sure that every column type is correct.
        :epkg:`pandas` tends to select the least generic type to
        hold the content of one column. :epkg:`ONNX` does not automatically
        cast the data it receives. The data must have the same type with
        the model is converted and when the converted model receives
        the data to predict.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning

            from io import StringIO
            from textwrap import dedent
            import numpy
            import pandas
            from pyquickhelper.pycode import ExtTestCase
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from mlprodict.onnx_conv import to_onnx
            from mlprodict.onnxrt import OnnxInference

            text = dedent('''
                __SCHEMA__
                7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
                7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
                7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
                11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,red
                ''')
            text = text.replace(
                "__SCHEMA__",
                "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,"
                "free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,"
                "alcohol,quality,color")

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
            pred = pipe.transform(X_train)
            print(pred)

            model_onnx = to_onnx(pipe, X_train, target_opset=12)
            oinf = OnnxInference(model_onnx)

            # The dataframe is converted into a dictionary,
            # each key is a column name, each value is a numpy array.
            inputs = {c: X_train[c].values for c in X_train.columns}
            inputs = {c: v.reshape((v.shape[0], 1)) for c, v in inputs.items()}

            onxp = oinf.run(inputs)
            print(onxp)
    """
    if isinstance(model, OnnxOperatorMixin):
        if not hasattr(model, 'op_version'):
            raise RuntimeError(  # pragma: no cover
                "Missing attribute 'op_version' for type '{}'.".format(
                    type(model)))
        return model.to_onnx(
            X=X, name=name, options=options, black_op=black_op,
            white_op=white_op, final_types=final_types)

    if rewrite_ops:
        old_values, old_shapes = register_rewritten_operators()
        register_converters()
    else:
        old_values, old_shapes = {}, {}

    def _guess_type_(X, itype, dtype):
        initial_types = guess_initial_types(X, itype)
        if dtype is None:
            if hasattr(X, 'dtypes'):  # DataFrame
                dtype = numpy.float32
            elif hasattr(X, 'dtype'):
                dtype = X.dtype
            elif hasattr(X, 'type'):
                dtype = guess_numpy_type(X.type)
            elif initial_types is not None:
                dtype = guess_numpy_type(initial_types[0][1])
            else:
                raise RuntimeError(  # pragma: no cover
                    "dtype cannot be guessed: {}".format(
                        type(X)))
            if dtype != numpy.float64:
                dtype = numpy.float32
        if dtype is None:
            raise RuntimeError("dtype cannot be None")  # pragma: no cover
        if isinstance(dtype, FloatTensorType):
            dtype = numpy.float32  # pragma: no cover
        elif isinstance(dtype, DoubleTensorType):
            dtype = numpy.float64  # pragma: no cover
        new_dtype = dtype
        if isinstance(dtype, numpy.ndarray):
            new_dtype = dtype.dtype  # pragma: no cover
        elif isinstance(dtype, DataType):
            new_dtype = numpy.float32  # pragma: no cover
        if new_dtype not in (numpy.float32, numpy.float64, numpy.int64,
                             numpy.int32, numpy.float16):
            raise NotImplementedError(  # pragma: no cover
                "dtype should be real not {} ({})".format(new_dtype, dtype))
        return initial_types, dtype, new_dtype

    if isinstance(model, _PredictScorer):
        if X is not None and not isinstance(X, OrderedDict):
            raise ValueError("For a scorer, parameter X should be a OrderedDict not {}."
                             "".format(type(X)))
        if initial_types is None:
            dts = []
            initial_types = []
            for k, v in X.items():
                if hasattr(v, 'dtype'):
                    dtype = guess_numpy_type(v.dtype)
                else:
                    dtype = v  # pragma: no cover
                it, _, ndt = _guess_type_(v, None, dtype)
                for i in range(len(it)):  # pylint: disable=C0200
                    it[i] = (k, it[i][1])  # pylint: disable=C0200
                initial_types.extend(it)
                dts.append(ndt)
            ndt = set(dts)
            if len(ndt) != 1:
                raise RuntimeError(  # pragma: no cover
                    "Multiple dtype is not efficient {}.".format(ndt))
        res = convert_scorer(model, initial_types, name=name,
                             target_opset=target_opset, options=options,
                             black_op=black_op, white_op=white_op,
                             final_types=final_types)
    else:
        if name is None:
            name = "mlprodict_ONNX(%s)" % model.__class__.__name__

        initial_types, dtype, _ = _guess_type_(X, initial_types, None)
        res = convert_sklearn(model, initial_types=initial_types, name=name,
                              target_opset=target_opset, options=options,
                              black_op=black_op, white_op=white_op,
                              final_types=final_types)

    register_rewritten_operators(old_values, old_shapes)
    return res
