# -*- encoding: utf-8 -*-
"""
@file
@brief Overloads a conversion function.
"""
from collections import OrderedDict
import numpy
import pandas
try:
    from sklearn.metrics._scorer import _PredictScorer
except ImportError:
    # scikit-learn < 0.22
    from sklearn.metrics.scorer import _PredictScorer
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType, DataType
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx import convert_sklearn
from skl2onnx.algebra.type_helper import _guess_type
from .register_rewritten_converters import register_rewritten_operators
from .register import register_converters
from .scorers import CustomScorerTransform


def convert_scorer(fct, initial_types, name=None,
                   target_opset=None, options=None,
                   dtype=numpy.float32,
                   custom_conversion_functions=None,
                   custom_shape_calculators=None,
                   custom_parsers=None):
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
    @param  dtype                       type to use to convert the model
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
    @return                             :epkg:`ONNX` graph
    """
    if hasattr(fct, '_score_func'):
        kwargs = fct._kwargs
        fct = fct._score_func
    else:
        kwargs = None
    if name is None:
        name = "mlprodict_fct_ONNX(%s)" % fct.__name__
    tr = CustomScorerTransform(fct.__name__, fct, kwargs)
    return convert_sklearn(tr, initial_types=initial_types,
                           target_opset=target_opset, options=options,
                           dtype=dtype,
                           custom_conversion_functions=custom_conversion_functions,
                           custom_shape_calculators=custom_shape_calculators,
                           custom_parsers=custom_parsers)


def guess_initial_types(X, initial_types):
    """
    Guesses initial types from an array or a dataframe.

    @param      X               array or dataframe
    @param      initial_types   hints about X
    @return                     data types
    """
    if X is None and initial_types is None:
        raise NotImplementedError("Initial types must be specified.")
    elif initial_types is None:
        if isinstance(X, (numpy.ndarray, pandas.DataFrame)):
            X = X[:1]
        if isinstance(X, pandas.DataFrame):
            initial_types = []
            for c in X.columns:
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
            return init
        if col.shape[0] is not None:
            return init
        if len(unique) > 0 and col.__class__ not in unique:
            return init
        unique.add(col.__class__)
    unique = list(unique)
    return [('X', unique[0]([None, sum(_[1].shape[1] for _ in init)]))]


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
        except NotImplementedError:
            return _replace_tensor_type(schema, tensor_type)
        if len(guessed) != len(schema):
            raise RuntimeError(
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

    import pprint
    data = pprint.pformat(model.__dict__)
    dirs = pprint.pformat(dir(model))
    if hasattr(model, 'dump_model'):
        dumped = model.dump_model()
        keys = list(sorted(dumped))
        last = pprint.pformat([keys, dumped])
        if len(last) >= 200000:
            last = last[:200000] + "\n..."
    else:
        last = ""
    raise NotImplementedError(
        "Unable to guess schema for model {}\n{}\n----\n{}\n------\n{}".format(
            model.__class__, data, dirs, last))


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None, options=None,
            dtype=numpy.float32, rewrite_ops=False):
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
    @param      dtype           type to use to convert the model
    @param      rewrite_ops     rewrites some existing converters,
                                the changes are permanent
    @return                     converted model

    The function rewrites function *to_onnx* from :epkg:`sklearn-onnx`
    but may changes a few converters if *rewrite_ops* is True.
    For example, :epkg:`ONNX` only supports *TreeEnsembleRegressor*
    for float but not for double. It becomes available
    if ``dtype=numpy.float64`` and ``rewrite_ops=True``.

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
        if options is not None:
            raise NotImplementedError(
                "options not yet implemented for OnnxOperatorMixin.")
        return model.to_onnx(X=X, name=name, dtype=dtype,
                             target_opset=target_opset)
    if rewrite_ops:
        old_values = register_rewritten_operators()
        register_converters()
    else:
        old_values = None

    def _guess_type_(X, itype, dtype):
        initial_types = guess_initial_types(X, itype)
        if dtype is None:
            raise RuntimeError("dtype cannot be None")
        if isinstance(dtype, FloatTensorType):
            dtype = numpy.float32
        elif isinstance(dtype, DoubleTensorType):
            dtype = numpy.float64
        new_dtype = dtype
        if isinstance(dtype, numpy.ndarray):
            new_dtype = dtype.dtype
        elif isinstance(dtype, DataType):
            new_dtype = numpy.float32
        if new_dtype not in (numpy.float32, numpy.float64, numpy.int64,
                             numpy.int32):
            raise NotImplementedError(
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
                it, _, ndt = _guess_type_(v, None, dtype)
                for i in range(len(it)):  # pylint: disable=C0200
                    it[i] = (k, it[i][1])  # pylint: disable=C0200
                initial_types.extend(it)
                dts.append(ndt)
            ndt = set(dts)
            if len(ndt) != 1:
                raise RuntimeError(
                    "Multiple dtype is not efficient {}.".format(ndt))
            dtype = dts[0]
            new_dtype = dts[0]
        res = convert_scorer(model, initial_types, name=name,
                             target_opset=target_opset, options=options,
                             dtype=new_dtype)
    else:
        if name is None:
            name = "mlprodict_ONNX(%s)" % model.__class__.__name__

        initial_types, dtype, new_dtype = _guess_type_(X, initial_types, dtype)
        try:
            res = convert_sklearn(model, initial_types=initial_types, name=name,
                                  target_opset=target_opset, options=options,
                                  dtype=new_dtype)
        except (TypeError, NameError):
            # older version of sklearn-onnx
            res = convert_sklearn(model, initial_types=initial_types, name=name,
                                  target_opset=target_opset, options=options)

    if old_values is not None:
        register_rewritten_operators(old_values)
    return res
