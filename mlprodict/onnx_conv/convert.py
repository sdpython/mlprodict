# -*- encoding: utf-8 -*-
# pylint: disable=C0302,R0914
"""
@file
@brief Overloads a conversion function.
"""
import json
import pprint
from collections import OrderedDict
import logging
import numpy
from onnx import ValueInfoProto
import pandas
try:
    from sklearn.metrics._scorer import _PredictScorer
except ImportError:  # pragma: no cover
    # scikit-learn < 0.22
    from sklearn.metrics.scorer import _PredictScorer
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.utils.metaestimators import _BaseComposition
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType, DataType, guess_numpy_type,
    StringTensorType, Int64TensorType, _guess_type_proto)
from skl2onnx import convert_sklearn
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx.algebra.type_helper import _guess_type
from ..onnx_tools.onnx_manipulations import onnx_rename_names
from ..onnx_tools.onnx2py_helper import (
    guess_dtype, get_tensor_shape, get_tensor_elem_type)
from .register_rewritten_converters import register_rewritten_operators
from .register import register_converters
from .scorers import CustomScorerTransform


logger = logging.getLogger('mlprodict')


def _fix_opset_skl2onnx():
    import skl2onnx
    from .. import __max_supported_opset__
    if skl2onnx.__max_supported_opset__ != __max_supported_opset__:
        skl2onnx.__max_supported_opset__ = __max_supported_opset__  # pragma: no cover


def convert_scorer(fct, initial_types, name=None,
                   target_opset=None, options=None,
                   custom_conversion_functions=None,
                   custom_shape_calculators=None,
                   custom_parsers=None, white_op=None,
                   black_op=None, final_types=None,
                   verbose=0):
    """
    Converts a scorer into :epkg:`ONNX` assuming
    there exists a converter associated to it.
    The function wraps the function into a custom
    transformer, then calls function *convert_sklearn*
    from :epkg:`sklearn-onnx`.

    :param fct: function to convert (or a scorer from :epkg:`scikit-learn`)
    :param initial_types: types information
    :param name: name of the produced model
    :param target_opset: to do it with a different target opset
    :param options: additional parameters for the conversion
    :param custom_conversion_functions: a dictionary for specifying the user
        customized conversion function, it takes precedence over
        registered converters
    :param custom_shape_calculators: a dictionary for specifying the user
        customized shape calculator it takes precedence over registered
        shape calculators.
    :param custom_parsers: parsers determine which outputs is expected
        for which particular task, default parsers are
        defined for classifiers, regressors, pipeline but
        they can be rewritten, *custom_parsers* is a dictionary
        ``{ type: fct_parser(scope, model, inputs,
        custom_parsers=None) }``
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline, if empty, all are allowed
    :param black_op: black list of ONNX nodes allowed
        while converting a pipeline, if empty, none are blacklisted
    :param final_types: a python list. Works the same way as
        initial_types but not mandatory, it is used
        to overwrites the type (if type is not None)
        and the name of every output.
    :param verbose: displays information while converting
    :return: :epkg:`ONNX` graph
    """
    if hasattr(fct, '_score_func'):
        kwargs = fct._kwargs
        fct = fct._score_func
    else:
        kwargs = None  # pragma: no cover
    if name is None:
        name = f"mlprodict_fct_ONNX({fct.__name__})"
    tr = CustomScorerTransform(fct.__name__, fct, kwargs)
    _fix_opset_skl2onnx()
    return convert_sklearn(
        tr, initial_types=initial_types,
        target_opset=target_opset, options=options,
        custom_conversion_functions=custom_conversion_functions,
        custom_shape_calculators=custom_shape_calculators,
        custom_parsers=custom_parsers, white_op=white_op,
        black_op=black_op, final_types=final_types,
        verbose=verbose)


def guess_initial_types(X, initial_types):
    """
    Guesses initial types from an array or a dataframe.

    :param X: array or dataframe
    :param initial_types: hints about X
    :return: data types
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
                if isinstance(X[c].values[0], (str, numpy.str_)):
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
            return X.astype(numpy.str_)
        if isinstance(ct, Int64TensorType):
            return X.astype(numpy.int64)
        raise RuntimeError(  # pragma: no cover
            f"Unexpected column type {ct} for type {type(X)}.")

    if schema is None:
        schema = guess_schema_from_data(X)
    if isinstance(X, numpy.ndarray):
        if len(schema) != 1:
            raise RuntimeError(  # pragma: no cover
                "More than one column but input is an array.")
        return {schema[0][0]: _cast_data(X, schema[0][1])}
    if isinstance(X, pandas.DataFrame):
        if len(schema) != X.shape[1]:
            raise RuntimeError(  # pragma: no cover
                "Mismatch between onnx columns {} and DataFrame columns {}"
                "".format(len(schema), X.shape[1]))
        return {sch[0]: _cast_data(X[c].values, sch[1]).reshape((-1, 1))
                for sch, c in zip(schema, X.columns)}
    raise TypeError(  # pragma: no cover
        f"Unexpected type {type(X)}, expecting an array or a dataframe.")


def guess_schema_from_model(model, tensor_type=None, schema=None):
    """
    Guesses initial types from a model.

    @param      model           model
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


def _guess_type_(X, itype, dtype):
    initial_types = guess_initial_types(X, itype)
    if dtype is None:
        if hasattr(X, 'dtypes'):  # DataFrame
            dtype = numpy.float32
        elif hasattr(X, 'dtype'):
            dtype = X.dtype
        elif hasattr(X, 'type'):
            dtype = guess_numpy_type(X.type)
        elif isinstance(initial_types[0], ValueInfoProto):
            dtype = guess_dtype(initial_types[0].type.tensor_type.elem_type)
        elif initial_types is not None:
            dtype = guess_numpy_type(initial_types[0][1])
        else:
            raise RuntimeError(  # pragma: no cover
                f"dtype cannot be guessed: {type(X)}")
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
            f"dtype should be real not {new_dtype} ({dtype})")
    return initial_types, dtype, new_dtype


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None, options=None, rewrite_ops=False,
            white_op=None, black_op=None, final_types=None,
            rename_strategy=None, verbose=0,
            as_function=False, prefix_name=None,
            run_shape=False, single_function=True):
    """
    Converts a model using on :epkg:`sklearn-onnx`.

    :param model: model to convert or a function
        wrapped into :epkg:`_PredictScorer` with
        function :epkg:`make_scorer`
    :param X: training set (at least one row),
        can be None, it is used to infered the
        input types (*initial_types*)
    :param initial_types: if *X* is None, then *initial_types*
        must be defined
    :param name: name of the produced model
    :param target_opset: to do it with a different target opset
    :param options: additional parameters for the conversion
    :param rewrite_ops: rewrites some existing converters,
        the changes are permanent
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline, if empty, all are allowed
    :param black_op: black list of ONNX nodes allowed
        while converting a pipeline, if empty,
        none are blacklisted
    :param final_types: a python list. Works the same way as
        initial_types but not mandatory, it is used
        to overwrites the type (if type is not None)
        and the name of every output.
    :param rename_strategy: rename any name in the graph, select shorter
        names, see @see fn onnx_rename_names
    :param verbose: display information while converting the model
    :param as_function: exposes every model in a pipeline as a function,
        the main graph contains the pipeline structure,
        see :ref:`onnxsklearnfunctionsrst` for an example
    :param prefix_name: used if *as_function* is True, to give
        a prefix to variable in a pipeline
    :param run_shape: run shape inference
    :param single_function: if *as_function* is True, the function returns one graph
        with one call to the main function if *single_function* is True or
        a list of node corresponding to the graph structure
    :return: converted model

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

    .. versionchanged:: 0.9
        Parameter *as_function* was added.
    """
    logger.debug("to_onnx(%s, X=%r, initial_types=%r, target_opset=%r, "
                 "options=%r, rewrite_ops=%r, white_op=%r, black_op=%r, "
                 "final_types=%r)",
                 model.__class__.__name__, type(X), initial_types,
                 target_opset, options, rewrite_ops, white_op, black_op,
                 final_types)

    if isinstance(model, OnnxOperatorMixin):
        if not hasattr(model, 'op_version'):
            raise RuntimeError(  # pragma: no cover
                f"Missing attribute 'op_version' for type '{type(model)}'.")
        _fix_opset_skl2onnx()
        return model.to_onnx(
            X=X, name=name, options=options, black_op=black_op,
            white_op=white_op, final_types=final_types,
            target_opset=target_opset)
        # verbose=verbose)

    if rewrite_ops:
        old_values, old_shapes = register_rewritten_operators()
        register_converters()
    else:
        old_values, old_shapes = {}, {}

    if as_function and isinstance(
            model, (ColumnTransformer, Pipeline, FeatureUnion)):
        res = to_onnx_function(
            model, X=X, name=name, initial_types=initial_types,
            target_opset=target_opset, options=options,
            rewrite_ops=False,  # already handled
            white_op=white_op, black_op=black_op, final_types=final_types,
            rename_strategy=None,  # already handled
            verbose=verbose, prefix_name=prefix_name,
            run_shape=run_shape, single_function=single_function)

    elif isinstance(model, _PredictScorer):
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
                    f"Multiple dtype is not efficient {ndt}.")
        res = convert_scorer(model, initial_types, name=name,
                             target_opset=target_opset, options=options,
                             black_op=black_op, white_op=white_op,
                             final_types=final_types, verbose=verbose)
    else:
        if name is None:
            name = f"mlprodict_ONNX({model.__class__.__name__})"

        initial_types, dtype, _ = _guess_type_(X, initial_types, None)

        _fix_opset_skl2onnx()
        res = convert_sklearn(model, initial_types=initial_types, name=name,
                              target_opset=target_opset, options=options,
                              black_op=black_op, white_op=white_op,
                              final_types=final_types, verbose=verbose)

    register_rewritten_operators(old_values, old_shapes)

    # optimisation
    if rename_strategy is not None:
        res = onnx_rename_names(res, strategy=rename_strategy)
    return res


def _guess_s2o_type(vtype: ValueInfoProto):
    return _guess_type_proto(
        get_tensor_elem_type(vtype), get_tensor_shape(vtype))


def _new_options(options, prefix, sklop):
    if sklop is None:
        raise RuntimeError(  # pragma: no cover
            "sklop cannot be None.")
    if isinstance(sklop, str):
        return None  # pragma: no cover
    if options is None:
        step_options = None
    else:
        step_options = {}
        for k, v in options.items():
            if k.startswith(prefix):
                step_options[k[len(prefix):]] = v
            elif '__' in k:
                step_options[k.split('__', maxsplit=1)[1]] = v
            if isinstance(sklop, _BaseComposition):
                step_options[k] = v
            else:
                from skl2onnx._supported_operators import _get_sklearn_operator_name
                from skl2onnx.common._registration import get_converter
                alias = _get_sklearn_operator_name(type(sklop))
                if alias is None:
                    step_options[k] = v
                else:
                    conv = get_converter(alias)
                    allowed = conv.get_allowed_options()
                    if allowed is not None and k in allowed:
                        step_options[k] = v
    return step_options


class _ParamEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=W0237
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as e:
            # Unable to serialize
            return '{"classname": "%s", "EXC": "%s"}' % (
                obj.__class__.__name__, str(e))


def get_sklearn_json_params(model):
    """
    Retrieves all the parameters of a :epkg:`scikit-learn` model.
    """
    pars = model.get_params(deep=False)
    try:
        return json.dumps(pars, cls=_ParamEncoder)
    except TypeError as e:  # pragma: no cover
        raise RuntimeError(
            f"Unable to serialize parameters {pprint.pformat(pars)}.") from e


def _to_onnx_function_pipeline(
        model, X=None, name=None, initial_types=None,
        target_opset=None, options=None, rewrite_ops=False,
        white_op=None, black_op=None, final_types=None,
        rename_strategy=None, verbose=0,
        prefix_name=None, run_shape=False,
        single_function=True):

    from ..npy.xop_variable import Variable
    from ..npy.xop import OnnxOperatorFunction, loadop
    from ..onnx_tools.onnx_manipulations import onnx_model_to_function

    OnnxIdentity = loadop('Identity')

    if len(model.steps) == 0:
        raise RuntimeError(  # pragma: no cover
            "The pipeline to be converted cannot be empty.")

    if target_opset is None:
        from .. import __max_supported_opset__
        op_version = __max_supported_opset__
    elif isinstance(target_opset, int):
        op_version = target_opset
    else:
        from .. import __max_supported_opset__
        op_version = target_opset.get('', __max_supported_opset__)

    i_types = guess_initial_types(X, initial_types)
    input_nodes = [OnnxIdentity(i[0], op_version=op_version)
                   for i in i_types]

    inputs = i_types
    last_op = None
    for i_step, step in enumerate(model.steps):
        prefix = step[0] + "__"
        step_options = _new_options(options, prefix, step[1])
        if prefix_name is not None:
            prefix = prefix_name + prefix
        protom = to_onnx(
            step[1], name=name, initial_types=inputs,
            target_opset=target_opset,
            options=step_options, rewrite_ops=rewrite_ops,
            white_op=white_op, black_op=black_op, verbose=verbose,
            as_function=True, prefix_name=prefix, run_shape=run_shape,
            single_function=False)
        for o in protom.graph.output:
            if get_tensor_elem_type(o) == 0:
                raise RuntimeError(  # pragma: no cover
                    "Unabble to guess output type of output %r "
                    "from model step %d: %r, output=%r." % (
                        protom.graph.output, i_step, step[1], o))
        jspar = 'HYPER:{"%s":%s}' % (
            step[1].__class__.__name__, get_sklearn_json_params(step[1]))
        protof, subf = onnx_model_to_function(
            protom, domain='sklearn',
            name=f"{prefix}_{step[1].__class__.__name__}_{i_step}",
            doc_string=jspar)
        input_names = [f"{step[0]}_{o}" for o in protof.input]
        if last_op is not None:
            if len(input_names) == 1:
                input_nodes = [OnnxIdentity(
                    last_op, output_names=input_names[0],
                    op_version=op_version)]
            else:
                input_nodes = [OnnxIdentity(last_op[i], output_names=[n],  # pylint: disable=E1136
                                            op_version=op_version)
                               for i, n in enumerate(input_names)]
        output_names = [f"{step[0]}_{o}" for o in protof.output]

        logger.debug("_to_onnx_function_pipeline:%s:%r->%r:%r:%s",
                     step[1].__class__.__name__,
                     input_names, output_names,
                     len(protof.node), jspar)

        op = OnnxOperatorFunction(
            protof, *input_nodes, output_names=output_names,
            sub_functions=subf)
        last_op = op
        inputs = [
            ('X%d' % i, _guess_s2o_type(o))
            for i, o in enumerate(protom.graph.output)]

    logger.debug("_to_onnx_function_pipeline:end:(%s-%d, X=%r, "
                 "initial_types=%r, target_opset=%r, "
                 "options=%r, rewrite_ops=%r, white_op=%r, black_op=%r, "
                 "final_types=%r, outputs=%r)",
                 model.__class__.__name__, id(model),
                 type(X), initial_types,
                 target_opset, options, rewrite_ops, white_op, black_op,
                 final_types, inputs)

    i_vars = [Variable.from_skl2onnx_tuple(i) for i in i_types]
    if final_types is None:
        outputs_tuple = [
            (n, _guess_s2o_type(o))
            for i, (n, o) in enumerate(zip(output_names, protom.graph.output))]
        outputs = [Variable.from_skl2onnx_tuple(i) for i in outputs_tuple]
    else:
        outputs = final_types

    onx = last_op.to_onnx(inputs=i_vars, target_opset=target_opset,
                          verbose=verbose, run_shape=run_shape,
                          outputs=outputs)

    for o in onx.graph.output:
        if get_tensor_elem_type(o) == 0:
            raise RuntimeError(  # pragma: no cover
                "Unable to guess output type of output %r "
                "from model %r." % (onx.graph.output, model))
    return onx


def get_column_index(i, inputs):
    """
    Returns a tuples (variable index, column index in that variable).
    The function has two different behaviours, one when *i* (column index)
    is an integer, another one when *i* is a string (column name).
    If *i* is a string, the function looks for input name with
    this name and returns `(index, 0)`.
    If *i* is an integer, let's assume first we have two inputs
    `I0 = FloatTensorType([None, 2])` and `I1 = FloatTensorType([None, 3])`,
    in this case, here are the results:

    ::

        get_column_index(0, inputs) -> (0, 0)
        get_column_index(1, inputs) -> (0, 1)
        get_column_index(2, inputs) -> (1, 0)
        get_column_index(3, inputs) -> (1, 1)
        get_column_index(4, inputs) -> (1, 2)
    """
    if isinstance(i, int):
        if i == 0:
            # Useful shortcut, skips the case when end is None
            # (unknown dimension)
            return 0, 0
        vi = 0
        pos = 0
        end = inputs[0][1].shape[1]
        if end is None:
            raise RuntimeError(  # pragma: no cover
                "Cannot extract a specific column %r when "
                "one input (%r) has unknown "
                "dimension." % (i, inputs[0]))
        while True:
            if pos <= i < end:
                return vi, i - pos
            vi += 1
            pos = end
            if vi >= len(inputs):
                raise RuntimeError(
                    "Input %r (i=%r, end=%r) is not available in\n%r" % (
                        vi, i, end, pprint.pformat(inputs)))
            rel_end = inputs[vi][1].shape[1]
            if rel_end is None:
                raise RuntimeError(  # pragma: no cover
                    "Cannot extract a specific column %r when "
                    "one input (%r) has unknown "
                    "dimension." % (i, inputs[vi]))
            end += rel_end
    else:
        for ind, inp in enumerate(inputs):
            if inp[0] == i:
                return ind, 0
        raise RuntimeError(  # pragma: no cover
            "Unable to find column name %r among names %r. "
            "Make sure the input names specified with parameter "
            "initial_types fits the column names specified in the "
            "pipeline to convert. This may happen because a "
            "ColumnTransformer follows a transformer without "
            "any mapped converter in a pipeline." % (
                i, [n[0] for n in inputs]))


def get_column_indices(indices, inputs, multiple):
    """
    Returns the requested graph inpudes based on their
    indices or names. See :func:`get_column_index`.

    :param indices: variables indices or names
    :param inputs: graph inputs
    :param multiple: allows column to come from multiple variables
    :return: a tuple *(variable name, list of requested indices)* if
        *multiple* is False, a dictionary *{ var_index: [ list of
        requested indices ] }*
        if *multiple* is True
    """
    if multiple:
        res = OrderedDict()
        for p in indices:
            ov, onnx_i = get_column_index(p, inputs)
            if ov not in res:
                res[ov] = []
            res[ov].append(onnx_i)
        return res

    onnx_var = None
    onnx_is = []
    for p in indices:
        ov, onnx_i = get_column_index(p, inputs)
        onnx_is.append(onnx_i)
        if onnx_var is None:
            onnx_var = ov
        elif onnx_var != ov:
            cols = [onnx_var, ov]
            raise NotImplementedError(  # pragma: no cover
                "sklearn-onnx is not able to merge multiple columns from "
                "multiple variables ({0}). You should think about merging "
                "initial types.".format(cols))
    return onnx_var, onnx_is


def _merge_initial_types(i_types, transform_inputs, merge):
    if len(i_types) == len(transform_inputs):
        new_types = []
        for it, sli in zip(i_types, transform_inputs):
            name, ty = it
            begin, end = sli.inputs[1], sli.inputs[2]
            delta = end - begin
            shape = [ty.shape[0], int(delta[0])]
            new_types.append((name, ty.__class__(shape)))
    else:
        raise NotImplementedError(  # pragma: no cover
            "Not implemented when i_types=%r, transform_inputs=%r."
            "" % (i_types, transform_inputs))
    if merge and len(new_types) > 1:
        raise NotImplementedError(  # pragma: no cover
            "Cannot merge %r built from i_types=%r, transform_inputs=%r."
            "" % (new_types, i_types, transform_inputs))
    return new_types


def _to_onnx_function_column_transformer(
        model, X=None, name=None, initial_types=None,
        target_opset=None, options=None, rewrite_ops=False,
        white_op=None, black_op=None, final_types=None,
        rename_strategy=None, verbose=0,
        prefix_name=None, run_shape=False,
        single_function=True):

    from sklearn.preprocessing import OneHotEncoder
    from ..npy.xop_variable import Variable
    from ..npy.xop import OnnxOperatorFunction, loadop
    from ..onnx_tools.onnx_manipulations import onnx_model_to_function

    OnnxConcat, OnnxSlice, OnnxIdentity = loadop('Concat', 'Slice', 'Identity')

    transformers = model.transformers_
    if len(transformers) == 0:
        raise RuntimeError(  # pragma: no cover
            "The ColumnTransformer to be converted cannot be empty.")

    if target_opset is None:
        from .. import __max_supported_opset__
        op_version = __max_supported_opset__
    elif isinstance(target_opset, int):
        op_version = target_opset
    else:  # pragma: no cover
        from .. import __max_supported_opset__
        op_version = target_opset.get('', __max_supported_opset__)

    i_types = guess_initial_types(X, initial_types)
    ops = []
    protoms = []
    output_namess = []
    for i_step, (name_step, op, column_indices) in enumerate(transformers):
        if op == 'drop':
            continue
        input_nodes = [OnnxIdentity(i[0], op_version=op_version)
                       for i in initial_types]
        if isinstance(column_indices, slice):
            column_indices = list(range(
                column_indices.start
                if column_indices.start is not None else 0,
                column_indices.stop, column_indices.step
                if column_indices.step is not None else 1))
        elif isinstance(column_indices, (int, str)):
            column_indices = [column_indices]
        names = get_column_indices(column_indices, i_types, multiple=True)
        transform_inputs = []
        for onnx_var, onnx_is in names.items():
            if max(onnx_is) - min(onnx_is) != len(onnx_is) - 1:
                raise RuntimeError(  # pragma: no cover
                    "The converter only with contiguous columns indices not %r "
                    "for step %r." % (column_indices, name_step))
            tr_inputs = OnnxSlice(input_nodes[onnx_var],
                                  numpy.array([onnx_is[0]], dtype=numpy.int64),
                                  numpy.array([onnx_is[-1] + 1],
                                              dtype=numpy.int64),
                                  numpy.array([1], dtype=numpy.int64),
                                  op_version=op_version)
            transform_inputs.append(tr_inputs)

        merged_cols = False
        if len(transform_inputs) > 1:
            if isinstance(op, Pipeline):
                if not isinstance(op.steps[0][1],
                                  (OneHotEncoder, ColumnTransformer)):
                    merged_cols = True
            elif not isinstance(op, (OneHotEncoder, ColumnTransformer)):
                merged_cols = True

        if merged_cols:
            concatenated = OnnxConcat(
                *transform_inputs, op_version=op_version, axis=1)
        else:
            concatenated = transform_inputs
        initial_types = _merge_initial_types(
            i_types, transform_inputs, merged_cols)

        prefix = name_step + "__"
        step_options = _new_options(options, prefix, op)
        if prefix_name is not None:
            prefix = prefix_name + prefix

        if op == 'passthrough':
            ops.extend(concatenated)
            continue

        protom = to_onnx(
            op, name=name_step, X=X, initial_types=initial_types,
            target_opset=target_opset,
            options=step_options, rewrite_ops=rewrite_ops,
            white_op=white_op, black_op=black_op, verbose=verbose,
            as_function=True, prefix_name=prefix, run_shape=run_shape,
            single_function=False)
        protoms.append(protom)

        for o in protom.graph.output:
            if get_tensor_elem_type(o) == 0:
                raise RuntimeError(  # pragma: no cover
                    "Unabble to guess output type of output %r "
                    "from model step %d: %r." % (
                        protom.graph.output, i_step, op))
        jspar = 'HYPER:{"%s":%s}' % (
            op.__class__.__name__, get_sklearn_json_params(op))
        protof, fcts = onnx_model_to_function(
            protom, domain='sklearn',
            name=f"{prefix}_{op.__class__.__name__}_{id(op)}",
            doc_string=jspar)
        output_names = [f"{name_step}_{o}" for o in protof.output]
        output_namess.append(output_names)

        logger.debug("_to_onnx_function_column_transformer:%s:->%r:%r:%s",
                     op.__class__.__name__, output_names, len(protof.node), jspar)

        op = OnnxOperatorFunction(
            protof, *concatenated, output_names=output_names,
            sub_functions=list(fcts))
        ops.append(op)

    logger.debug("_to_onnx_function_column_transformer:end:(%s-%d, X=%r, "
                 "initial_types=%r, target_opset=%r, "
                 "options=%r, rewrite_ops=%r, white_op=%r, black_op=%r, "
                 "final_types=%r, outputs=%r)",
                 model.__class__.__name__, id(model),
                 type(X), initial_types, target_opset,
                 options, rewrite_ops, white_op, black_op,
                 final_types, i_types)

    i_vars = [Variable.from_skl2onnx_tuple(i) for i in i_types]
    if final_types is None:
        outputs_tuple = []
        for protom, output_names in zip(protoms, output_namess):
            outputs_tuple.extend([
                (n, _guess_s2o_type(o))
                for i, (n, o) in enumerate(zip(output_names, protom.graph.output))])
        outputs = [Variable.from_skl2onnx_tuple(i) for i in outputs_tuple]
    else:
        outputs = final_types

    last_op = OnnxConcat(*ops, op_version=op_version, axis=1)

    onx = last_op.to_onnx(inputs=i_vars, target_opset=target_opset,
                          verbose=verbose, run_shape=run_shape,
                          outputs=outputs)

    for o in onx.graph.output:
        if get_tensor_elem_type(o) == 0:
            raise RuntimeError(  # pragma: no cover
                "Unable to guess output type of output %r "
                "from model %r." % (onx.graph.output, model))
    return onx


def to_onnx_function(model, X=None, name=None, initial_types=None,
                     target_opset=None, options=None, rewrite_ops=False,
                     white_op=None, black_op=None, final_types=None,
                     rename_strategy=None, verbose=0,
                     prefix_name=None, run_shape=False,
                     single_function=True):
    """
    Converts a model using on :epkg:`sklearn-onnx`.
    The functions works as the same as function @see fn to_onnx
    but every model is exported as a single function and the main
    graph represents the pipeline structure.

    :param model: model to convert or a function
        wrapped into :epkg:`_PredictScorer` with
        function :epkg:`make_scorer`
    :param X: training set (at least one row),
        can be None, it is used to infered the
        input types (*initial_types*)
    :param initial_types: if *X* is None, then *initial_types*
        must be defined
    :param name: name of the produced model
    :param target_opset: to do it with a different target opset
    :param options: additional parameters for the conversion
    :param rewrite_ops: rewrites some existing converters,
        the changes are permanent
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline, if empty, all are allowed
    :param black_op: black list of ONNX nodes allowed
        while converting a pipeline, if empty,
        none are blacklisted
    :param final_types: a python list. Works the same way as
        initial_types but not mandatory, it is used
        to overwrites the type (if type is not None)
        and the name of every output.
    :param rename_strategy: rename any name in the graph, select shorter
        names, see @see fn onnx_rename_names
    :param verbose: display information while converting the model
    :param prefix_name: prefix for variable names
    :param run_shape: run shape inference on the final onnx model
    :param single_function: if True, the main graph only includes one node
        calling the main function
    :return: converted model
    """
    if rename_strategy is not None or rewrite_ops:
        return to_onnx(
            model, X=X, name=name, initial_types=initial_types,
            target_opset=target_opset, options=options, rewrite_ops=rewrite_ops,
            white_op=white_op, black_op=black_op, final_types=final_types,
            rename_strategy=rename_strategy, verbose=verbose,
            run_shape=run_shape)

    logger.debug("to_onnx_function:begin:(%s-%d, X=%r, initial_types=%r, target_opset=%r, "
                 "options=%r, rewrite_ops=%r, white_op=%r, black_op=%r, "
                 "final_types=%r)",
                 model.__class__.__name__, id(model), type(X), initial_types,
                 target_opset, options, rewrite_ops, white_op, black_op,
                 final_types)

    if final_types is not None:
        raise NotImplementedError(  # pragma: no cover
            "final_types != None, not implemented yet.")

    if single_function and (not isinstance(model, Pipeline) or
                            len(model.steps) != 1):
        # Wraps the model into a single pipeline.
        new_model = Pipeline(steps=[('main', model)])
        return to_onnx_function(
            new_model, X=X, name=name, initial_types=initial_types,
            target_opset=target_opset, options=options, rewrite_ops=rewrite_ops,
            white_op=white_op, black_op=black_op, final_types=final_types,
            rename_strategy=rename_strategy, verbose=verbose,
            prefix_name=prefix_name, run_shape=run_shape, single_function=False)

    if isinstance(model, Pipeline):
        return _to_onnx_function_pipeline(
            model, X=X, name=name, initial_types=initial_types,
            target_opset=target_opset, options=options, rewrite_ops=rewrite_ops,
            white_op=white_op, black_op=black_op, final_types=final_types,
            rename_strategy=rename_strategy, verbose=verbose,
            prefix_name=prefix_name, run_shape=run_shape,
            single_function=single_function)

    if isinstance(model, ColumnTransformer):
        return _to_onnx_function_column_transformer(
            model, X=X, name=name, initial_types=initial_types,
            target_opset=target_opset, options=options, rewrite_ops=rewrite_ops,
            white_op=white_op, black_op=black_op, final_types=final_types,
            rename_strategy=rename_strategy, verbose=verbose,
            prefix_name=prefix_name, run_shape=run_shape,
            single_function=single_function)

    raise TypeError(  # pragma: no cover
        f"Unexpected type {type(model)!r} for model to convert.")
