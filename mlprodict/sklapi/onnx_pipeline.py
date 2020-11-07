"""
@file
@brief A pipeline which serializes into ONNX steps by steps.
"""
import numpy
from sklearn.base import clone
from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils.validation import check_memory
from sklearn.utils import _print_elapsed_time
from ..onnx_conv import to_onnx
from .onnx_transformer import OnnxTransformer


class OnnxPipeline(Pipeline):
    """
    The pipeline overwrites method *fit*, it trains and converts
    every steps into ONNX before training the next step
    in order to minimize discrepencies. By default,
    ONNX is using float and not double which is the default
    for :epkg:`scikit-learn`. It may introduce discrepencies
    when a non-continuous model (mathematical definition) such
    as tree ensemble and part of the pipeline.

    :param steps:
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    :param memory: str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    :param verbose: bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.
    :param output_name: string
        requested output name or None to request all and
        have method *transform* to store all of them in a dataframe
    :param enforce_float32: boolean
        :epkg:`onnxruntime` only supports *float32*,
        :epkg:`scikit-learn` usually uses double floats, this parameter
        ensures that every array of double floats is converted into
        single floats
    :param runtime: string, defined the runtime to use
        as described in @see cl OnnxInference.
    :param options: see @see fn to_onnx
    :param white_op: see @see fn to_onnx
    :param black_op: see @see fn to_onnx
    :param final_types: see @see fn to_onnx
    :param op_version: ONNX targeted opset

    The class stores transformers before converting them into ONNX
    in attributes ``raw_steps_``.

    See notebook :ref:`onnxdiscrepenciesrst` to see it can
    be used to reduce discrepencies after it was converted into
    *ONNX*.
    """

    def __init__(self, steps, *, memory=None, verbose=False,
                 output_name=None, enforce_float32=True,
                 runtime='python', options=None,
                 white_op=None, black_op=None, final_types=None,
                 op_version=None):
        self.output_name = output_name
        self.enforce_float32 = enforce_float32
        self.runtime = runtime
        self.options = options
        self.white_op = white_op
        self.white_op = white_op
        self.black_op = black_op
        self.final_types = final_types
        self.op_version = op_version
        # The constructor calls _validate_step and it checks the value
        # of black_op.
        Pipeline.__init__(
            self, steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        """
        Fits the model, fits all the transforms one after the
        other and transform the data, then fit the transformed
        data using the final estimator.

        :param X: iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        :param y: iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        :param fit_params: dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        :return: self, Pipeline, this estimator
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time('OnnxPipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        if hasattr(self, 'raw_steps_') and self.raw_steps_ is not None:  # pylint: disable=E0203
            # Let's reuse the previous training.
            self.steps = list(self.raw_steps_)  # pylint: disable=E0203
            self.raw_steps_ = list(self.raw_steps_)
        else:
            self.steps = list(self.steps)
            self.raw_steps_ = list(self.steps)

        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)

            # Fit or load from cache the current transformer
            x_train = X
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, X, y, None,
                message_clsname='Pipeline',
                message=self._log_message(step_idx),
                **fit_params_steps[name])
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.raw_steps_[step_idx] = (name, fitted_transformer)
            self.steps[step_idx] = (
                name, self._to_onnx(name, fitted_transformer, x_train))
        return X

    def _to_onnx(self, name, fitted_transformer, x_train):
        """
        Converts a transformer into ONNX.

        @param  name                model name
        @param  fitted_transformer  fitted transformer
        @param  x_train             training dataset
        @return                     corresponding @see cl OnnxTransformer
        """
        if not isinstance(x_train, numpy.ndarray):
            raise RuntimeError(  # pragma: no cover
                "The pipeline only handle numpy arrays not {}.".format(
                    type(x_train)))
        atts = {'options', 'white_op', 'black_op', 'final_types'}
        kwargs = {k: getattr(self, k) for k in atts}
        if self.enforce_float32 or x_train.dtype != numpy.float64:
            x_train = x_train.astype(numpy.float32)
        if 'options' in kwargs:
            kwargs['options'] = self._preprocess_options(
                name, kwargs['options'])
        kwargs['target_opset'] = self.op_version
        onx = to_onnx(fitted_transformer, x_train, **kwargs)
        tr = OnnxTransformer(
            onx.SerializeToString(), output_name=self.output_name,
            enforce_float32=self.enforce_float32, runtime=self.runtime)
        return tr.fit()

    def _preprocess_options(self, name, options):
        """
        Preprocesses the options.

        @param      name        option name
        @param      options     conversion options
        @return                 new options
        """
        if options is None:
            return None
        prefix = name + '__'
        new_options = {}
        for k, v in options.items():
            if isinstance(k, str):
                if k.startswith(prefix):
                    new_options[k[len(prefix):]] = v
            else:
                new_options[k] = v
        return new_options
