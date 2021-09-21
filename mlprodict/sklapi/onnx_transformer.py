# coding: utf-8
"""
@file
@brief Wraps runtime into a :epkg:`scikit-learn` transformer.
"""
from io import BytesIO
import numpy
import pandas
import onnx
from sklearn.base import BaseEstimator, TransformerMixin
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx.helpers.onnx_helper import (
    load_onnx_model, enumerate_model_node_outputs)
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType,
    Int64TensorType)
from ..onnx_tools.onnx2py_helper import _var_as_dict, onnx_model_opsets
from ..onnx_tools.exports.skl2onnx_helper import add_onnx_graph
from ..onnxrt import OnnxInference


class OnnxTransformer(BaseEstimator, TransformerMixin, OnnxOperatorMixin):
    """
    Calls :epkg:`onnxruntime` or the runtime implemented
    in this package to transform input based on a ONNX graph.
    It follows :epkg:`scikit-learn` API
    so that it can be included in a :epkg:`scikit-learn` pipeline.
    See notebook :ref:`transferlearningrst` for an example.

    :param onnx_bytes: bytes
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
    :param change_batch_size: some models are converted for
        a specific batch size, this parameter changes it,
        None to avoid changing it, 0 to fix an undefined
        first dimension
    :param reshape: reshape the output to get
        a matrix and not a multidimensional array
    """

    def __init__(self, onnx_bytes, output_name=None, enforce_float32=True,
                 runtime='python', change_batch_size=None, reshape=False):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.onnx_bytes = (onnx_bytes
                           if not hasattr(onnx_bytes, 'SerializeToString')
                           else onnx_bytes.SerializeToString())
        self.output_name = output_name
        self.enforce_float32 = enforce_float32
        self.runtime = runtime
        self.change_batch_size = change_batch_size
        self.reshape = reshape

    def __repr__(self):  # pylint: disable=W0222
        """
        usual
        """
        ob = self.onnx_bytes
        if len(ob) > 20:
            ob = ob[:10] + b"..." + ob[-10:]
        return ("{0}(onnx_bytes={1}, output_name={2}, enforce_float32={3}, "
                "runtime='{4}')".format(
                    self.__class__.__name__, ob, self.output_name,
                    self.enforce_float32, self.runtime))

    def fit(self, X=None, y=None, **fit_params):
        """
        Loads the :epkg:`ONNX` model.

        :param X: unused
        :param y: unused
        :param fit_params: additional parameter (unused)
        :return: self
        """
        from ..onnx_tools.optim.onnx_helper import change_input_first_dimension
        onx = onnx.load(BytesIO(self.onnx_bytes))
        self.op_version = onnx_model_opsets(onx)

        output_names = set(
            o.name for o in onx.graph.output)  # pylint: disable=E1101
        updated = False
        if (self.output_name is not None and
                self.output_name not in output_names):
            # The model refers to intermediate outputs.
            onx = select_model_inputs_outputs(
                onx, outputs=[self.output_name])
            updated = True

        if self.change_batch_size is not None:
            onx = change_input_first_dimension(
                onx, self.change_batch_size)
            updated = True

        onnx_bytes = (
            onx.SerializeToString() if updated else self.onnx_bytes)
        self.onnxrt_ = OnnxInference(onnx_bytes, runtime=self.runtime)
        self.inputs_ = self.onnxrt_.input_names
        self.inputs_shape_types_ = self.onnxrt_.input_names_shapes_types
        return self

    def _check_arrays(self, inputs):
        """
        Ensures that double floats are converted into single floats
        if *enforce_float32* is True or raises an exception.
        """
        has = hasattr(self, "onnxrt_")
        sht = self.inputs_shape_types_ if has else None
        if sht is not None and len(sht) < len(inputs):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of inputs {} > {} (expected).".format(
                    len(inputs), len(sht)))
        for i, k in enumerate(inputs):
            v = inputs[k]
            if isinstance(v, numpy.ndarray):
                if v.dtype == numpy.float64 and self.enforce_float32:
                    inputs[k] = v.astype(numpy.float32)
                    continue
                if not has:
                    continue
                exp = sht[i]
                if exp[1] != ('?', ) and exp[1][1:] != v.shape[1:]:
                    raise RuntimeError(  # pragma: no cover
                        "Unexpected shape for input '{}': {} != {} "
                        "(expected).".format(
                            k, v.shape, exp[1]))
                if ((v.dtype == numpy.float32 and exp[2] != 'tensor(float)') or
                        (v.dtype == numpy.float64 and exp[2] != 'tensor(double)')):
                    raise TypeError(  # pragma: no cover
                        "Unexpected dtype for input '{}': {} != {} "
                        "(expected).".format(
                            k, v.dtype, exp[2]))

    def transform(self, X, y=None, **inputs):
        """
        Runs the predictions. If *X* is a dataframe,
        the function assumes every columns is a separate input,
        otherwise, *X* is considered as a first input and *inputs*
        can be used to specify extra inputs.

        :param X: iterable, data to process
            (or first input if several expected)
        :param y: unused
        :param inputs: :epkg:`ONNX` graph support multiple inputs,
            each column of a dataframe is converted into as many inputs if
            *X* is a dataframe, otherwise, *X* is considered as the first input
            and *inputs* can be used to specify the other ones
        :return: :epkg:`DataFrame`
        """
        if not hasattr(self, "onnxrt_"):
            raise AttributeError(  # pragma: no cover
                "Transform OnnxTransformer must be fit first.")
        rt_inputs = {}
        if isinstance(X, numpy.ndarray):
            rt_inputs[self.inputs_[0]] = X
        elif isinstance(X, pandas.DataFrame):
            for c in X.columns:
                rt_inputs[c] = X[c]
        elif isinstance(X, dict) and len(inputs) == 0:
            for k, v in X.items():
                rt_inputs[k] = v
        elif isinstance(X, list):
            if len(self.inputs_) == 1:
                rt_inputs[self.inputs_[0]] = numpy.array(X)
            else:
                for i in range(len(self.inputs_)):  # pylint: disable=C0200
                    rt_inputs[self.inputs_[i]] = [row[i] for row in X]

        for k, v in inputs.items():
            rt_inputs[k] = v

        names = ([self.output_name]
                 if self.output_name else self.onnxrt_.output_names)
        self._check_arrays(rt_inputs)
        doutputs = self.onnxrt_.run(rt_inputs)
        outputs = [doutputs[n] for n in names]

        if self.reshape:
            n = outputs[0].shape[0]
            outputs = [o.reshape((n, -1)) for o in outputs]

        if self.output_name or len(outputs) == 1:
            if isinstance(outputs[0], list):
                return pandas.DataFrame(outputs[0])
            return outputs[0]

        names = self.output_name if self.output_name else [
            o for o in self.onnxrt_.output_names]
        concat = []
        colnames = []
        for k, v in zip(names, outputs):
            if isinstance(v, numpy.ndarray):
                if len(v.shape) == 1:
                    v = v.reshape((-1, 1))
                    colnames.append(k)
                elif len(v.shape) == 2:
                    colnames.extend("%s%d" % (k, i) for i in range(v.shape[1]))
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unexpected shape for results %r: %r." % (k, v.shape))
            if isinstance(v, list):
                if len(v) == 0:
                    raise RuntimeError(  # pragma: no cover
                        "Output %r is empty." % k)
                if not isinstance(v[0], dict):
                    raise RuntimeError(  # pragma: no cover
                        "Unexpected type for output %r - value=%r."
                        "" % (k, v[0]))
                df = pandas.DataFrame(v)
                cols = list(sorted(df.columns))
                v = df[cols].copy().values
                colnames.extend("%s%d" % (k, i) for i in range(v.shape[1]))
            concat.append(v)
        res = numpy.hstack(concat)
        return pandas.DataFrame(res, columns=colnames)

    def fit_transform(self, X, y=None, **inputs):
        """
        Loads the *ONNX* model and runs the predictions.

        :param X: iterable, data to process
            (or first input if several expected)
        :param y: unused
        :param inputs: :epkg:`ONNX` graph support multiple inputs,
            each column of a dataframe is converted into as many inputs if
            *X* is a dataframe, otherwise, *X* is considered as the first input
            and *inputs* can be used to specify the other ones
        :return: :epkg:`DataFrame`
        """
        return self.fit(X, y=y, **inputs).transform(X, y)

    @staticmethod
    def enumerate_create(onnx_bytes, output_names=None, enforce_float32=True):
        """
        Creates multiple *OnnxTransformer*,
        one for each requested intermediate node.

        onnx_bytes : bytes
        output_names: string
            requested output names or None to request all and
            have method *transform* to store all of them in a dataframe
        enforce_float32 : boolean
            :epkg:`onnxruntime` only supports *float32*,
            :epkg:`scikit-learn` usually uses double floats, this parameter
            ensures that every array of double floats is converted into
            single floats
        :return: iterator on OnnxTransformer *('output name', OnnxTransformer)*
        """
        selected = None if output_names is None else set(output_names)
        model = load_onnx_model(onnx_bytes)
        for out in enumerate_model_node_outputs(model):
            m = select_model_inputs_outputs(model, out)
            if selected is None or out in selected:
                tr = OnnxTransformer(m.SerializeToString(),
                                     enforce_float32=enforce_float32)
                yield out, tr

    def onnx_parser(self):
        """
        Returns a parser for this model.
        """
        def parser(scope=None, inputs=None):
            if scope is None:
                raise RuntimeError(
                    "scope cannot be None (parser of class %r)."
                    "" % type(self))
            if inputs is None:
                raise RuntimeError(
                    "inputs cannot be None (parser of class %r)."
                    "" % type(self))
            if (not hasattr(self, 'onnxrt_') or
                    not hasattr(self.onnxrt_, 'output_names')):
                raise RuntimeError(  # pragma: no cover
                    'OnnxTransformer not fit.')
            if len(inputs) != len(self.inputs_):
                raise RuntimeError(  # pragma: no cover
                    "Mismatch between the number of inputs, expected %r, "
                    "got %r." % (self.inputs_, inputs))
            return self.onnxrt_.output_names
        return parser

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            cout = self.onnxrt_.output_names
            if len(operator.outputs) != len(cout):
                raise RuntimeError(  # pragma: no cover
                    "Mismatched number of outputs: {} != {}."
                    "".format(len(operator.outputs), len(cout)))
            for out_op, out in zip(operator.outputs, self.onnxrt_.obj.graph.output):
                var = _var_as_dict(out)
                if var['type']['kind'] != 'tensor':
                    raise NotImplementedError(  # pragma: no cover
                        "Noy yet implemented for output:\n{}".format(out))
                shape = var['type']['shape']
                if shape[0] == 0:
                    shape = (None,) + tuple(shape[1:])
                elem = var['type']['elem']
                if elem == 'float':
                    out_op.type = FloatTensorType(shape=shape)
                elif elem == 'int64':
                    out_op.type = Int64TensorType(shape=shape)
                elif elem == 'double':
                    out_op.type = DoubleTensorType(shape=shape)
                else:
                    raise NotImplementedError(  # pragma: no cover
                        "Not yet implemented for elem_type:\n{}".format(elem))
        return shape_calculator

    def onnx_converter(self):
        """
        Returns a converter for this model.
        If not overloaded, it fetches the converter
        mapped to the first *scikit-learn* parent
        it can find.
        """
        def converter(scope, operator, container, onnx_model=None):
            op = operator.raw_operator
            onx = onnx_model or op.onnxrt_.obj
            add_onnx_graph(scope, operator, container, onx)

        return converter

    @property
    def opsets(self):
        """
        Returns the opsets as dictionary ``{domain: opset}``.
        """
        if hasattr(self, 'onnxrt_'):
            model = self.onnxrt_.obj
        else:
            model = load_onnx_model(self.onnx_bytes)
        res = {}
        for oimp in model.opset_import:
            res[oimp.domain] = oimp.version
        return res
