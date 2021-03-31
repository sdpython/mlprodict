"""
@file
@brief Implements :epkg:`numpy` functions with onnx and a runtime.

.. versionadded:: 0.6
"""
import inspect
from typing import Any
import numpy
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx import __max_supported_opset__
from ..onnxrt import OnnxInference
from .onnx_version import FctVersion
from .onnx_numpy_annotation import get_args_kwargs
from .onnx_variable import OnnxVar


class OnnxNumpyFunction:
    """
    Class wrapping a function build with
    @see cl OnnxNumpyCompiler.

    .. versionadded:: 0.6
    """

    def __init__(self, compiler, rt, inputs, outputs,
                 n_optional, n_variables):
        self.compiler = compiler
        self.inputs = inputs
        self.outputs = outputs
        self.rt = rt
        self.n_optional = n_optional
        self.n_variables = n_variables
        if n_optional < 0:
            raise RuntimeError(  # pragma: no cover
                "Wrong configuration, n_optional %r must be >= 0."
                "" % n_optional)
        if n_optional >= len(inputs):
            raise RuntimeError(  # pragma: no cover
                "Wrong configuration, n_optional %r must be >= %r "
                "the number of inputs." % (n_optional, len(inputs)))

    def _check_(self, *args, **kwargs):
        if self.n_variables > 0:
            return
        if (len(args) < len(self.inputs) - self.n_optional or
                len(args) > len(self.inputs)):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of inputs %d. It should be in "
                "[%r, %r] len(args)=%d n_optional=%d n_variables=%d"
                "\nargs=%s\nkwargs=%s\ninputs=%s" % (
                    len(args), len(self.inputs) - self.n_optional,
                    len(args), self.n_optional, self.n_variables,
                    len(self.inputs), args, kwargs, self.inputs))


class OnnxNumpyFunctionOnnxInference(OnnxNumpyFunction):
    """
    Overwrites @see cl OnnxNumpyFunction to run an instance of
    @see cl OnnxInference.

    .. versionadded:: 0.6
    """

    def __call__(self, *args, **kwargs):
        self._check_(*args, **kwargs)
        inp = {k[0]: a for k, a in zip(self.inputs, args)}
        out = self.rt.run(inp, **kwargs)
        if len(out) != len(self.outputs):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of outputs %d instead of %d." % (
                    len(out), len(self.outputs)))
        return tuple([out[o[0]] for o in self.outputs])


class OnnxNumpyFunctionInferenceSession(OnnxNumpyFunction):
    """
    Overwrites @see cl OnnxNumpyFunction to run an instance of
    `InferenceSession` from :epkg:`onnxruntime`.

    .. versionadded:: 0.6
    """

    def __call__(self, *args, **kwargs):
        self._check_(*args, **kwargs)
        if len(kwargs) > 0:
            raise RuntimeError(  # pragma: no cover
                "kwargs is not used but it is not empty: %r." % kwargs)
        inp = {k[0]: a for k, a in zip(self.inputs, args)}
        out = self.rt.run(None, inp)

        if len(out) != len(self.outputs):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of outputs %d instead of %d." % (
                    len(out), len(self.outputs)))
        return tuple(out)


class OnnxNumpyCompiler:
    """
    Implements a class which runs onnx graph.

    :param fct: a function with annotations which returns an ONNX graph,
        it can also be an ONNX graph.
    :param op_version: :epkg:`ONNX` opset to use, None
        for the latest one
    :param runtime: runtime to choose to execute the onnx graph,
        `python`, `onnxruntime`, `onnxruntime1`
    :param signature: used when the function is not annotated
    :param version: the same function can be instantiated with
        different type, this parameter is None or a numpy type
        if the signature allows multiple types, it must an instance
        of type @see cl FctVersion
    :param fctsig: function used to overwrite the fct signature
        in case this one is using `*args, **kwargs`

    .. versionadded:: 0.6
    """

    def __init__(self, fct, op_version=None, runtime=None, signature=None,
                 version=None, fctsig=None):
        if version is not None and not isinstance(version, FctVersion):
            raise TypeError(
                "version must be of Type 'FctVersion' not %s - %s"
                "." % (type(version), version))
        self.fctsig = fctsig
        if op_version is None:
            op_version = __max_supported_opset__
        if hasattr(fct, 'SerializeToString'):
            self.fct_ = None
            self.onnx_ = fct
        else:
            self.fct_ = fct
            if not inspect.isfunction(fct):
                raise TypeError(  # pragma: no cover
                    "Unexpected type for fct=%r, it must be "
                    "function." % type(fct))
            self.onnx_ = None
            self.onnx_ = self._to_onnx(
                op_version=op_version, signature=signature,
                version=version)
        self.runtime_ = self._build_runtime(
            op_version=op_version, runtime=runtime,
            signature=signature, version=version)
        ann = self._parse_annotation(signature=signature, version=version)
        inputs, outputs, kwargs, n_optional, n_variables = ann
        n_opt = 0 if signature is None else signature.n_optional
        args, kwargs2 = get_args_kwargs(self.fctsig or self.fct_, n_opt)
        self.meta_ = dict(op_version=op_version, runtime=runtime,
                          signature=signature, version=version,
                          inputs=inputs, outputs=outputs,
                          kwargs=kwargs, n_optional=n_optional,
                          n_variables=n_variables,
                          args=args, kwargs2=kwargs2,
                          annotations=self.fct_.__annotations__)

    def __getstate__(self):
        """
        Serializes everything but function `fct_`.
        Function `fct_` is used to build the onnx graph
        and is not needed anymore.
        """
        return dict(onnx_=self.onnx_, meta_=self.meta_)

    def __setstate__(self, state):
        """
        Restores serialized data.
        """
        for k, v in state.items():
            setattr(self, k, v)
        self.runtime_ = self._build_runtime(
            op_version=self.meta_['op_version'],
            runtime=self.meta_['runtime'],
            signature=self.meta_['signature'],
            version=self.meta_['version'])

    def __repr__(self):
        "usual"
        if self.fct_ is not None:
            return "%s(%s)" % (self.__class__.__name__, repr(self.fct_))
        if self.onnx_ is not None:
            return "%s(%s)" % (self.__class__.__name__, "... ONNX ... ")
        raise NotImplementedError(
            "fct_ and onnx_ are empty.")

    def _to_onnx_shape(self, shape):
        if shape is Any or shape is Ellipsis:
            shape = None
        elif isinstance(shape, tuple):
            shape = [None if s is Any or s is Ellipsis else s
                     for s in shape]
        else:
            raise RuntimeError(  # pragma: no cover
                "Unexpected annotated shape %r." % shape)
        return shape

    def _to_onnx_dtype(self, dtype, shape):
        from skl2onnx.common.data_types import _guess_numpy_type
        return _guess_numpy_type(dtype, shape)

    def _parse_annotation(self, signature, version):
        """
        Returns the annotations for function `fct_`.

        :param signature: needed if the annotation is missing,
            then version might be needed to specify which type
            to use if the signature allows many
        :param version: version inside the many signatures possible
        :return: *tuple(inputs, outputs, kwargs)*, each of them
            is a list of tuple with the name and the dtype,
            *kwargs* is the list of additional parameters
        """
        n_opt = 0 if signature is None else signature.n_optional
        if hasattr(self, 'meta_'):
            args, kwargs = self.meta_['args'], self.meta_['kwargs2']
        else:
            args, kwargs = get_args_kwargs(self.fctsig or self.fct_, n_opt)
        if version is not None:
            nv = len(version) - len(args) - n_opt
            if (signature is not None and not
                    signature.n_variables and nv > len(kwargs)):
                raise RuntimeError(  # pragma: no cover
                    "Mismatch (%d - %d - %d ? %d) between version=%r and kwargs=%r for "
                    "function %r, optional argument is %d, "
                    "signature=%r." % (
                        len(version), len(args), n_opt, len(kwargs),
                        version, kwargs, self.fct_,
                        signature.n_variables, signature))
            vvers = {} if version.kwargs is None else version.kwargs
            up = {}
            for k, v in zip(kwargs, vvers):
                up[k] = v
            kwargs = kwargs.copy()
            kwargs.update(up)

        for k, v in kwargs.items():
            if isinstance(v, (type, numpy.dtype)):
                raise RuntimeError(  # pragma: no cover
                    "Unexpected value for argument %r: %r from %r." % (
                        k, v, kwargs))

        if signature is not None:
            inputs, kwargs, outputs, n_optional, n_variables = (
                signature.get_inputs_outputs(args, kwargs, version))
            return inputs, outputs, kwargs, n_optional, n_variables

        def _possible_names():
            yield 'y'
            yield 'z'  # pragma: no cover
            yield 'o'  # pragma: no cover
            for i in range(0, 10000):  # pragma: no cover
                yield 'o%d' % i

        if hasattr(self, 'meta_'):
            annotations = self.meta_['annotations']
        else:
            annotations = self.fct_.__annotations__
        inputs = []
        outputs = []
        for a in args:
            if a == "op_version":
                continue
            if a not in annotations:
                raise RuntimeError(  # pragma: no cover
                    "Unable to find annotation for argument %r. "
                    "You should annotate the arguments and the results "
                    "or specify a signature." % a)
            ann = annotations[a]
            shape, dtype = ann.__args__
            shape = self._to_onnx_shape(shape)
            dtype = self._to_onnx_dtype(dtype, shape)
            inputs.append((a, dtype))

        ret = annotations['return']
        names_in = set(inp[0] for inp in inputs)

        if isinstance(ret, tuple):
            # multiple outputs
            names_none = set()
            for shape_dtype in ret:
                shape, dtype = shape_dtype.__args__
                shape = self._to_onnx_shape(shape)
                dtype = self._to_onnx_dtype(dtype, shape)
                name_out = None
                for name in _possible_names():
                    if name not in names_in and name not in names_none:
                        name_out = name
                        break
                outputs.append((name_out, dtype))
                names_none.add(name_out)
            return (inputs, outputs, kwargs, 0,
                    signature.n_variables if signature is not None else False)

        # single outputs
        shape, dtype = ret.__args__
        shape = self._to_onnx_shape(shape)
        dtype = self._to_onnx_dtype(dtype, shape)
        name_out = None
        for name in _possible_names():
            if name not in names_in:
                name_out = name
                break
        outputs.append((name_out, dtype))
        return (inputs, outputs, kwargs, 0,
                signature.n_variables if signature is not None else False)

    def _to_onnx(self, op_version=None, signature=None, version=None):
        """
        Returns the onnx graph produced by function `fct_`.
        """
        if self.onnx_ is None and self.fct_ is not None:
            inputs, outputs, kwargs, n_optional, n_variables = (  # pylint: disable=W0612
                self._parse_annotation(
                    signature=signature, version=version))
            if ((signature is None or not signature.n_variables) and
                    isinstance(version, tuple) and
                    len(inputs) > len(version)):
                raise NotImplementedError(  # pragma: no cover
                    "Mismatch between additional parameters %r "
                    "(n_optional=%r) and version %r for function %r from %r."
                    "" % (kwargs, n_optional, version, self.fct_,
                          getattr(self.fct_, '__module__', None)))
            names_in = [oi[0] for oi in inputs]
            names_out = [oi[0] for oi in outputs]
            names_var = [OnnxVar(n, dtype=guess_numpy_type(dt[1]))
                         for n, dt in zip(names_in, inputs)]

            if 'op_version' in self.fct_.__code__.co_varnames:
                onx_algebra = self.fct_(
                    *names_in, op_version=op_version, **kwargs)
            else:
                onx_var = self.fct_(*names_var, **kwargs)
                if not hasattr(onx_var, 'to_algebra'):
                    raise TypeError(  # pragma: no cover
                        "The function %r to convert must return an instance of "
                        "OnnxVar but returns type %r." % (self.fct_, type(onx_var)))
                onx_algebra = onx_var.to_algebra(op_version=op_version)

            if isinstance(onx_algebra, str):
                raise RuntimeError(  # pragma: no cover
                    "Unexpected str type %r." % onx_algebra)
            if isinstance(onx_algebra, tuple):
                raise NotImplementedError(  # pragma: no cover
                    "Not implemented when the function returns multiple results.")
            if hasattr(onx_algebra, 'to_onnx'):
                # skl2onnx algebra
                onx_algebra.output_names = names_out
                onx = onx_algebra.to_onnx(inputs=inputs,
                                          target_opset=op_version,
                                          outputs=outputs)
                self.onnx_ = onx

        if self.onnx_ is None:
            raise RuntimeError(  # pragma: no cover
                "Unable to get the ONNX graph (class %r, fct_=%r)" % (
                    type(self), self.fct_))
        return self.onnx_

    def _build_runtime(self, op_version=None, runtime=None,
                       signature=None, version=None):
        """
        Creates the runtime for the :epkg:`ONNX` graph.

        :param op_version: :epkg:`ONNX` opset to use, None
            for the latest one
        :param runtime: runtime to choose to execute the onnx graph,
            `python`, `onnxruntime`, `onnxruntime1`
        :param signature: used when the function is not annotated
        """
        onx = self._to_onnx(op_version=op_version, signature=signature,
                            version=version)
        inputs, outputs, _, n_optional, n_variables = self._parse_annotation(
            signature=signature, version=version)
        if runtime != 'onnxruntime':
            rt = OnnxInference(onx, runtime=runtime)
            self.rt_fct_ = OnnxNumpyFunctionOnnxInference(
                self, rt, inputs=inputs, outputs=outputs,
                n_optional=n_optional, n_variables=n_variables)
        else:
            from onnxruntime import InferenceSession
            rt = InferenceSession(onx.SerializeToString())
            self.rt_fct_ = OnnxNumpyFunctionInferenceSession(
                self, rt, inputs=inputs, outputs=outputs,
                n_optional=n_optional, n_variables=n_variables)
        return self.rt_fct_

    def __call__(self, *args, **kwargs):
        """
        Executes the function and returns the results.

        :param args: arguments
        :return: results
        """
        res = self.rt_fct_(*args, **kwargs)
        if len(res) == 1:
            return res[0]
        return res
