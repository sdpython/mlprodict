"""
@file
@brief OnnxInferenceNode definition.
"""
import sys
import pprint
import numpy
from onnx import onnx_pb as onnx_proto
from onnx.onnx_cpp2py_export.defs import SchemaError  # pylint: disable=E0401,E0611
from ..onnx_tools.onnx2py_helper import get_onnx_schema
from .excs import MissingOperatorError
from .ops import load_op


class OnnxInferenceNode:
    """
    A node to execute.
    """
    class OnnxInferenceWrapper:
        """
        Wraps @see cl OnnxInference in a wrapper and exposes
        the necessary function.

        :param oinf: instance of @see cl OnnxInference
        """

        def __init__(self, oinf):
            if oinf is None:
                raise ValueError(  # pragma: no cover
                    "oinf cannot be None.")
            self.oinf = oinf

        @property
        def args_default(self):
            "Returns the list of default arguments."
            return []

        @property
        def args_default_modified(self):
            "Returns the list of modified arguments."
            return []

        @property
        def args_mandatory(self):
            "Returns the list of mandatory arguments."
            return self.oinf.input_names

        @property
        def args_optional(self):
            "Returns the list of optional arguments."
            return []

        @property
        def obj(self):
            "Returns the ONNX graph."
            return self.oinf.obj

        def run(self, *args, **kwargs):
            "Calls run."
            return self.oinf.run(*args, **kwargs)

        def to_python(self, inputs, *args, **kwargs):
            "Calls to_python."
            res = self.oinf.to_python(*args, **kwargs)
            if len(res) != 1:
                raise NotImplementedError(  # pragma: no cover
                    "Not implemented if the code has multiple files.")
            keys = list(res)
            value = res[keys[0]]
            lines = value.split('\n')
            last = 0
            for i, line in enumerate(lines):
                if line.startswith('def '):
                    last = i - 1
                    break
            imports = '\n'.join(
                line for line in lines[:last] if 'import ' in line)
            lines.append('')
            lines.append("return OnnxPythonInference().run(%s)" %
                         ', '.join(inputs))
            code = '\n'.join(lines[last:])
            return imports, code

        def need_context(self):
            "Needs context?"
            return False

        def infer_types(self, *args):
            "Calls infer_types."
            res = self.oinf.infer_types(args)
            names = self.oinf.obj.output
            dtypes = [res[n] for n in names]
            return tuple(dtypes)

        def infer_sizes(self, *args):
            "Calls infer_sizes."
            values = {name: value
                      for name, value in zip(self.oinf.input_names, args)}
            res = self.oinf.infer_sizes(values)
            names = self.oinf.obj.output
            sizes = [res.get(n, 0) for n in names]
            return (res['#'], ) + tuple(sizes)

        def enable_inplace_compute(self, index):
            "Not implemented."
            pass

    def __init__(self, onnx_node, desc, global_index):
        """
        @param      onnx_node       onnx_node
        @param      desc            internal description
        @param      global_index    it is a function which returns a unique index
                                    for the output this operator generates
        """
        if desc is None:
            raise ValueError("desc should not be None.")  # pragma: no cover
        self.desc = desc
        self.onnx_node = onnx_node
        self._init(global_index)

    @property
    def name(self):
        "Returns the ONNX name."
        return "_".join(
            [self.desc['domain'], self.onnx_node.op_type]).replace(
                ".", "_").replace('__', '_').strip('_')

    def _init(self, global_index):
        """
        Prepares the node.
        """
        self.op_type = self.onnx_node.op_type
        self.order = -1
        self.variable_to_clean = []
        self.inputs = list(self.onnx_node.input)
        self.outputs = list(self.onnx_node.output)
        self.inplaces = []
        self.inputs_indices = [global_index(name) for name in self.inputs]
        self.outputs_indices = [global_index(name) for name in self.outputs]
        self._global_index = global_index

    def set_order(self, order):
        """
        Defines the order of execution.
        """
        self.order = order

    def add_variable_to_clean(self, name):
        """
        Adds a variable which can be cleaned after the node
        execution.
        """
        self.variable_to_clean.append(name)

    def __str__(self):
        "usual"
        return "Onnx-{}({}) -> {}{}".format(
            self.op_type, ", ".join(self.inputs), ", ".join(self.outputs),
            "    (name=%r)" % self.onnx_node.name
            if self.onnx_node.name else "")

    def __repr__(self):
        "usual"
        return self.__str__()

    def setup_runtime(self, runtime=None, variables=None, rt_class=None,
                      target_opset=None, dtype=None, domain=None,
                      ir_version=None, runtime_options=None,
                      build_inference_node_function=None):
        """
        Loads runtime.

        :param runtime: runtime options
        :param variables: registered variables created by previous operators
        :param rt_class: runtime class used to compute
            prediction of subgraphs
        :param target_opset: use a specific target opset
        :param dtype: float computational type
        :param domain: node domain
        :param ir_version: if not None, changes the default value
            given by :epkg:`ONNX`
        :param runtime_options: runtime options
        :param build_inference_node_function: function creating an inference
            runtime from an ONNX graph

        .. versionchanged:: 0.9
            Parameter *build_inference_node_function* was added.
        """
        if self.desc is None:
            raise AttributeError(
                "desc should not be None.")  # pragma: no cover
        if rt_class is None:
            # path used when this operator is a function.
            self.function_ = OnnxInferenceNode.OnnxInferenceWrapper(runtime)
            self.ops_ = None
        else:
            self.function_ = None
            self.preprocess_parameters(
                runtime, rt_class, ir_version=ir_version,
                target_opset=target_opset)
            options = {'provider': runtime} if runtime else {}
            if domain is not None:
                options['domain'] = domain
            if target_opset is not None:
                options['target_opset'] = target_opset
            if ir_version is not None:
                options['ir_version'] = ir_version
            if runtime_options is not None:
                options.update({
                    k: v for k, v in runtime_options.items()
                    if k not in {'log_severity_level'}})
            try:
                if runtime == 'onnxruntime2':
                    self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                        options=options if options else None,
                                        variables=variables, dtype=dtype)
                elif runtime in ('python_compiled', 'python_compiled_debug'):
                    options['provider'] = 'python'
                    self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                        options=options if options else None,
                                        variables=variables, dtype=dtype)
                else:
                    self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                        options=options if options else None,
                                        variables=variables, dtype=dtype)
            except MissingOperatorError as e:
                try:
                    onnx_schema = get_onnx_schema(
                        self.onnx_node.op_type, self.onnx_node.domain,
                        opset=target_opset)
                except SchemaError:
                    raise e  # pylint: disable=W0707
                if onnx_schema is None or not onnx_schema.has_function:
                    raise e
                self.function_ = OnnxInferenceNode.OnnxInferenceWrapper(
                    build_inference_node_function(onnx_schema.function_body))
                self.ops_ = None

    @staticmethod
    def _find_static_inputs(body):
        """
        Determines the loop inputs. It is any defined inputs
        by the subgraphs + any results used as a constant
        in the subgraphs.
        """
        inputs_set = set(i.name for i in body.input)
        for init in body.initializer:
            inputs_set.add(init.name)
        for node in body.node:
            for i in node.output:
                inputs_set.add(i)
        add_inputs = []
        for node in body.node:
            for i in node.input:
                if i not in inputs_set:
                    #  no graph input or output node matches
                    # it must be a constant from the below graph
                    add_inputs.append(i)
                    inputs_set.add(i)
        return add_inputs

    def preprocess_parameters(self, runtime, rt_class, ir_version=None,
                              target_opset=None):
        """
        Preprocesses the parameters, loads *GraphProto*
        (equivalent to :epkg:`ONNX` graph with less metadata).

        @param  runtime         runtime options
        @param  rt_class        runtime class used to compute
                                prediction of subgraphs
        @param  ir_version      if not None, overwrites the default value
        @param  target_opset    use a specific target opset
        """
        if 'atts' not in self.desc:
            return  # pragma: no cover
        inside_loop = self.onnx_node.op_type in {'Loop'}
        for _, v in self.desc['atts'].items():
            if 'value' not in v:
                continue  # pragma: no cover
            value = v['value']
            if isinstance(value, onnx_proto.GraphProto):
                static_inputs = OnnxInferenceNode._find_static_inputs(value)
                try:
                    sess = rt_class(v['value'], runtime=runtime,
                                    ir_version=ir_version,
                                    target_opset=target_opset,
                                    inside_loop=inside_loop,
                                    static_inputs=static_inputs)
                except RuntimeError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to instantiate a node of type %r and name %r."
                        "" % (self.onnx_node.op_type, self.onnx_node.name)) from e
                v['value_rt'] = sess

    def run(self, values):
        """
        Runs the node.
        the function updates values with outputs.

        @param      values      list of existing values
        """
        if self.ops_ is None:
            # Then a function.
            feeds = {name: val
                     for name, val in zip(self.function_.obj.input, values)}
            outputs = self.function_.run(feeds)
            res = [outputs[k] for k in self.function_.obj.output]

            if self.outputs_indices is None:
                for name, value in zip(self.outputs, res):
                    values[name] = value
            else:
                for i, r in enumerate(res):
                    values[self.outputs_indices[i]] = r
            return

        # This code takes time if the graph contains many nodes.
        # Maybe a C++ container would help in that case (to skip GIL).
        if self.inputs_indices is None:
            args = list(values[k] for k in self.inputs)
        else:
            args = list(values[k] for k in self.inputs_indices)
        try:
            if self.ops_.need_context():
                context = {n: values[self._global_index(n)]
                           for n in self.ops_.additional_inputs}
                res = self.ops_.run(*args, context=context)
            else:
                res = self.ops_.run(*args)
        except TypeError as e:
            raise RuntimeError(  # pragma: no cover
                "Unable to run operator %r, inputs=%r."
                "" % (type(self.ops_), self.inputs)) from e
        except OverflowError as e:
            raise RuntimeError(  # pragma: no cover
                "Unable to run operator %r, inputs=%r."
                "" % (type(self.ops_), self.inputs)) from e

        if not isinstance(res, tuple):
            raise RuntimeError(  # pragma: no cover
                "Results of operator %r should be a tuple." % type(self.ops_))
        if len(self.outputs) != len(res):
            raise RuntimeError(  # pragma: no cover
                "Mismatch number of outputs got {} for names {}.\n{}".format(
                    len(res), list(sorted(self.outputs)),
                    pprint.pformat(self.desc)))

        # This code takes times if the graph contains many nodes.
        # Maybe a C++ container would help in that case (to skip GIL).
        if self.outputs_indices is None:
            for name, value in zip(self.outputs, res):
                values[name] = value
        else:
            for i, r in enumerate(res):
                values[self.outputs_indices[i]] = r

    def switch_initializers_dtype(self, dtype_in=numpy.float32,
                                  dtype_out=numpy.float64):
        """
        Switches all initializers to ``numpy.float64``.
        This only works if the runtime is ``'python'``.

        @param      dtype_in    previous type
        @param      dtype_out   next type
        @return                 done operations
        """
        done = []
        for k, v in self.desc['atts'].items():
            if 'value_rt' not in v:
                continue
            if isinstance(v['value_rt'], numpy.ndarray):
                if v['value_rt'].dtype == dtype_in:
                    v['value_rt'] = v['value_rt'].astype(dtype_out)
                    done.append(("+", "desc", k, v['value_rt']))
                else:
                    done.append(("-", "desc", k, v['value_rt']))
        if hasattr(self, 'ops_') and self.ops_ is not None:
            res = self.ops_.switch_initializers_dtype(dtype_in, dtype_out)
            for r in res:
                done.append(("ops_", ) + r)
        return done

    def _set_shape_inference_runtime(self, values):
        """
        Updates *values* which shapes of the outputs.

        :param values: container for shapes
        """
        if self.ops_ is None:
            # A function, unknown types.
            for name in self.outputs:
                values[name] = None
            return values
        args = [values[k] for k in self.inputs if k != '']
        try:
            res = self.ops_.infer_shapes(*args)
        except (TypeError, ValueError, AttributeError) as e:  # pragma: no cover
            raise TypeError(
                "Unable to call infer_shapes with {} arguments for class"
                " '{}' ({})".format(
                    len(args), self.ops_.__class__.__name__,
                    self.ops_.infer_shapes)) from e
        if res is not None:
            if not isinstance(res, tuple):
                raise RuntimeError(  # pragma: no cover
                    "Results of an operator should be a tuple for operator "
                    "'{}'.".format(type(self.ops_)))
            if len(self.outputs) != len(res):
                raise RuntimeError(  # pragma: no cover
                    "Mismatch number of outputs got {} != {} for names {} "
                    "(node='{}').\n{}".format(
                        len(res), len(self.outputs), list(self.outputs),
                        self.ops_.__class__.__name__,
                        pprint.pformat(self.desc, depth=2)))
            for name, value in zip(self.outputs, res):
                values[name] = value
        return values

    def _set_type_inference_runtime(self, values):
        """
        Updates *values* which types of the outputs.

        :param values: container for types
        """
        args = [values[k] for k in self.inputs]
        if self.ops_ is None:
            res = self.function_.infer_types(*args)
        else:
            res = self.ops_.infer_types(*args)
        try:
            if self.ops_ is None:
                res = self.function_.infer_types(*args)
            else:
                res = self.ops_.infer_types(*args)
        except (TypeError, ValueError) as e:  # pragma: no cover
            raise TypeError(
                "Unable to call infer_types with {} arguments for class"
                " '{}'".format(
                    len(args), self.ops_.__class__.__name__)) from e
        if not isinstance(res, tuple):
            raise RuntimeError(  # pragma: no cover
                "Results of an operator should be a tuple for operator '{}'"
                ".".format(type(self.ops_)))
        if len(self.outputs) != len(res):
            raise RuntimeError(  # pragma: no cover
                "Mismatch number of outputs got {} != {} for names {} (node='{}')."
                "\n{}".format(
                    len(res), len(self.outputs), list(self.outputs),
                    self.ops_.__class__.__name__,
                    pprint.pformat(self.desc, depth=2)))
        for name, value in zip(self.outputs, res):
            values[name] = value
        return values

    def _set_size_inference_runtime(self, values):
        """
        Updates *values* which types of the outputs.

        :param values: container for sizes
        """
        args = [values[k] for k in self.inputs]
        try:
            if (self.ops_ or self.function_).need_context():
                context = {n: values[n]
                           for n in self.ops_.additional_inputs}
                res = self.ops_.infer_sizes(*args, context=context)
            else:
                res = (self.ops_ or self.function_).infer_sizes(*args)
        except (TypeError, ValueError) as e:  # pragma: no cover
            raise TypeError(
                "Unable to call infer_sizes with {} arguments for class"
                " '{}' ({})".format(len(args), self.ops_.__class__.__name__,
                                    self.ops_.infer_sizes)) from e
        if not isinstance(res, tuple):
            raise RuntimeError(  # pragma: no cover
                "Results of an operator should be a tuple for operator '{}'"
                ".".format(type(self.ops_)))
        if len(self.outputs) + 1 != len(res):
            raise RuntimeError(  # pragma: no cover
                "Mismatch number of outputs got {} != {} + 1 for names {} "
                "(node='{}').\n{}".format(
                    len(res), len(self.outputs), list(self.outputs),
                    self.ops_.__class__.__name__,
                    pprint.pformat(self.desc, depth=2)))
        for name, value in zip(self.outputs, res[1:]):
            values[name] = value
        values['#' + self.onnx_node.name] = res[0]
        return values

    def enable_inplace_compute(self, name):
        """
        Let the node know that one input can be overwritten.

        @param      name        input name
        """
        self.inplaces.append(name)
        (self.ops_ or self.function_).enable_inplace_compute(
            self.inputs.index(name))

    @property
    def inputs_args(self):
        """
        Returns the list of arguments as well as
        the list of parameters with the default values
        (close to the signature).
        """
        if not hasattr(self, 'ops_'):
            raise AttributeError(
                "Attribute 'ops_' is missing.")  # pragma: no cover
        sigs = []
        ops_or_function = self.function_ if self.ops_ is None else self.ops_
        mand = ops_or_function.args_mandatory
        if mand is None:
            mand = self.python_inputs
        sigs.extend(mand)
        if len(ops_or_function.args_optional) > 0:
            sigs.extend(ops_or_function.args_optional)
            if sys.version_info[:2] >= (3, 8):
                sigs.append('/')
        sigs.extend(ops_or_function.args_default)
        return sigs

    @property
    def python_inputs(self):
        """
        Returns the python arguments.
        """
        if not hasattr(self, 'ops_'):
            raise AttributeError(
                "Attribute 'ops_' is missing.")  # pragma: no cover
        if hasattr(self.ops_, 'python_inputs'):
            return self.ops_.python_inputs
        return self.inputs

    @property
    def modified_args(self):
        """
        Returns the list of modified parameters.
        """
        if not hasattr(self, 'ops_'):
            raise AttributeError(
                "Attribute 'ops_' is missing.")  # pragma: no cover
        if self.ops_ is None:
            return self.function_.args_default_modified
        return self.ops_.args_default_modified

    def to_python(self, inputs):
        """
        Returns a python code for this operator.

        @param      inputs      inputs name
        @return                 imports, python code, both as strings
        """
        if not hasattr(self, 'ops_'):
            raise AttributeError(
                "Attribute 'ops_' is missing.")  # pragma: no cover
        if self.ops_ is None:
            return self.function_.to_python(inputs)
        return self.ops_.to_python(inputs)
