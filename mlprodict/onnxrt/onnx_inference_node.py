"""
@file
@brief OnnxInferenceNode definition.
"""
import sys
import pprint
import numpy
from onnx import GraphProto, onnx_pb as onnx_proto
from onnx.onnx_cpp2py_export.defs import SchemaError  # pylint: disable=E0401,E0611
from ..onnx_tools.onnx2py_helper import get_onnx_schema
from .excs import MissingOperatorError
from .ops import load_op


class OnnxInferenceNode:
    """
    A node to execute.

    :param onnx_node: onnx_node
    :param desc: internal description
    :param global_index: it is a function which returns a unique index
        for the output this operator generates
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
            lines.append(
                f"return OnnxPythonInference().run({', '.join(inputs)})")
            code = '\n'.join(lines[last:])
            return imports, code

        def need_context(self):
            "Needs context?"
            return False

        def enable_inplace_compute(self, index):
            "Not implemented."
            pass

    def __init__(self, onnx_node, desc, global_index):
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
                      build_inference_node_function=None,
                      existing_functions=None):
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
        :param existing_functions: existing function as a dictionary
            `{ (domain, name): fct }`

        .. versionchanged:: 0.9
            Parameters *build_inference_node_function* and *existing_functions*
            were added.
        """
        if self.desc is None:
            raise AttributeError(
                "desc should not be None.")  # pragma: no cover
        if rt_class is None:
            # path used when this operator is a function.
            self.function_ = OnnxInferenceNode.OnnxInferenceWrapper(runtime)
            self.ops_ = None
            return

        self.function_ = None
        self.preprocess_parameters(
            runtime, rt_class, ir_version=ir_version,
            target_opset=target_opset, existing_functions=existing_functions)
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

        # existing functions?
        key = (self.onnx_node.domain, self.onnx_node.name)
        if existing_functions is not None and key in existing_functions:
            self.ops_ = existing_functions[key]
            return

        # regular node
        try:
            if runtime is not None and runtime.startswith('onnxruntime2'):
                self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                    options=options if options else None,
                                    variables=variables, dtype=dtype,
                                    runtime=runtime)
            elif runtime in ('python_compiled', 'python_compiled_debug'):
                options['provider'] = 'python'
                self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                    options=options if options else None,
                                    variables=variables, dtype=dtype,
                                    runtime=runtime)
            else:
                self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                    options=options if options else None,
                                    variables=variables, dtype=dtype,
                                    runtime=runtime)
        except MissingOperatorError as e:
            try:
                onnx_schema = get_onnx_schema(
                    self.onnx_node.op_type, self.onnx_node.domain,
                    opset=target_opset)
            except SchemaError:
                fct_names = (
                    list(existing_functions.keys()) if existing_functions
                    else [])
                raise MissingOperatorError(
                    "Unable to find runtime for node (%r, %r), "
                    "available functions=%r." % (
                        self.onnx_node.domain, self.onnx_node.op_type,
                        fct_names)) from e
            if onnx_schema is None or not onnx_schema.has_function:
                raise e
            self.function_ = OnnxInferenceNode.OnnxInferenceWrapper(
                build_inference_node_function(onnx_schema.function_body))
            self.ops_ = None

    @staticmethod
    def _find_static_inputs(body):
        """
        Determines the loop inputs. It is any defined inputs
        by the subgraphs + any result used as a constant
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
                    # no graph input or output node matches
                    # it must be a constant from the below graph
                    add_inputs.append(i)
                    inputs_set.add(i)
            for att in node.attribute:
                if (att.type == onnx_proto.AttributeProto.GRAPH and  # pylint: disable=E1101
                        hasattr(att, 'g') and att.g is not None):
                    inside = OnnxInferenceNode._find_static_inputs(att.g)
                    for i in inside:
                        if i not in inputs_set:
                            add_inputs.append(i)
                            inputs_set.add(i)
        # If there is no node, we add the outputs as well.
        if len(body.node) == 0:
            for o in body.output:
                i = o.name
                if i not in inputs_set:
                    add_inputs.append(i)
                    inputs_set.add(i)
        return add_inputs

    @staticmethod
    def _find_local_inputs(graph):
        """
        Determines the local inputs. It is any defined input
        used by the subgraph and defined in the parent graph.
        """
        if not isinstance(graph, GraphProto):
            raise TypeError(
                f"Unexpected type {type(graph)!r}.")
        local = set()
        known = set()
        for init in graph.initializer:
            known.add(init.name)
        for init in graph.input:
            known.add(init.name)
        for node in graph.node:
            for o in node.output:
                known.add(o)
            for i in node.input:
                if i not in known:
                    local.add(i)
        return list(local)

    def get_local_inputs(self):
        """
        Returns any local input used by this node in a subgraph
        defined as an attribute and not declared as an input of this subgraph.
        """
        req = set()
        for att in self.onnx_node.attribute:
            if hasattr(att, 'g') and att.g is not None:
                req |= set(self._find_local_inputs(att.g))
        return req

    def preprocess_parameters(self, runtime, rt_class, ir_version=None,
                              target_opset=None, existing_functions=None):
        """
        Preprocesses the parameters, loads *GraphProto*
        (equivalent to :epkg:`ONNX` graph with less metadata).

        :param runtime: runtime options
        :param rt_class: runtime class used to compute
            prediction of subgraphs
        :param ir_version: if not None, overwrites the default value
        :param target_opset: use a specific target opset
        :param existing_functions: existing functions
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
                if len(value.node) > 0:
                    try:
                        sess = rt_class(value, runtime=runtime,
                                        ir_version=ir_version,
                                        target_opset=target_opset,
                                        inside_loop=inside_loop,
                                        static_inputs=static_inputs,
                                        existing_functions=existing_functions)
                    except RuntimeError as e:  # pragma: no cover
                        raise RuntimeError(
                            "Unable to instantiate a node of type %r and name %r."
                            "" % (self.onnx_node.op_type, self.onnx_node.name)) from e
                else:
                    # outputs already exists, usually branch then of else for If node
                    sess = rt_class(value, runtime=runtime,
                                    ir_version=ir_version,
                                    target_opset=target_opset,
                                    inside_loop=inside_loop,
                                    static_inputs=static_inputs,
                                    existing_functions=existing_functions)
                v['value_rt'] = sess

    def _build_context(self, values, input_list):
        context = {}
        # input_list does not need to be sorted but when
        # an input is not found, the returned error is always
        # related to the same input.
        for n in sorted(input_list):
            try:
                v = values[self._global_index(n)]
            except IndexError as e:
                raise IndexError(  # pragma: no cover
                    f"Unable to find an index for result {n!r} in onnx object.") from e
            if v is None:
                raise ValueError(  # pragma: no cover
                    f"Input {n!r} is None.")
            context[n] = v
        return context

    def run(self, values, attributes=None, verbose=0, fLOG=None):
        """
        Runs the node.
        The function updates values with outputs.

        :param values: list of existing values
        :param attributes: attributes known at function level
        :param verbose: verbosity
        :param fLOG: logging function
        """
        # This code takes time if the graph contains many nodes.
        # Maybe a C++ container would help in that case (to skip GIL).
        if self.inputs_indices is None:
            args = list(values[k] for k in self.inputs)
        else:
            args = list(values[k] for k in self.inputs_indices)

        if self.ops_ is None:
            # Then a function.
            if 'atts' in self.desc:
                # attributes of a function
                if attributes is None:
                    attributes = {}
                else:
                    attributes = attributes.copy()
                attributes.update(self.desc['atts'])

            feeds = {}
            for name, val in zip(self.function_.obj.input, args):
                if val is None:
                    raise ValueError(  # pragma: no cover
                        f"Input name {name!r} is None.")
                feeds[name] = val

            if verbose == 0 or fLOG is None:
                outputs = self.function_.run(feeds, attributes=attributes)
            else:
                if verbose > 0:
                    fLOG('-- >%s[%s](%s)  -- len(feeds)=%d' %
                         (self.function_.obj.name, self.function_.obj.domain,
                          ", ".join(self.function_.obj.input), len(feeds)))
                outputs = self.function_.run(
                    feeds, attributes=attributes, verbose=verbose, fLOG=fLOG)
                if verbose > 0:
                    fLOG('-- <%s[%s][%s]' %
                         (self.function_.obj.name, self.function_.obj.domain,
                          ", ".join(self.function_.obj.output)))

            res = [outputs[k] for k in self.function_.obj.output]
        else:
            # Or an operator.
            try:
                if self.ops_.need_context():
                    context = self._build_context(values,
                                                  self.ops_.additional_inputs)
                    res = self.ops_.run(*args, context=context,
                                        attributes=attributes,
                                        verbose=verbose, fLOG=fLOG)
                else:
                    res = self.ops_.run(
                        *args, attributes=attributes, verbose=verbose, fLOG=fLOG)
            except (ValueError, TypeError) as e:
                raise RuntimeError(  # pragma: no cover
                    "Unable to run operator %r, inputs=%r."
                    "" % (type(self.ops_), self.inputs)) from e
            except OverflowError as e:
                raise RuntimeError(  # pragma: no cover
                    "Unable to run operator %r, inputs=%r."
                    "" % (type(self.ops_), self.inputs)) from e

            if not isinstance(res, tuple):
                raise RuntimeError(  # pragma: no cover
                    f"Results of operator {type(self.ops_)!r} should be a tuple.")

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
