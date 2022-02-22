# pylint: disable=E1101,C0302
"""
@file
@brief Xop API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import os
import pprint
import numpy
from scipy.sparse.coo import coo_matrix
import onnx
from onnx import GraphProto, TensorProto
from onnx.helper import (
    make_node, make_graph, make_model,
    make_tensor_value_info)
from onnx.numpy_helper import from_array, to_array
from onnx.shape_inference import infer_shapes
from ._cache import cache_folder
from .xop_variable import (
    Variable, is_numpy_dtype, numpy_type_prototype, max_supported_opset)
from .xop_auto import get_rst_doc


def _default_OPSET_TO_IR_VERSION():
    """
    Returns the default mapping between opset and ir_version.

    .. runpython::
        :showcode:

        import pprint
        from mlprodict.npy.xop import _default_OPSET_TO_IR_VERSION
        pprint.pprint(_default_OPSET_TO_IR_VERSION())
    """
    return {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
        7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7,
        13: 7, 14: 7, 15: 8}


def _domain_to_class_name(domain):
    """
    Converts domain into a name.

    :param domain: domain name such as `ai.onnx.ml`
    :return: string

    .. runpython::
        :showcode:

        from mlprodict.npy.xop import _domain_to_class_name
        print(_domain_to_class_name('ai.onnx.ml'))
    """
    if domain == 'ai.onnx':
        return ''
    dom = domain.split('.')
    res = []
    for d in dom:
        if len(d) == 0:
            res.append(d)
        elif len(d) == 1:
            res.append(d.upper())
        else:
            res.append(d[0].upper() + d[1:])
    return "".join(res)


def _populate_schemas():
    """
    Populates all schemas.
    """
    res = {}
    versions = {}
    domains = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.support_level == schema.SupportType.EXPERIMENTAL:
            # Skips experimental operators.
            continue
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.domain, schema.name] = schema
        else:
            res[schema.domain, schema.name] = schema
        full_name = schema.name + '_' + str(schema.since_version)
        res[schema.domain, full_name] = schema
        key = schema.domain, schema.name
        if key not in versions:
            versions[key] = set()
        if schema.name not in domains:
            domains[schema.name] = set()
        domains[schema.name].add(schema.domain)
        versions[key].add(full_name)
    return res, versions, domains


def _find_operator_domain(name):
    """
    Determines the domain of an operator.
    Raises an exception if not found or if there is an ambiguity.

    :param name: operator name
    :return: domain
    """
    if name not in _all_domains:
        raise ValueError(
            "Unable to guess domain for operator %r. "
            "Not found in %r." % (name, list(_all_domains)))
    domains = _all_domains[name]
    if len(domains) == 1:
        return list(domains)[0]
    raise ValueError(
        "Unable to guess domain of operator %r, found domains %r." % (
            name, domains))


def ClassFactory(class_name, op_name, inputs, outputs,
                 input_range, output_range,
                 domain, attr_names, doc,
                 deprecated, since_version,
                 past_version):
    """
    Dynamically creates a class for a specific operator.

    :param class_name: class name
    :param op_name: operator type
    :param inputs: expected inputs
    :param outputs: expected outputs
    :param input_range: input range
    :param output_range: output_range
    :param domain: domain
    :param attr_names: attributes names
    :param doc: docstring
    :param deprecated: is the operator deprecated
    :param since_version: available since version
    :param past_version: list of versions
    """

    def __init__(self, *args, **kwargs):

        op_version = kwargs.pop('op_version', None)
        if isinstance(op_version, dict):
            op_version = op_version.get(domain, None)

        if op_version is None:
            if len(args) == 0 and input_range[0] == input_range[1]:
                args = [_[0] for _ in self.__class__.expected_inputs]
            if not (input_range[0] <= len(args) <= input_range[1]):
                raise RuntimeError("Unexpected number of inputs, "
                                   "got {}, expecting {} for operator "
                                   "'{}'.".format(
                                       len(args), len(inputs), op_name))

        attr_names = self.attr_names
        if '_' in self.__class__.__name__:
            op_version_class = int(self.__class__.__name__.split('_')[-1])
            if op_version is None:
                op_version = op_version_class
            try:
                op_version = min(op_version, op_version_class)
            except TypeError:
                raise TypeError(  # pylint: disable=W0707
                    "Could not compare versions {} ? {} for "
                    "class '{}' since_version {}. Parameter 'op_version' "
                    "is probably missing when the class "
                    "is instantiated.".format(
                        op_version, op_version_class, class_name,
                        since_version))
        else:
            op_version_class = None

        # By default, the op_version is None.
        # None means the latest available.
        if op_version is None:
            op_version = since_version

        found = None
        if op_version is not None:
            # attr_names refers to the most recent version of
            # this operator. We may need an older one.
            for op in range(op_version, 0, -1):
                name = '{}_{}'.format(self.__class__.__name__, op)
                if name in self.past_version:
                    found = (name, op)
                    attr_names = self.past_version[name].attr_names
                    break
        if (op_version_class is not None and found is not None and
                found[-1] != op_version_class):
            raise RuntimeError(
                "op_version={} does not refer to the same opset as the class "
                "name ('{}').".format(op_version, self.__class__.__name__))
        for key in kwargs:
            if key in {'output_names', 'op_version', 'domain', 'ir_version',
                       'global_context', 'clear_subgraph_inputs'}:
                continue
            if key not in attr_names:
                raise TypeError("Argument '%s' not valid for '%s' opset=%s."
                                % (key, op_name, op_version))

        if op_version is not None:
            kwargs['op_version'] = op_version
        # This class can only be created by a user. Let's check
        # types are either a variable, an operator or an array.
        for i, a in enumerate(args):
            if isinstance(a, tuple):
                if len(a) != 2:
                    raise TypeError(
                        "Input %r is a tuple or class %r, it must have two "
                        "elements (name, type) not %r." % (i, class_name, a))
                if not isinstance(a[0], str):
                    raise TypeError(
                        "Input %r is a tuple or class %r, it must be a tuple "
                        "(name, type) not %r." % (i, class_name, a))
                continue
            if not isinstance(a, (
                    Variable, OnnxOperator, numpy.ndarray, str,
                    OnnxOperatorItem, coo_matrix)):
                raise TypeError(
                    "Unexpected type %r for input %r of operator %r. "
                    "It must be an instance of Variable (or a string), "
                    "OnnxOperator, OnnxOperatorItem, numpy.ndarray, "
                    "coo_matrix)." % (
                        type(a), i, class_name))
        OnnxOperator.__init__(self, *args, **kwargs)

    newclass = type(class_name, (OnnxOperator,),
                    {"__init__": __init__, '__doc__': doc,
                     'expected_inputs': inputs,
                     'expected_outputs': outputs,
                     'operator_name': op_name,
                     'input_range': input_range,
                     'output_range': output_range,
                     'domain': domain,
                     'is_deprecated': deprecated,
                     'since_version': since_version,
                     'past_version': past_version,
                     'attr_names': attr_names,
                     '__module__': __name__})
    return newclass


def _dynamic_class_creation(operator_names=None, cache=False, include_past=False,
                            verbose=0, fLOG=print):
    """
    Automatically generates classes for each of the operators
    module *onnx* defines and described at
    `Operators
    <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
    and `Operators
    <https://github.com/onnx/onnx/blob/master/docs/
    Operators-ml.md>`_.

    :param operator_names: list of operators to request or None for all
    :param cache: extract the documentation from onnx package and
        saves it on disk it True
    :param include_past: includes past versions if operator_names is None
    :param verbose: display some progress
    :param fLOG: logging function
    :return: list of requested operators as a tuple
    """
    def _c(obj, label, i):
        name = '%s%d' % (obj.name or label, i)
        tys = obj.typeStr or ''
        return (name, tys)

    cache_dir = cache_folder()
    if operator_names is None:
        operator_names = list(_all_schemas_versions)
        if include_past:
            add = []
            for domain, op in operator_names:
                add.extend(
                    [(domain, k)
                     for k in _all_schemas_versions[domain, op]])
            operator_names.extend(add)
            operator_names.sort()

    # type verification
    ops = []
    for name in operator_names:
        if isinstance(name, str):
            if name.startswith('Onnx'):
                raise ValueError(
                    "Operator name cannot start with Onnx: %r." % name)
            domain = _find_operator_domain(name.split('_', maxsplit=1)[0])
            ops.append((domain, name))
        elif isinstance(name, tuple) and len(name) == 2:
            if name[1].startswith('Onnx'):
                raise ValueError(
                    "Operator name cannot starts with Onnx: %r." % name)
            ops.append(name)
        else:
            raise ValueError(
                "Operator to fetch must be a string or a "
                "`tuple(domain, name)` not %r." % (name))
    operator_names = ops

    # versions
    res = _all_schemas
    cls = {}
    set_names = dict()
    set_skip = set()
    for pos, (op_domain, op_name) in enumerate(operator_names):
        if op_domain == 'ai.onnx':
            op_domain = ''
        set_names[op_domain, op_name] = pos
        if '_' in op_name and not include_past:
            n = op_name.split('_')[0]
            set_skip.add((op_domain, n))
            if n not in set_names:
                set_names[op_domain, n] = -1

    if verbose > 1 and fLOG is not None:
        fLOG("[_dynamic_class_creation] set_names=%r" % set_names)
        fLOG("[_dynamic_class_creation] set_skip=%r" % set_skip)

    returned_classes = []
    positions = {}

    for (op_domain, op_name), position in set_names.items():
        cl_name = 'Onnx' + _domain_to_class_name(op_domain) + op_name
        if verbose > 3 and fLOG is not None:
            fLOG('[_dynamic_class_creation] cl_name=%r op_domain=%r op_name=%r (in=%d)' % (
                cl_name, op_domain, op_name, 1 if cl_name in _all_classes else 0))
        if cl_name in _all_classes:
            if cl_name not in set_skip:
                if position >= 0:
                    returned_classes.append((position, _all_classes[cl_name]))
            continue

        # operator name without domain
        if '_' in op_name:
            names = [op_name]
        else:
            try:
                names = _all_schemas_versions[op_domain, op_name].copy()
            except KeyError as e:
                raise ValueError(
                    "Operator %r (domain=%r) does not exists." % (
                        op_name, op_domain)) from e
            names.add(op_name)

        if verbose > 0 and fLOG is not None:
            fLOG("[_dynamic_class_creation] op_domain=%r op_name=%r, cl_name=%r names=%r"
                 "" % (op_domain, op_name, cl_name, names))

        for name in names:
            try:
                schema = res[op_domain, name]
            except KeyError as e:
                raise ValueError(
                    "Operator (%r, %r) does not exists (available=%r)" % (
                        op_domain, name, pprint.pformat(list(res)))) from e
            inputs = [_c(o, 'I', i) for i, o in enumerate(schema.inputs)]
            outputs = [_c(o, 'O', i) for i, o in enumerate(schema.outputs)]
            args = [p for p in schema.attributes]

            if '_' in name:
                class_name = "Onnx" + _domain_to_class_name(op_domain) + name
            else:
                class_name = (
                    "Onnx" + _domain_to_class_name(op_domain) + schema.name)

            if verbose > 0 and fLOG is not None:
                fLOG("[_dynamic_class_creation] op_name=%r, cl_name=%r cache=%r"
                     "" % (op_name, class_name, cache))

            filename = os.path.join(
                cache_dir,
                schema.name + '_' + str(schema.since_version) + ".rst")
            if not cache and os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    doc = f.read()
            else:
                doc = get_rst_doc(schema)
                if cache:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(doc)

            cl = ClassFactory(class_name, schema.name, inputs, outputs,
                              [schema.min_input, schema.max_input],
                              [schema.min_output, schema.max_output],
                              schema.domain, args,
                              "**Version**" + doc.split('**Version**')[-1],
                              getattr(schema, 'deprecated', False),
                              schema.since_version, {})
            cls[class_name] = cl
            if name == op_name:
                positions[class_name] = position

    # Retrieves past classes.
    for name in cls:  # pylint: disable=C0206
        if '_' not in name:
            continue
        main, _ = name.split('_')
        if main in cls:  # pylint: disable=R1715
            last = cls[main]
        else:
            last = _all_classes[main]
        last.past_version[name] = cls[name]

    _all_classes.update(cls)
    for cl_name, v in cls.items():
        if v not in set_skip and positions.get(cl_name, -1) >= 0:
            returned_classes.append((positions[cl_name], v))

    returned_classes.sort()
    return tuple(e[1] for e in returned_classes)


def loadop(*names, cache=False, verbose=0, fLOG=print):
    """
    Dynamically creates a class for a every operator type in
    the given list.
    """
    res = _dynamic_class_creation(
        names, cache=cache, verbose=verbose, fLOG=fLOG)
    if len(res) == 1:
        return res[0]
    return res


class OnnxLoadFactory:
    """
    Automatically creating all operators from onnx packages
    takes time. That's why function @see cl loadop only creates
    classes for the requested operators. This class does the same
    when an attributes is requested.

    ::

        cl = OnnxLoadOperators()
        x = cl.Add(...)

    It is equivalent to:

    ::

        OnnxAdd = loadop('Add')
        x = OnnxAdd(...)
    """

    def __init__(self):
        self._loaded_classes = {}

    def __getattr__(self, name):
        """

        """
        if name == '_loaded_classes':
            return self._loaded_classes
        if name in self._loaded_classes:
            return self._loaded_classes[name]
        cl = loadop(name)
        self._loaded_classes[name] = cl
        self._loaded_classes[cl.__name__] = cl
        return cl


class OnnxOperatorItem:
    """
    Accessor to one of the output returned by a @see cl OnnxOperator.

    :param onx_op: @see cl OnnxOperator
    :param index: integer
    :param op_version: defines the opset version
    """

    def __init__(self, onx_op, index, op_version=None):
        if not isinstance(index, int):
            raise TypeError("index must be an integer not %r." % type(index))
        self.onx_op = onx_op
        self.index = index
        self.op_version = op_version

    @property
    def inputs(self):
        "Returns the only inputs in a list."
        inp = self.onx_op.output
        return [inp[self.index]]

    def add_to(self, builder):
        """
        Adds to graph builder.
        Does nothing because the original node is already added.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        pass

    def __str__(self):
        """
        usual
        """
        return "%s[%d]" % (str(self.onx_op), self.index)

    def get_output_name(self, i=0):
        """
        Returns the output name at position *i*.
        """
        if i != 0:
            raise IndexError("Can only return the first item.")
        return self.onx_op.get_output_name(self.index)

    def get_output(self, i=0):
        """
        Returns the output.
        """
        if i != 0:
            raise IndexError("Can only return the first item.")
        return self.onx_op.get_output(self.index)


class OnnxOperator:
    """
    Ancestor to every *ONNX* operator exposed in
    :mod:`mlprodict.npy.xops` and :mod:`mlprodict.npy.xops_ml`.

    :param inputs: list of inputs expected by the operator
    :param op_version: to select a specific version of the operator
    :param output_names: used defined names for the outputs
    :param domain: to overwrite the default domain
    :param global_context: operator *If* executes one subgraph
        whose nodes may use one existing output in the current
        context. If not used in the main graph, these operators
        are not linked to the output and cannot be retrieved.
        *global_context* is a dictionary mapped the subgraph input
        names to these operators.
    :param kwargs: additional parameters of the operator

    .. versionadd:: 0.9
    """

    def __init__(self, *inputs, op_version=None, output_names=None,
                 domain=None, global_context=None, **kwargs):

        if (output_names is None and
                self.__class__.__name__.startswith("OnnxScan")):
            raise NotImplementedError(
                "The class cannot infer the number of variables "
                "for node '{}' yet. output_names must be specified"
                ".".format(self.__class__.__name__))
        if isinstance(output_names, (str, Variable)):
            output_names = [output_names]
            if isinstance(output_names[0], str):
                output_names[0] = Variable(output_names[0])
        elif isinstance(output_names, list):
            if len(output_names) == 0:
                raise ValueError(
                    "output_names cannot be empty (operator %r)."
                    "" % self.__class__.__name__)
            output_names = output_names.copy()
            for i in range(len(output_names)):  # pylint: disable=C0200
                if isinstance(output_names[i], str):
                    output_names[i] = Variable(output_names[i])
        elif output_names is not None:
            raise TypeError(
                "output_names must be a string or a list not %r."
                "" % type(output_names))

        if op_version is None:
            if domain == '':
                self.op_version = max_supported_opset()
            else:
                self.op_version = None
        else:
            self.op_version = op_version
        self.since_version = self.__class__.since_version

        if (self.op_version is not None and
                self.op_version < self.since_version):
            schema = self.find_schema(self.op_version)
            self.since_version = schema.since_version
            self.expected_inputs = schema.expected_inputs.copy()
            self.expected_outputs = schema.expected_outputs.copy()
            self.input_range = schema.input_range
            self.output_range = schema.output_range
        else:
            self.expected_inputs = (
                None if self.__class__.expected_inputs is None
                else self.__class__.expected_inputs.copy())
            self.expected_outputs = (
                None if self.__class__.expected_outputs is None
                else self.__class__.expected_outputs.copy())
            self.input_range = self.__class__.input_range
            self.output_range = self.__class__.output_range
            if self.__class__.__name__ not in {
                    'OnnxScan', 'OnnxLoop', 'OnnxIf'}:
                # The minimum opset depends on embedded graph
                # by default, it takes the given op_version but the
                # optimal value could be lower.
                self.op_version = self.since_version
            if self.op_version is None:
                self.op_version = self.since_version

        if (self.op_version is not None and
                self.op_version < self.since_version):
            raise RuntimeError(
                "Operator '{}': requested version {} < "
                "{} schema version.".format(
                    self.__class__.__name__,
                    self.op_version, self.since_version))

        self.state = None
        self.domain = domain
        self.kwargs = kwargs
        self.onnx_prefix_name = None
        self.max_item_ = None

        # check inputs
        if len(inputs) == 0:
            if self.input_range[0] == self.input_range[1]:
                self.inputs = [OnnxOperator.UnscopedVariable(_[0])
                               for _ in self.expected_inputs]
            else:
                # The number of inputs may vary.
                self.inputs = None
        else:
            self.inputs = []
            for inp in inputs:
                if isinstance(inp, str):
                    self.inputs.append(Variable(inp))
                elif isinstance(inp, (OnnxOperator, Variable,
                                      OnnxOperatorItem)):
                    self.inputs.append(inp)
                elif isinstance(inp, (numpy.ndarray, coo_matrix, TensorProto)):
                    self.inputs.append(inp)
                else:
                    raise TypeError(
                        "Unable to interpret the input name for type {} in "
                        "operator '{}' (value={}).".format(
                            type(inp), self.__class__.__name__, inp))

        if self.inputs is not None:
            if (len(self.inputs) < self.input_range[0] or
                    len(self.inputs) > self.input_range[1]):
                raise RuntimeError(
                    "Operator '{}' expects a number of inputs "
                    "in [{}, {}] not {} (expected opset={}, "
                    "class opset={})".format(
                        self.operator_name, *self.input_range,
                        len(self.inputs), op_version, self.op_version))
        # global context
        if global_context is None:
            self.global_context = None
        else:
            if not isinstance(global_context, dict):
                raise TypeError(
                    "global_context must be a dictionary not %r."
                    "" % type(global_context))
            for k, v in global_context.items():
                if not isinstance(v, (OnnxOperator, OnnxOperatorItem)):
                    raise TypeError(
                        "Value %r in must be an OnnxOperator or an "
                        "OnnxOperatorItem not %r." % (k, type(v)))
            self.global_context = global_context

        # check output
        self.output_names = output_names
        self.output_variables = None

        if self.output_names is not None:
            if len(self.output_names) == 0:
                raise ValueError(
                    "output_names can be None but cannot be empty for "
                    "operator %r." % self)
            if self.output_variables is None:
                self.output_variables = [None for o in self.output_names]
            for i in range(len(self.output_names)):  # pylint: disable=C0200
                name = self.output_names[i]
                if isinstance(name, Variable):
                    self.output_variables[i] = name
                else:
                    raise TypeError("output_names must be a list of strings "
                                    "and element %r is %r (%r)" % (
                                        i, type(name), name))
            if all(map(lambda x: x is None, self.output_variables)):
                self.output_variables = None

        if (self.output_names is not None and (
                self.expected_outputs is None or
                len(self.output_names) > len(self.expected_outputs))):
            if self.expected_outputs is None:
                self.expected_outputs = []
            for i in range(len(self.expected_outputs),
                           len(self.output_names)):
                self.expected_outputs.append((self.output_names[i], None))

        if (self.expected_inputs is None or
                len(self.inputs) > len(self.expected_inputs)):
            if self.expected_inputs is None:
                self.expected_inputs = []
            for i in range(len(self.expected_inputs),
                           len(self.inputs)):
                inp = self.inputs[i]
                if isinstance(inp, str):
                    inp = (inp, None)
                elif hasattr(inp, 'add_to'):
                    # OnnxOperator
                    existing = set(_[0] for _ in self.expected_inputs)
                    i = 10
                    name = "input%d" % (10 + i)
                    while name in existing:
                        i += 1
                        name = "input%d" % (10 + i)
                    inp = (name, None)
                self.expected_inputs.append(inp)

        self._post_process_attributes()
        self._check()

    def _check(self):
        input_types = (Variable, OnnxOperator,
                       OnnxOperatorItem, numpy.ndarray)
        for o in self.inputs:
            if not isinstance(o, input_types):
                raise TypeError(
                    "Wrong type for inputs %r." % (
                        self.inputs, ))
        if self.output_names is not None:
            for o in self.output_names:
                if not isinstance(o, Variable):
                    raise TypeError(
                        "Wrong type for output_names %r." % (
                            self.output_names, ))

    def _post_process_attributes(self):
        """
        Walks through attributes and replaces them by ONNX values.
        """
        # Looks into attributes if there is any tuple
        # (GraphProto, OnnxOperator). In that case, the function
        # replaces the tuple by the graph proto and keeps
        # in attributes graph_algebra the OnnxOperator
        # which is the source of it.
        updates = {}
        graph_algebra = {}
        for k, v in self.kwargs.items():
            if isinstance(v, tuple) and isinstance(v[0], GraphProto):
                updates[k] = v[0]
                graph_algebra[k] = v[1]

        if len(graph_algebra) > 0:
            self.kwargs.update(updates)
            self.graph_algebra = graph_algebra

        if self.__class__.__name__ == "OnnxConstantOfShape":
            if "value" in self.kwargs:
                value = self.kwargs['value']
                if isinstance(value, TensorProto):
                    return
                if isinstance(value, numpy.ndarray):
                    if value.shape == (1, ):
                        val = value[0]
                    elif len(value.shape) == 0:
                        val = value
                    else:
                        raise RuntimeError(
                            "Unexpected shape %r for value, it must be "
                            "an array of one element." % value.shape)
                    self.kwargs['value'] = from_array(
                        numpy.array([val], dtype=value.dtype))
                    return
                raise TypeError(
                    "Unexpected type %r for value. It should be an array "
                    "of one element." % type(value))
            return

        if self.__class__.__name__ == "OnnxCast":
            if "to" in self.kwargs:
                value = self.kwargs['to']
                if not isinstance(value, int):
                    try:
                        to = numpy_type_prototype(value)
                    except ValueError as e:  # pragma: no cover
                        raise ValueError(
                            "Unable to convert argument to in operator cast, "
                            "type is %r, value is %r." % (type(value), value)) from e
                    self.kwargs['to'] = to
            return

    def update_max_item(self, index):
        """
        Some operators return a undefined number of outputs.
        The method is called when require one of them (with `__getitem__`)
        and keeps the greater requested index assuming the node does
        not output any result beyond that index.

        :param index: requested index
        """
        if self.max_item_ is None:
            self.max_item_ = index
        else:
            self.max_item_ = max(self.max_item_, index)
        if self.expected_outputs is None:
            self.expected_outputs = []
        while len(self.expected_outputs) <= self.max_item_:
            self.expected_outputs.append(
                (("NEWOUTPUT", len(self.expected_outputs)), None))

    def find_schema(self, op_version):
        """
        Checks if there is an existing schema for a specific version.

        :param op_version: requested version
        :return: schema
        """
        if not hasattr(self.__class__, 'past_version'):
            raise RuntimeError("Missing attribute 'past_version', there is "
                               "no other available schema.")
        found = None
        for v in self.past_version.values():
            if v.since_version > op_version:
                continue
            if found is None or v.since_version > found.since_version:
                found = v
        if found is None:
            raise RuntimeError(
                "Operator '{}': requested version {} < "
                "{} schema version (past_version {}).".format(
                    self.__class__.__name__,
                    op_version, self.since_version,
                    [v.since_version for v in self.past_version.values()]))
        return found

    def __str__(self):
        """
        usual
        """
        return "{}({} in) -> {}".format(
            self.__class__.__name__,
            len(self.inputs) if self.inputs is not None else 0,
            [str(o) for o in self.output_names]
            if self.output_names is not None else "?")

    def set_onnx_name_prefix(self, onnx_prefix_name):
        """
        Provides a name to define a prefix in the onnx graph
        to avoid to get unreadable node names. The method
        does not overwrite an existing name, it propagates
        the prefix to inputs and stops the propagation
        if the prefix is already defined.
        """
        if self.onnx_prefix_name is None:
            self.onnx_prefix_name = onnx_prefix_name
            for inp in self.inputs:
                if hasattr(inp, 'set_onnx_prefix_name'):
                    inp.set_onnx_name_prefix(onnx_prefix_name)
        return self

    @property
    def onnx_prefix(self):
        "Returns a prefix for results coming out from this node."
        if self.onnx_prefix_name is None:
            name = self.__class__.__name__
            if name.startswith("Onnx"):
                name = name[4:]
            return 'out_' + name[:3].lower()
        return self.onnx_prefix_name

    def __getitem__(self, index):
        """
        Returns an accessor to one of the output
        of this node.
        """
        self.update_max_item(index)
        return OnnxOperatorItem(self, index, self.op_version)

    def __iter__(self):
        """
        Allows expressions such as ``a, b = OnnxTopK(...)``.
        """
        n = None
        if self.output_names is not None:
            n = len(self.output_names)
        else:
            rg = self.output_range
            if rg[0] == rg[1] and rg[0] > 0:
                n = rg[0]
        if n is None and self.max_item_ is not None:
            n = self.max_item_ + 1
        if n is None:
            raise RuntimeError(
                "Unable to guess the number of outputs of node type %r. "
                "Uses operator [] to select a specific output." %
                self.__class__.__name__)
        if self.max_item_ is not None:
            n = max(n, self.max_item_ + 1)
        for i in range(n):
            yield self[i]

    def add_to(self, builder):
        """
        Adds to graph builder.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        inputs = builder.get_input_names(self, self.inputs)
        if self.output_names is not None:
            n_outputs = len(self.output_names)
        elif self.expected_outputs is not None:
            n_outputs = len(self.expected_outputs)
        else:
            n_outputs = self.output_range[0]
        outputs = [builder.get_output_name(self, i) for i in range(n_outputs)]
        builder.add_node(
            self.operator_name,
            builder.get_unique_name('_' + self.operator_name.lower()),
            inputs, outputs, domain=self.domain, opset=self.op_version,
            **self.kwargs)

    def _node_to_graph(self, other_outputs=None, inputs=None, outputs=None):
        """
        Builds a graph as a list of nodes to walk through in that order.
        """
        def _preprocess_list(inputs):
            new_inputs = {}
            for el in inputs:
                if isinstance(el, str):
                    new_inputs[el] = Variable(el)
                elif isinstance(el, Variable):
                    new_inputs[el.name] = el
                else:
                    raise TypeError(
                        "Unable to handle input type %r (%r)." % (
                            type(el), el))
            return new_inputs

        def _process_input(inputs, set_inputs, inp, new_inputs):
            if isinstance(inp, OnnxOperator):
                new_stack.append(inp)
            elif isinstance(inp, OnnxOperatorItem):
                new_stack.append(inp)
                new_stack.append(inp.onx_op)
            elif isinstance(inp, Variable):
                if inp.name in set_inputs:
                    return
                set_inputs.add(inp.name)
                if inputs is None:
                    new_inputs.append(inp)
                elif isinstance(inputs, dict):
                    if inp.name in inputs:
                        new_inputs.append(inp.copy_merge(inputs[inp.name]))
                    else:
                        raise ValueError(  # pragma: no cover
                            "Unable to find input %r in %r." % (
                                inp, inputs))
                elif is_numpy_dtype(inputs):
                    new_inputs.append(inp.copy_add(inputs))
                elif isinstance(inputs, Variable):
                    if inp.name == inputs.name:
                        new_inputs.append(inp.copy_merge(inputs))
                    else:
                        new_inputs.append(inp)
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to handle inputs=%r." % inputs)
            elif isinstance(inp, numpy.ndarray):
                pass
            else:
                raise TypeError(
                    "Unexpected input type %r in node type %r." % (
                        type(inp), type(obj)))

        def _get_type(node, name=None, outputs=None):
            if outputs is None:
                return None
            if isinstance(outputs, Variable):
                if name is None:
                    return outputs.dtype
                if isinstance(name, Variable):
                    return outputs.dtype or name.dtype
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to handle outputs=%r." % outputs)
            if isinstance(outputs, dict):
                if name is None:
                    raise RuntimeError(
                        "Unable to get type among %r, name=None." % (
                            outputs, ))
                if isinstance(name, Variable):
                    n = name.name
                else:
                    n = name
                if n not in outputs:
                    return None
                return outputs[n]
            if isinstance(outputs, list):
                raise NotImplementedError(
                    "Unexpected type for name=%r, outputs=%r." % (
                        name, outputs))
            if is_numpy_dtype(outputs):
                return outputs
            raise RuntimeError(  # pragma: no cover
                "Unable to handle outputs=%r." % outputs)

        node_outputs = [self]
        if other_outputs is not None:
            node_outputs += other_outputs

        # preprocess inputs, outputs
        _keep_inputs = None
        if isinstance(inputs, list):
            _keep_inputs = inputs
            inputs = _preprocess_list(inputs)
        _keep_outputs = None
        if isinstance(outputs, list):
            _keep_outputs = outputs
            outputs = _preprocess_list(outputs)

        # walk through graphs
        stack = list(node_outputs)
        new_inputs = []
        set_inputs = set()
        memo = []
        while len(stack) > 0:
            memo.extend(stack)
            new_stack = []
            for obj in stack:
                if isinstance(obj, OnnxOperatorItem):
                    pass
                else:
                    for inp in obj.inputs:
                        _process_input(inputs, set_inputs, inp, new_inputs)
            stack = new_stack

        # eliminate duplicates
        done = set()
        nodes = []
        for node in reversed(memo):
            if id(node) in done:
                continue
            done.add(id(node))
            nodes.append(node)

        # outputs
        set_names = set()
        new_outputs = []
        run_shape = False
        for node in node_outputs:
            if node.output_names is None:
                n = self.output_range[0]
                for i in range(n):
                    to = _get_type(node, outputs=outputs)
                    if to is None:
                        run_shape = True
                    res = 'out%d' % i
                    var = Variable(res, added_dtype=to)
                    if var.name in set_names:
                        raise RuntimeError(
                            "Duplicated output name var=%r." % var)
                    set_names.add(var.name)
                    new_outputs.append(var)
            else:
                for o in node.output_names:
                    to = _get_type(node, o, outputs=outputs)
                    if to is None:
                        run_shape = True
                    res = (o, to)
                    var = o.copy_merge(to)
                    if var.name in set_names:
                        raise RuntimeError(
                            "Duplicated output name o=%r var=%r." % (o, var))
                    set_names.add(var.name)
                    new_outputs.append(var)
        if len(new_outputs) == 0:
            raise RuntimeError(
                "No detected outputs inputs=%r outputs=%r." % (
                    inputs, outputs))

        return nodes, new_inputs, new_outputs, run_shape

    def to_onnx(self, inputs=None, outputs=None,
                other_outputs=None, target_opset=None,
                verbose=0, run_shape=True):
        """
        Converts this operator into an ONNX graph.

        :param inputs: information about type, it should not be None
        :param outputs: information about types, if None, the function
            will use shape inference to guess the final output type
            and shape
        :param other_outputs: additional nodes to consider
            as graph outputs but not outputs of this particular
            node
        :param target_opset: dictionary with target opset per domain,
            None for the default one
        :param run_shape: in case output shapes are not specify,
            the function runs function :epkg:`infer_shapes`
            to guess them, False would disable that
            default behaviour
        :param verbose: prints information
        """
        # opsets
        if isinstance(target_opset, dict):
            dom = self.domain or ''
            target_opset = target_opset.get(dom, None)
        elif isinstance(target_opset, int):
            if self.domain not in ('', None):
                # The target_opset is for the domain '' we ignore it.
                target_opset = None
        elif target_opset is not None:
            raise TypeError(
                "target_opset must be a dictionary {domain: "
                "target_opset} not %r for operator %r." % (
                    target_opset, self.__class__.__name__))

        if self.domain in ('', None) and target_opset == 1:
            raise RuntimeError("target_opset cannot be 1.")
        if (self.op_version is not None and target_opset is not None and
                self.op_version > target_opset):
            raise RuntimeError(
                "target_opset={} is lower than the version={} requested "
                "for this node '{}'.".format(
                    target_opset, self.op_version, self.__class__.__name__))

        # get the graph
        nodes, graph_inputs, graph_outputs, run_shape2 = self._node_to_graph(
            other_outputs, inputs, outputs)
        if len(nodes) == 0:
            raise RuntimeError(  # pragma: no cover
                "Node list is empty.")
        if verbose > 1:
            for i, n in enumerate(nodes):
                print("nodes[%d]=%r" % (i, n))
            for i, n in enumerate(graph_inputs):
                print("graph_inputs[%d]=%r" % (i, n))
        builder = _GraphBuilder()
        for node in nodes:
            node.add_to(builder)

        return builder.to_onnx(
            inputs=graph_inputs, outputs=graph_outputs,
            target_opset=target_opset, verbose=verbose,
            run_shape=run_shape and run_shape2)

    @staticmethod
    def _merge_op_version(n1, n2):
        if isinstance(n2, OnnxOperator):
            if n1.op_version is None:
                opv = n2.op_version
            elif n2.op_version is None:
                opv = n1.op_version
            elif n1.op_version == n2.op_version:
                opv = n1.op_version
            else:
                opv = max(n1.op_version, n2.op_version)
        elif isinstance(n2, OnnxOperatorItem):
            opv = OnnxOperator._merge_op_version(n1, n2.onx_op)
        else:
            opv = n1.op_version
        return opv

    def __add__(self, ov):
        """
        Automatically adds operator `OnnxAdd` to the graph.

        :param ov: onnx node
        :return: `OnnxAdd(self, ov)`
        """
        OnnxAdd = loadop('Add')
        opv = self._merge_op_version(self, ov)
        return OnnxAdd(self, ov, op_version=opv)

    def __sub__(self, ov):
        """
        Automatically adds operator `OnnxSub` to the graph.

        :param ov: onnx node
        :return: `OnnxSub(self, ov)`
        """
        OnnxSub = loadop('Sub')
        opv = self._merge_op_version(self, ov)
        return OnnxSub(self, ov, op_version=opv)

    def __mul__(self, ov):
        """
        Automatically adds operator `OnnxMul` to the graph.

        :param ov: onnx node
        :return: `OnnxMul(self, ov)`
        """
        OnnxMul = loadop('Mul')
        opv = self._merge_op_version(self, ov)
        return OnnxMul(self, ov, op_version=opv)

    def __truediv__(self, ov):
        """
        Automatically adds operator `OnnxDiv` to the graph.

        :param ov: onnx node
        :return: `OnnxDiv(self, ov)`
        """
        OnnxDiv = loadop('Div')
        opv = self._merge_op_version(self, ov)
        return OnnxDiv(self, ov, op_version=opv)

    def __pow__(self, ov):
        """
        Automatically adds operator `OnnxPow` to the graph.

        :param ov: onnx node
        :return: `OnnPow(self, ov)`
        """
        OnnxPow = loadop('Pow')
        opv = self._merge_op_version(self, ov)
        return OnnxPow(self, ov, op_version=opv)

    def __mod__(self, ov):
        """
        Automatically adds operator `OnnxMod` to the graph.

        :param ov: onnx node
        :return: `OnnxMod(self, ov)`
        """
        OnnxMod = loadop('Mod')
        opv = self._merge_op_version(self, ov)
        return OnnxMod(self, ov, op_version=opv)

    def __matmul__(self, ov):
        """
        Automatically adds operator `OnnxMatMul` to the graph.

        :param ov: onnx node
        :return: `OnnMatMul(self, ov)`
        """
        OnnxMatMul = loadop('MatMul')
        opv = self._merge_op_version(self, ov)
        return OnnxMatMul(self, ov, op_version=opv)

    def __gt__(self, ov):
        """
        Automatically adds operator `OnnxGreater` to the graph.

        :param ov: onnx node
        :return: `OnnxGreater(self, ov)`
        """
        OnnxGreater = loadop('Greater')
        opv = self._merge_op_version(self, ov)
        return OnnxGreater(self, ov, op_version=opv)

    def __lt__(self, ov):
        """
        Automatically adds operator `OnnxLess` to the graph.

        :param ov: onnx node
        :return: `OnnxLess(self, ov)`
        """
        OnnxLess = loadop('Less')
        opv = self._merge_op_version(self, ov)
        return OnnxLess(self, ov, op_version=opv)

    def __eq__(self, ov):
        """
        Automatically adds operator `OnnxEqual` to the graph.

        :param ov: onnx node
        :return: `OnnxEqual(self, ov)`
        """
        OnnxEqual = loadop('Equal')
        opv = self._merge_op_version(self, ov)
        return OnnxEqual(self, ov, op_version=opv)

    def and_(self, ov):
        """
        Automatically adds operator `OnnxAnd` to the graph.

        :param ov: onnx node
        :return: `OnnxAnd(self, ov)`
        """
        OnnxAnd = loadop('And')
        opv = self._merge_op_version(self, ov)
        return OnnxAnd(self, ov, op_version=opv)

    def or_(self, ov):
        """
        Automatically adds operator `OnnxOr` to the graph.

        :param ov: onnx node
        :return: `OnnxOr(self, ov)`
        """
        OnnxOr = loadop('Or')
        opv = self._merge_op_version(self, ov)
        return OnnxOr(self, ov, op_version=opv)

    def __ne__(self, ov):
        """
        Automatically adds operator `OnnxNot x OnnxEqual` to the graph.

        :param ov: onnx node
        :return: `OnnxNot(OnnxEqual(self, ov))`
        """
        OnnxNot, OnnxEqual = loadop('Not', 'Equal')
        opv = self._merge_op_version(self, ov)
        return OnnxNot(OnnxEqual(self, ov, op_version=opv), op_version=opv)

    def __abs__(self):
        """
        Automatically adds operator `OnnxAbs` to the graph.

        :param ov: onnx node
        :return: `OnnxAbs(self, ov)`
        """
        OnnxAbs = loadop('Abs')
        return OnnxAbs(self, op_version=self.op_version)

    def not_(self):
        """
        Automatically adds operator `OnnxNot` to the graph.

        :param ov: onnx node
        :return: `OnnxNot(self, ov)`
        """
        OnnxNot = loadop('Not')
        return OnnxNot(self, op_version=self.op_version)


class _GraphBuilder:
    """
    Graph builder.
    """

    def __init__(self):
        self.initializer = []
        self.node = []
        self.input = []
        self.output = []
        self.opsets = {}
        self.names = set()
        self.input_names = {}
        self.output_names = {}
        self.output_names_rev = {}

    @staticmethod
    def number2alpha(index):
        """
        Converts a numbers into a string keeping the same
        alphabetical order.
        """
        dec = str(int(index))
        if len(dec) == 1:
            return dec
        return chr(96 + len(dec)) + dec

    def get_unique_name(self, name):
        """
        Returns a unique name to name an output.
        """
        if not isinstance(name, str):
            raise TypeError(  # pragma: no cover
                "name must be a string not %r." % type(name))
        if name not in self.names:
            self.names.add(name)
            return name
        i = 1
        new_name = "%s_%s" % (name, self.number2alpha(i))
        while new_name in self.names:
            i += 1
            new_name = "%s_%s" % (name, self.number2alpha(i))
        self.names.add(new_name)
        return new_name

    def get_output_name(self, node, index):
        """
        Returns the output name for a node.
        """
        key = id(node), index
        if key in self.output_names:
            name = self.output_names[key]
            return name

        if node.output_names is None:
            if node.expected_outputs is None:
                prefix = node.onnx_prefix
                n = '%s%d' % (prefix, index)
            else:
                n = node.expected_outputs[index][0]
                if isinstance(n, tuple):
                    if n[0] == 'NEWOUTPUT':
                        # This case happen for node with undefined number
                        # of outputs like Split.
                        prefix = node.onnx_prefix
                        n = '%s%d' % (prefix, index)
                    else:
                        raise RuntimeError(
                            "Unexpected value for node=%r and output=%r." % (
                                node, n))
        else:
            output = node.output_names[index]
            if isinstance(output, Variable):
                n = output.name
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for output %d (output_names=%r)." % (
                        type(output), index, node.output_names))

        name = self.get_unique_name(n)
        self.output_names[key] = name
        self.output_names_rev[name] = key
        if node.output_names is not None:
            var = node.output_names[index]
            if isinstance(var, Variable):
                var = var.name
            if var != name:
                raise RuntimeError(
                    "Output unique name %r is different from the "
                    "expected name %r at position %r." % (
                        name, node.output_names[index], index))
        return name

    def get_input_names(self, node, inputs):
        """
        Returns input names for node *node* and inputs *inputs*.

        :param node: node
        :param inputs: inputs
        :return: name
        """
        names = []
        for i in inputs:
            if isinstance(i, Variable):
                names.append(i.name)
                self.names.add(i.name)
                self.input_names[i.name] = i
            elif isinstance(i, OnnxOperator):
                name = self.get_output_name(i, 0)
                names.append(name)
                self.names.add(name)
            elif isinstance(i, OnnxOperatorItem):
                name = self.get_output_name(i.onx_op, i.index)
                names.append(name)
                self.names.add(name)
            elif isinstance(i, numpy.ndarray):
                # Adding an initializer
                name = self.get_unique_name('init')
                init = from_array(i, name)
                self.initializer.append(init)
                names.append(name)
                self.names.add(name)
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type for an input %r." % type(i))
        return names

    def add_initializer(self, name, init):
        """
        Adds an initializer to the graph.

        :param name: initializer name
        :param init: initializer to copy
        :return: created intializer
        """
        value = to_array(init)
        val = from_array(value, name)
        self.initializer.append(val)
        return val

    def add_node(self, op_type, name, inputs, outputs, domain='',
                 opset=None, **attributes):
        """
        Adds a node to the graph.

        :param op_type: operator type
        :param name: node name
        :param inputs: inputs name list
        :param outputs: outputs name list
        :param domain: node domain
        :param opset: node opset
        :return: created node
        """
        if not isinstance(inputs, list):
            raise TypeError(  # pragma: no cover
                "inputs must be a list not %r." % type(inputs))
        if not isinstance(outputs, list):
            raise TypeError(  # pragma: no cover
                "inputs must be a list not %r." % type(outputs))
        if any(map(lambda x: not isinstance(x, str), inputs)):
            raise TypeError(  # pragma: no cover
                "inputs must be all strings not %r." % inputs)
        if any(map(lambda x: not isinstance(x, (str, Variable)), outputs)):
            raise TypeError(  # pragma: no cover
                "outputs must be all strings not %r." % outputs)
        if opset is not None:
            if domain not in self.opsets:
                self.opsets[domain] = opset
            else:
                self.opsets[domain] = max(opset, self.opsets[domain])
        node = make_node(op_type, inputs, outputs, name=name,
                         domain=domain, **attributes)
        self.node.append(node)
        return node

    def _process_io(self, inputs, input_names):
        if inputs is None:
            return [
                make_tensor_value_info(
                    'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
                for name in self.input_names]

        if not isinstance(inputs, list):
            if is_numpy_dtype(inputs):
                inputs = [inputs]

        if input_names is None:
            # outputs
            set_names = set()
            input_names = []
            for inp in inputs:
                if isinstance(inp, Variable):
                    if inp.name in set_names:
                        raise ValueError(
                            "Names already taken %r in %r." % (
                                inp.name, inputs))
                    set_names.add(inp.name)
                    if inp.name in self.output_names_rev:
                        input_names.append(inp)
                else:
                    raise TypeError(
                        "Unexpected type %r in %r." % (inp, inputs))
            if len(input_names) == 0:
                raise RuntimeError(
                    "Unable to cross %r and %r." % (input, self.output_names_rev))
        elif not isinstance(input_names, list):
            raise RuntimeError(
                "Unexpected type for input_names %r." % type(input_names))

        if len(input_names) != len(inputs):
            raise RuntimeError(  # pragma: no cover
                "Mismatch between %r and %r." % (
                    input_names, inputs))

        if isinstance(input_names, list):
            d_input_names = {}
            for inp in input_names:
                if inp.name in d_input_names:
                    raise ValueError(
                        "Duplicated name %r in %r." % (inp.name, input_names))
                d_input_names[inp.name] = inp
        elif isinstance(input_names, dict):
            d_input_names = input_names
        else:
            raise TypeError(
                "Unexpected type for input_names %r (%r)." % (
                    type(input_names), input_names))

        res = []
        for inp in inputs:
            if not isinstance(inp, Variable):
                raise TypeError(
                    "inp not Variable but %r (%r)." % (type(inp), inp))
            var = d_input_names[inp.name]
            if not isinstance(var, Variable):
                raise TypeError(
                    "var not Variable but %r (%r)." % (type(var), var))
            # inp: Variable
            # var: str
            if inp != var:
                raise RuntimeError(
                    "Unexpected %r != %r." % (inp, var))
            res.append(make_tensor_value_info(
                inp.name, inp.proto_added_type, inp.proto_added_shape))

        return res

    def to_onnx(self, inputs=None, outputs=None,
                target_opset=None, run_shape=False,
                verbose=0):
        """
        Converts this operator into an ONNX graph.

        :param inputs: specific inputs (as a dictionary) or
            default inputs if not specified
        :param outputs: specific outputs
        :param target_opset: dictionary with target opset per domain,
            None for the default one
        :param run_shape: run shape inference before returning the model
        :param verbose: prints information
        :return: onnx graph
        """
        # inputs and outputs
        self.input = self._process_io(inputs, list(self.input_names.values()))
        self.output = self._process_io(outputs, None)

        graph = make_graph(
            self.node, 'XOP', self.input, self.output, self.initializer)
        onnx_model = make_model(graph)
        opv = self.opsets.get('', max_supported_opset())
        opset2ir = _default_OPSET_TO_IR_VERSION()
        irv = opset2ir.get(opv, max(opset2ir.values()))
        onnx_model.ir_version = irv

        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for k, v in self.opsets.items():
            op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
            op_set.domain = k or ''
            op_set.version = v

        if run_shape:
            return infer_shapes(onnx_model)
        return onnx_model


_all_schemas, _all_schemas_versions, _all_domains = _populate_schemas()
_all_classes = {}
onnx_load_factory = Xop = OnnxLoadFactory()
