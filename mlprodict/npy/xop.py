# pylint: disable=E1101,C0302
"""
@file
@brief Xop API to build onnx graphs. Inspired from :epkg:`sklearn-onnx`.

.. versionadded:: 0.9
"""
import os
import pprint
import logging
import hashlib
import json
from collections import OrderedDict
import numpy
from scipy.sparse.coo import coo_matrix
import onnx
from onnx import GraphProto, TensorProto, ValueInfoProto
from onnx.helper import (
    make_node, make_graph, make_model, make_value_info,
    make_tensor_value_info, make_function, make_opsetid,
    make_tensor_type_proto, make_operatorsetid)
from onnx.numpy_helper import from_array, to_array
from onnx.shape_inference import infer_shapes
from ..onnx_tools.model_checker import check_onnx
from ._cache import cache_folder
from .xop_variable import (
    Variable, is_numpy_dtype, numpy_type_prototype, max_supported_opset,
    DetectedVariable, InputDetectedVariable, OutputDetectedVariable,
    NodeResultName, guess_numpy_type, ExistingVariable)
from .xop_auto import get_rst_doc
from .xop_helper import _infer_node_output


class _WrapperLogger:
    """
    Wrappers around class :class:`logging.Logger`
    to take indentation into account.
    """

    def __init__(self, lg):
        "constructor"
        self._logger = lg
        self._indent = 0

    def debug(self, msg, *args):
        "debug"
        self._logger.debug("%s" + msg, "  " * self._indent, *args)

    def indent(self):
        "indent"
        self._indent += 1

    def dedent(self):
        "unindent"
        self._indent -= 1
        if self._indent < 0:
            raise RuntimeError(  # pragma: no cover
                "Indentation cannot be negative.")


logger = _WrapperLogger(logging.getLogger('xop'))


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
        13: 7, 14: 7, 15: 8, 16: 8}


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


class _CustomSchema:
    """
    For operators defined outside onnx.
    """

    class _empty:
        "dummy class"

        @staticmethod
        def from_attribute(data):
            "Creates an instance of `_CustomSchema._attribute`."
            if not isinstance(data, dict):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type {type(data)!r}.")
            self = _CustomSchema._empty()
            setattr(self, 'name', data['name'])
            setattr(self, 'description', data['description'])
            setattr(self, 'required', data['required'])
            setattr(self, 'type', _CustomSchema._empty())
            setattr(self.type, 'value', data['type'])
            setattr(self, 'default_value', '?')
            return self

        @staticmethod
        def from_io(data):
            "Creates an instance of `_CustomSchema._io`."
            if not isinstance(data, dict):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type {type(data)!r}.")
            self = _CustomSchema._empty()
            setattr(self, 'name', data['name'])
            setattr(self, 'typeStr', data['typeStr'])
            setattr(self, 'description', data['description'])
            setattr(self, 'option', _CustomSchema._empty())
            setattr(self.option, 'value', data['option'])
            setattr(self, 'isHomogeneous', data['isHomogeneous'])
            return self

    class _io:
        "input, output"

        def __init__(self, t):
            self.name = t.name
            self.typeStr = t.typeStr
            if isinstance(t.option, int):
                self.option = t.option
            else:
                self.option = t.option.value
            self.description = t.description
            self.isHomogeneous = t.isHomogeneous

        def data(self):
            "Returns all data in that class in a dictionary."
            return {'name': self.name, 'typeStr': self.typeStr,
                    'description': self.description,
                    'isHomogeneous': self.isHomogeneous,
                    'option': self.option}

        def __eq__(self, ot):
            return self.name == ot.name and self.typeStr == ot.typeStr

    class _attribute:
        "attribute"

        def __init__(self, att):
            self.name = att.name
            if isinstance(att.type, int):
                self.type = att.type
            else:
                self.type = att.type.value
            self.default_value = '?'
            self.description = att.description
            self.required = att.required

        def data(self):
            "Returns all data in that class in a dictionary."
            return {'name': self.name, 'type': self.type,
                    'description': self.description,
                    'required': self.required}

        def __eq__(self, ot):
            return self.name == ot.name and self.type == ot.type

    def __init__(self, schema):
        self._schema = schema
        self.domain = schema.domain
        self.name = schema.name
        self.since_version = schema.since_version
        try:
            self.inputs = [_CustomSchema._io(t) for t in schema.inputs]
        except AttributeError as e:  # pragma: no cover
            raise AttributeError(
                "Issue with operator=%r domain=%r since_version=%r, "
                "type(schema)=%r" % (
                    schema.name, schema.domain, schema.since_version,
                    type(schema))) from e
        try:
            self.outputs = [_CustomSchema._io(t) for t in schema.outputs]
        except AttributeError as e:  # pragma: no cover
            raise AttributeError(
                "Issue with operator=%r domain=%r since_version=%r, "
                "type(schema)=%r" % (
                    schema.name, schema.domain, schema.since_version,
                    type(schema))) from e
        self.attributes = {a.name: _CustomSchema._attribute(a)
                           for a in schema.attributes.values()}
        self.min_input = schema.min_input
        self.max_input = schema.max_input
        self.min_output = schema.min_output
        self.max_output = schema.max_output
        self.doc = schema.doc

    _atts = ['domain', 'name', 'since_version', 'inputs', 'outputs',
             'attributes', 'min_input', 'max_input',
             'min_output', 'max_output', 'doc']

    def __eq__(self, ot):
        for k in _CustomSchema._atts:
            if getattr(self, k) == getattr(ot, k):
                continue
            return False
        return True

    def data(self):
        "Returns all data in that class in a dictionary."
        def _(x):
            if x is None:
                return None
            if isinstance(x, (str, int)):
                return x
            if isinstance(x, list):
                return [_(e) for e in x]
            if isinstance(x, dict):
                return {k: _(v) for k, v in x.items()}
            if hasattr(x, 'data'):
                return x.data()
            raise TypeError(  # pragma: no cover
                f"Unable to handle type {type(x)!r} - {x!r}.")

        return {k: _(getattr(self, k)) for k in _CustomSchema._atts}

    def SerializeToString(self):
        "Serializes this class into json."
        return json.dumps(self.data())

    @staticmethod
    def ParseFromString(s):
        "Parses this class from a json string."
        obj = json.loads(s)
        e = _CustomSchema._empty()
        for k in _CustomSchema._atts:
            if k == 'attributes':
                setattr(e, k, {a['name']: _CustomSchema._empty.from_attribute(a)
                               for a in obj[k].values()})
            elif k in ('inputs', 'outputs'):
                setattr(e, k, [_CustomSchema._empty.from_io(o)
                               for o in obj[k]])
            else:
                setattr(e, k, obj[k])
        return _CustomSchema(e)

    def __repr__(self):
        return f"_CustomSchema(**{pprint.pformat(self.data())})"


def _get_all_operator_schema():
    data = os.path.join(os.path.dirname(__file__),
                        "ort_get_all_operator_schema.tmpl")
    with open(data, 'r', encoding='utf-8') as f:
        js = f.readlines()
    return [_CustomSchema.ParseFromString(j) for j in js[1:]]


def _populate_schemas():
    """
    Populates all schemas.
    """
    def _populate_schema(schema):
        # Multiple version can coexist. The last one is kept.
        key = schema.domain, schema.name
        if key in res:
            if schema.since_version > res[key].since_version:
                # We keep the most recent one.
                res[key] = schema
        else:
            res[key] = schema
        full_name = schema.name + '_' + str(schema.since_version)
        res[schema.domain, full_name] = schema
        if key not in versions:
            versions[key] = set()
        if schema.name not in domains:
            domains[schema.name] = set()
        domains[schema.name].add(schema.domain)
        versions[key].add(full_name)

    res = {}
    versions = {}
    domains = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.support_level == schema.SupportType.EXPERIMENTAL:
            # Skips experimental operators.
            continue
        _populate_schema(schema)

    try:
        import onnxruntime.capi.onnxruntime_pybind11_state as rtpy
    except ImportError:  # pragma: no cover
        rtpy = None

    if rtpy is not None:
        # If onnxruntime is available, it is being populated with these operators as well.
        try:
            get_schemas = rtpy.get_all_operator_schema
        except AttributeError:
            # onnxruntime must be compiled with flag --gen_doc.
            # a local copy is retrieved.
            get_schemas = _get_all_operator_schema
        for op in get_schemas():
            if (op.domain, op.name) in res:
                # an existing onnx schema
                continue
            sch = _CustomSchema(op)
            _populate_schema(sch)

    return res, versions, domains


def _find_operator_domain(name):
    """
    Determines the domain of an operator.
    Raises an exception if not found or if there is an ambiguity.

    :param name: operator name
    :return: domain
    """
    if name not in _S.all_domains:
        raise ValueError(
            "Unable to guess domain for operator %r. "
            "Not found in %r." % (name, list(_S.all_domains)))
    domains = _S.all_domains[name]
    if len(domains) == 1:
        return list(domains)[0]
    raise ValueError(  # pragma: no cover
        f"Unable to guess domain of operator {name!r}, found domains {domains!r}.")


def _split_op_name(name):
    spl = name.split('_')
    try:
        i = int(spl[-1])
    except ValueError:
        return name, None
    return "_".join(spl[:-1]), i


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

        if op_version is None:
            if len(args) == 0 and input_range[0] == input_range[1]:
                args = [_[0] for _ in self.__class__.expected_inputs]
            if not (input_range[0] <= len(args) <= input_range[1]):
                raise RuntimeError(  # pragma: no cover
                    "Unexpected number of inputs, "
                    "got {}, expecting {} for operator "
                    "'{}'.".format(
                        len(args), len(inputs), op_name))

        attr_names = self.attr_names
        _, op_version_class = _split_op_name(self.__class__.__name__)
        if op_version_class is not None:
            if op_version is None:
                op_version = op_version_class
            try:
                op_version = min(op_version, op_version_class)
            except TypeError:  # pragma: no cover
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
                name = f'{self.__class__.__name__}_{op}'
                if name in self.past_version:
                    found = (name, op)
                    attr_names = self.past_version[name].attr_names
                    if len(attr_names) > 0 and not isinstance(attr_names[0], str):
                        raise TypeError(  # pragma: no cover
                            "attr_names must be a list of string not a list of %r for "
                            "operator %r and domain %r." % (
                                type(attr_names[0]), name, domain))
                    break
        if (op_version_class is not None and found is not None and
                found[-1] != op_version_class):
            raise RuntimeError(  # pragma: no cover
                "op_version={} does not refer to the same opset as the class "
                "name ('{}').".format(op_version, self.__class__.__name__))
        for key in kwargs:
            if key in {'output_names', 'op_version', 'domain', 'ir_version',
                       'global_context', 'clear_subgraph_inputs'}:
                continue
            if key not in attr_names:
                raise TypeError(  # pragma: no cover
                    "Argument '%s' not valid for '%s' domain=%r opset=%s "
                    "(should be in %r, type(self)=%r)." % (
                        key, op_name, domain, op_version, attr_names,
                        type(self)))

        if op_version is not None:
            kwargs['op_version'] = op_version
        if 'domain' not in kwargs:
            kwargs['domain'] = domain
        # This class can only be created by a user. Let's check
        # types are either a variable, an operator or an array.
        for i, a in enumerate(args):
            if isinstance(a, tuple):
                if len(a) != 2:
                    raise TypeError(  # pragma: no cover
                        "Input %r is a tuple or class %r, it must have two "
                        "elements (name, type) not %r." % (i, class_name, a))
                if not isinstance(a[0], str):
                    raise TypeError(  # pragma: no cover
                        "Input %r is a tuple or class %r, it must be a tuple "
                        "(name, type) not %r." % (i, class_name, a))
                continue
            if not isinstance(a, (
                    Variable, OnnxOperator, numpy.ndarray, str,
                    OnnxOperatorItem, coo_matrix)):
                raise TypeError(  # pragma: no cover
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
                     'op_type': op_name,
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
        operator_names = list(_S.all_schemas_versions)
        if include_past:
            add = []
            for domain, op in operator_names:
                add.extend(
                    [(domain, k)
                     for k in _S.all_schemas_versions[domain, op]])
            operator_names.extend(add)
            operator_names.sort()

    # type verification
    ops = []
    for name in operator_names:
        if isinstance(name, str):
            if name.startswith('Onnx'):
                raise ValueError(
                    f"Operator name cannot start with Onnx: {name!r}.")
            n_name, _ = _split_op_name(name)
            domain = _find_operator_domain(n_name)
            ops.append((domain, name))
        elif isinstance(name, tuple) and len(name) == 2:
            if name[1].startswith('Onnx'):
                raise ValueError(  # pragma: no cover
                    f"Operator name cannot starts with Onnx: {name!r}.")
            ops.append(name)
        else:
            raise ValueError(  # pragma: no cover
                "Operator to fetch must be a string or a "
                "`tuple(domain, name)` not %r." % (name))
    operator_names = ops

    # versions
    res = _S.all_schemas
    cls = {}
    set_names = dict()
    set_skip = set()
    for pos, (op_domain, op_name) in enumerate(operator_names):
        if op_domain == 'ai.onnx':
            op_domain = ''
        set_names[op_domain, op_name] = pos
        n, v = _split_op_name(op_name)
        if v is not None and not include_past:
            set_skip.add((op_domain, n))
            if n not in set_names:
                set_names[op_domain, n] = -1

    if verbose > 1 and fLOG is not None:  # pragma: no cover
        fLOG(f"[_dynamic_class_creation] set_names={set_names!r}")
        fLOG(f"[_dynamic_class_creation] set_skip={set_skip!r}")

    returned_classes = []
    positions = {}

    for (op_domain, op_name), position in set_names.items():
        cl_name = 'Onnx' + _domain_to_class_name(op_domain) + op_name
        if verbose > 3 and fLOG is not None:
            fLOG(  # pragma: no cover
                '[_dynamic_class_creation] cl_name=%r op_domain=%r op_name=%r (in=%d) '
                'position=%r' % (
                    cl_name, op_domain, op_name,
                    1 if cl_name in _S.all_classes else 0,
                    position))
        if cl_name in _S.all_classes:
            if cl_name not in set_skip:
                if position >= 0:
                    returned_classes.append(
                        (position, _S.all_classes[cl_name]))
            continue

        # operator name without domain
        n, v = _split_op_name(op_name)
        if v is not None:
            names = [op_name]
        else:
            try:
                names = _S.all_schemas_versions[op_domain, op_name].copy()
            except KeyError as e:  # pragma: no cover
                raise ValueError(
                    "Operator %r (domain=%r) does not exists." % (
                        op_name, op_domain)) from e
            names.add(op_name)

        if verbose > 0 and fLOG is not None:
            fLOG(  # pragma: no cover
                "[_dynamic_class_creation] op_domain=%r op_name=%r, cl_name=%r names=%r"
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
            args = [p if isinstance(p, str) else p.name
                    for p in schema.attributes]
            if len(args) > 0 and not isinstance(args[0], str):
                raise TypeError(  # pragma: no cover
                    "args must be a list of string not a list of %r for "
                    "operator %r and domain %r." % (
                        type(args[0]), name, op_domain))

            n_name, v = _split_op_name(name)

            if v is not None:
                if op_domain == 'com.microsoft' and name in {
                        'SoftmaxGrad_13', 'LogSoftmaxGrad_13'}:
                    # exception
                    pass
                elif v != schema.since_version:
                    raise ValueError(  # pragma: no cover
                        "Inconsistent version number %d != %d for operator "
                        " %r, %r (%r)." % (
                            v, schema.since_version, schema.domain,
                            schema.name, name))
                class_name = "Onnx" + _domain_to_class_name(op_domain) + name
            else:
                class_name = (
                    "Onnx" + _domain_to_class_name(op_domain) + schema.name)

            if verbose > 0 and fLOG is not None:
                fLOG(  # pragma: no cover
                    "[_dynamic_class_creation] op_name=%r, cl_name=%r cache=%r v=%r"
                    "" % (op_name, class_name, cache, v))

            filename = os.path.join(
                cache_dir,
                schema.name + '_' + str(schema.since_version) + ".rst")
            if not cache and os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:  # pragma: no cover
                    doc = f.read()
            else:
                doc = get_rst_doc(schema.name, domain=schema.domain,
                                  version=schema.since_version)
                if cache:  # pragma: no cover
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
        main, v = _split_op_name(name)
        if v is None:
            continue
        if main in cls:  # pylint: disable=R1715
            last = cls[main]
        else:
            last = _S.all_classes[main]
        last.past_version[name] = cls[name]

    # final
    _S.all_classes.update(cls)
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
        Enables expressions such as:

        ::

            ops = OnnxLoadFactory()
            op = ops.Abs('X')
        """
        if name == '_loaded_classes':
            return self._loaded_classes
        if name in self._loaded_classes:
            return self._loaded_classes[name]
        cl = loadop(name)
        self._loaded_classes[name] = cl
        self._loaded_classes[cl.__name__] = cl
        return cl


class OnnxOperatorBase:
    """
    Base class for @see cl OnnxOperator, @see cl OnnxOperatorItem,
    @see cl OnnxOperatorTuple.
    """

    def __init__(self):
        pass

    def add_to(self, builder):
        "This method should be overwritten."
        raise NotImplementedError(  # pragma: no cover
            f"Not overwritten for class {type(self)!r}.")

    @property
    def output_names(self):
        "This method should be overwritten."
        raise NotImplementedError(  # pragma: no cover
            f"Not overwritten for class {type(self)!r}.")

    def find_named_inputs(self):
        """
        Returns all inputs to the graph.
        """
        raise NotImplementedError(  # pragma: no cover
            f"Method 'find_named_inputs' must be overloaded for type {type(self)}.")

    def f(self, *args, **kwargs):
        """
        Evaluates this node.
        """
        raise NotImplementedError(  # pragma: no cover
            f"Method 'f' must be overloaded for type {type(self)}.")

    def _set_control_op(self, op):
        """
        Tells this operator is part of a subgraph.
        """
        raise NotImplementedError(  # pragma: no cover
            f"Method '_set_control_op' must be overloaded for type {type(self)}.")

    def add_external_input(self, op):
        """
        Tells a subgraph this node comes from the main graph.
        It may be used only by the subgraph but it must be processed as well.
        """
        raise NotImplementedError(  # pragma: no cover
            f"Method '_set_control_op' must be overloaded for type {type(self)}.")


class OnnxOperatorItem(OnnxOperatorBase):
    """
    Accessor to one of the output returned by a @see cl OnnxOperator.

    :param onx_op: @see cl OnnxOperator
    :param index: integer
    :param op_version: defines the opset version
    """

    def __init__(self, onx_op, index, op_version=None):
        OnnxOperatorBase.__init__(self)
        if not isinstance(index, int):
            raise TypeError(  # pragma: no cover
                f"index must be an integer not {type(index)!r}.")
        logger.debug("op:%s-%d(%r, %d, op_version=%r)",
                     self.__class__.__name__, id(self), onx_op, index, op_version)
        if not isinstance(onx_op, OnnxOperatorBase):
            raise TypeError(  # pragma: no cover
                f"onx_op must be an OnnxOperator not {type(onx_op)!r}.")
        self.onx_op = onx_op
        self.index = index
        self.op_version = op_version

    @property
    def output_names(self):
        "Returns None."
        return None

    @property
    def inputs(self):
        "Returns the only inputs in a list."
        return [NodeResultName(self.onx_op, self.index)]

    def add_to(self, builder):
        """
        Adds to graph builder.
        Does nothing because the original node is already added.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        pass

    def __str__(self):
        "usual"
        return "%s[%d]" % (str(self.onx_op), self.index)

    def __repr__(self):
        "usual"
        return "%s(%s[%d])" % (
            self.__class__.__name__,
            self.onx_op.__class__.__name__,
            self.index)

    def get_output_result(self, i=0):
        """
        Returns the output name at position *i*.
        """
        if i != 0:
            raise IndexError(  # pragma: no cover
                "Can only return the first item.")
        return self.onx_op.get_output_result(self.index)

    def _to_onnx_attributes(self, inputs=None, target_opset=None,
                            optim=True, verbose=0, run_shape=True,
                            fLOG=print, processed=None):
        """
        Calls `self.onx_op._to_onnx_attributes`.
        """
        return self.onx_op._to_onnx_attributes(
            inputs=inputs, target_opset=target_opset, optim=optim,
            run_shape=run_shape, verbose=verbose, fLOG=fLOG,
            processed=processed)

    def find_named_inputs(self):
        """
        Returns all inputs to the graph.
        """
        return self.onx_op.find_named_inputs()

    def f(self, *inputs, verbose=0, fLOG=None,  # pylint: disable=W0221
          clear_cache=False, runtime=None):
        """
        Computes the predictions for this node.
        Similar to an eager evaluation.

        :param inputs: inputs as dictionary or a list of inputs
            (see below)
        :param verbose: display information while predicting
        :param fLOG: logging function if *verbose > 0*
        :param clear_cache: onnx graph is created once unless
            this parameter is True
        :param runtime: runtime to use for the evaluation,
            see @see cl OnnxInference
        :return: outputs as a dictionary if the input were given as a
            dictionary or a single result or a tuple otherwise

        The inputs refer to the inputs of the graph.
        The method walks through all inputs and finds inputs defined as
        string. It replaces them by the value found in the dictionary.
        If the inputs are specified in a list, the function retrieves the
        list of inputs defined as a string and assigns them a value.
        Logging function can be used to get more insight about it.
        During the evaluation every node is independently converted
        into ONNX. The ONNX graph is cached in the class itself.
        """
        res = self.onx_op.f(*inputs, verbose=verbose, fLOG=fLOG,
                            clear_cache=clear_cache, runtime=runtime)
        if isinstance(res, dict):
            names = self.onx_op.output_names
            if names is None:
                names = self.onx_op.expected_outputs
                name = names[self.index][0]
            else:
                name = names[self.index]
            return {name: res[name]}
        return res[self.index]


class OnnxOperatorTuple(OnnxOperatorBase):
    """
    Class used to return multiple @see cl OnnxVar
    at the same time.
    """

    def __init__(self, first, *args):
        OnnxOperatorBase.__init__(self)
        logger.debug("op:%s-%d([%r], %d in)",
                     self.__class__.__name__, id(self), type(first),
                     len(args))
        if isinstance(first, (list, tuple)):
            raise TypeError(  # pragma: no cover
                f"Unexpected type for first {type(first)!r}.")
        logger.debug('op:%s-%d(%d in)', self.__class__.__name__,
                     id(self), 1 + len(args))
        if len(args) > 0:
            self.values = (first,) + args
            self.unique = None
        else:
            self.values = None
            self.unique = first
        if self.values is not None and self.unique is not None:
            raise RuntimeError(  # pragma: no cover
                "Unexpected configuration. One member (values or unique) must be "
                "null, unique=%r, values=%r" % (self.unique, self.values))
        if self.values is None and self.unique is None:
            raise RuntimeError(  # pragma: no cover
                "Unexpected configuration. One member (values or unique) must be "
                "not null.")

    def __repr__(self):
        "usual"
        if self.values is None:
            return f"{self.__class__.__name__}({type(self.unique)!r})"
        return "%s(%s)" % (self.__class__.__name__, ", ".join(
            "%r" % type(v) for v in self.values))

    @property
    def inputs(self):
        "Returns the only inputs in a list."
        if self.values is None:
            return [self.unique]
        raise NotImplementedError(  # pragma: no cover
            "OnnxOperatorTuple.inputs is missing.")

    @property
    def external_inputs(self):
        """
        Returns the list of implicit inputs the subgraph
        assumes to be existing even if they are not referenced as
        explicit input for the graph.
        """
        if self.values is None:
            return self.unique.external_inputs
        res = []
        for op in self.values:
            res.extend(op.external_inputs)
        return res

    def add_to(self, builder):
        """
        Adds to graph builder.
        Does nothing because the original node is already added.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        pass

    def __len__(self):
        "usual"
        if self.values is None:
            raise NotImplementedError(  # pragma: no cover
                "Not yet implemented in this case unique=%r, "
                "values=%r." % (self.unique, self.values))
        return len(self.values)

    def __iter__(self):
        "Iterates on the outputs."
        if self.values is None:
            raise NotImplementedError(  # pragma: no cover
                "Not yet implemented in this case.")
        for v in self.values:
            yield v

    def __getitem__(self, i):
        "usual"
        if self.values is None:
            return self.unique[i]
        return self.values[i]

    @property
    def outputs(self):
        "Returns 'output_names' of attribute 'unique'."
        if self.values is None:
            if hasattr(self.unique, 'to_onnx'):
                return self.unique.outputs
        raise NotImplementedError(  # pragma: no cover
            f"Not implemented yet unique={self.unique!r} values={self.values!r}.")

    @property
    def output_names(self):
        "Returns 'output_names' of attribute 'unique'."
        if self.values is None:
            if hasattr(self.unique, 'to_onnx'):
                return self.unique.output_names
        raise NotImplementedError(  # pragma: no cover
            f"Not implemented yet unique={self.unique!r} values={self.values!r}.")

    @output_names.setter
    def output_names(self, value):
        """
        Updates 'output_names' of attribute 'unique'
        or every output name of attribute 'values'.
        """
        logger.debug("op:%s:output_names:set(%r)",
                     self.__class__.__name__, value)
        OnnxIdentity = loadop('Identity')  # pylint: disable=W0621
        if self.values is None:
            if (hasattr(self.unique, 'to_onnx') or
                    hasattr(self.unique, 'add_to')):
                if len(value) > 1:
                    self.values = tuple(
                        OnnxIdentity(
                            self.unique[i], output_names=value[i:i + 1],
                            op_version=self.unique.op_version)
                        for i in range(0, len(value)))
                    self.unique = None
                    return
                self.unique.output_names = [Variable(v) for v in value]
                return
            raise NotImplementedError(  # pragma: no cover
                "Not implemented yet, value=%r, unique=%r values=%r." % (
                    value, self.unique, self.values))
        if self.values is not None and len(self.values) == len(value):
            for name, v in zip(value, self.values):
                v.output_names = [Variable(name)]
            return
        raise NotImplementedError(  # pragma: no cover
            "Not implemented yet, value=%r, unique=%r values=%r." % (
                value, self.unique, self.values))

    def _to_onnx_attributes(self, inputs=None, target_opset=None,
                            optim=True, verbose=0, run_shape=True,
                            fLOG=print, processed=None):
        """
        Calls `self.onx_op._to_onnx_attributes`.
        """
        if self.values is None:
            return self.unique._to_onnx_attributes(
                inputs=inputs, target_opset=target_opset, optim=optim,
                run_shape=run_shape, verbose=verbose, fLOG=fLOG,
                processed=processed)
        res = []
        for v in self.values:
            res.append(v._to_onnx_attributes(
                inputs=inputs, target_opset=target_opset, optim=optim,
                run_shape=run_shape, verbose=verbose, fLOG=fLOG,
                processed=processed))
        return res

    def to_onnx(self, inputs=None, outputs=None,
                other_outputs=None, target_opset=None,
                optim=True, verbose=0, run_shape=True):
        """
        Converts this operator into an ONNX graph.
        It follows the same signature as :meth:`OnnxOperator.to_onnx
        <mlprodict.npy.xop.OnnxOperator.to_onnx>` and calls this
        method of the unique input object or the first one
        if there are several. In that case, other inputs in
        attribute `values` are moved into container
        `other_outputs`. (OnnxOperatorTuple)
        """
        logger.debug('op:%s-%d.to_onnx:%r:%r:%r',
                     self.__class__.__name__, id(self),
                     inputs, outputs, other_outputs)
        logger.indent()
        if self.values is None:
            res = self.unique.to_onnx(
                inputs=inputs, outputs=outputs, other_outputs=other_outputs,
                target_opset=target_opset, optim=optim, verbose=verbose,
                run_shape=run_shape)
            logger.dedent()
            return res
        new_other_outputs = self.values[1:]
        if other_outputs is not None:
            new_other_outputs.extend(other_outputs)
        res = self.values[0].to_onnx(
            inputs=inputs, outputs=outputs, other_outputs=new_other_outputs,
            target_opset=target_opset, optim=optim, verbose=verbose,
            run_shape=run_shape)
        logger.dedent()
        return res


class OnnxOperator(OnnxOperatorBase):
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
    @classmethod
    def __class_getitem__(cls, opset):
        """
        Enables expression `cls[opset]`. It returns the appropriate class
        `cls_opset`. Parameter *op_version* should be specified.
        """
        if not isinstance(opset, int):
            raise ValueError(
                f"opset must an integer not {type(opset)!r}.")
        best = None
        for _, v in cls.past_version.items():
            if v.since_version == opset:
                return lambda *args, **kwargs: v(
                    *args, op_version=opset, **kwargs)
            if v.since_version <= opset and (
                    best is None or best.since_version < v.since_version):
                best = v
        if best is None:
            raise ValueError(
                "Unable to find a version of operator %r and opset %r." % (
                    cls.__name__, opset))
        return lambda *args, **kwargs: best(
            *args, op_version=opset, **kwargs)

    def __init__(self, *inputs, op_version=None, output_names=None,
                 domain=None, global_context=None, **kwargs):

        OnnxOperatorBase.__init__(self)
        logger.debug("op:%s-%d(%d in, op_version=%r, output_names=%r)",
                     self.__class__.__name__, id(self),
                     len(inputs), op_version,
                     output_names)
        if (output_names is None and
                self.__class__.__name__.startswith("OnnxScan")):
            raise NotImplementedError(  # pragma: no cover
                "The class cannot infer the number of variables "
                "for node '{}' yet. output_names must be specified"
                ".".format(self.__class__.__name__))
        if isinstance(output_names, (str, Variable)):
            output_names = [output_names]
            if isinstance(output_names[0], str):
                output_names[0] = Variable(output_names[0])
        elif isinstance(output_names, (list, OnnxOperator._InputContainer)):
            if len(output_names) == 0:
                raise ValueError(  # pragma: no cover
                    "output_names cannot be empty (operator %r)."
                    "" % self.__class__.__name__)
            output_names = output_names.copy()
            for i in range(len(output_names)):  # pylint: disable=C0200
                if isinstance(output_names[i], str):
                    output_names[i] = Variable(output_names[i])
        elif output_names is not None:
            raise TypeError(  # pragma: no cover
                f"output_names must be a string or a list not {type(output_names)!r}.")

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
            raise RuntimeError(  # pragma: no cover
                "Operator '{}': requested version {} < "
                "{} schema version.".format(
                    self.__class__.__name__,
                    self.op_version, self.since_version))

        self.state = None
        self.domain = domain
        self.kwargs = kwargs
        self.max_item_ = None

        # check inputs
        self.inputs = []
        if len(inputs) > 0:
            for inp in inputs:
                if isinstance(inp, str):
                    self.inputs.append(Variable(inp))
                elif isinstance(inp, tuple):
                    if len(inp) != 2:
                        raise RuntimeError(  # pragma: no cover
                            f"Unexpected tuple {inp!r}.")
                    self.inputs.append(
                        Variable(inp[0], dtype=guess_numpy_type(inp[1]),
                                 shape=inp[1].shape))
                elif isinstance(inp, (OnnxOperatorBase, Variable)):
                    self.inputs.append(inp)
                elif isinstance(inp, (numpy.ndarray, coo_matrix, TensorProto)):
                    self.inputs.append(inp)
                elif isinstance(inp, ValueInfoProto):
                    self.inputs.append(inp.type.tensor_type)
                else:
                    raise TypeError(  # pragma: no cover
                        "Unable to interpret the input name for type {} in "
                        "operator '{}' (value={}).".format(
                            type(inp), self.__class__.__name__, inp))

        if (self.inputs is not None and
                (len(self.inputs) < self.input_range[0] or
                    len(self.inputs) > self.input_range[1])):
            raise RuntimeError(  # pragma: no cover
                "Operator '{}' expects a number of inputs in [{}, {}] not {} "
                "(expected opset={}, class opset={})".format(
                    getattr(self, 'operator_name', '?'), *self.input_range,
                    len(self.inputs), op_version, self.op_version))
        # global context
        if global_context is None:
            self.global_context = None
        else:
            if not isinstance(global_context, dict):
                raise TypeError(  # pragma: no cover
                    "global_context must be a dictionary not %r."
                    "" % type(global_context))
            for k, v in global_context.items():
                if not isinstance(v, OnnxOperatorBase):
                    raise TypeError(  # pragma: no cover
                        f"Value {k!r} in must be an OnnxOperatorBase not {type(v)!r}.")
            self.global_context = global_context

        # check output
        self.output_names_ = output_names
        self.output_variables = None

        if self.output_names is not None:
            if len(self.output_names) == 0:
                raise ValueError(  # pragma: no cover
                    "output_names can be None but cannot be empty for "
                    "operator %r." % self)
            if self.output_variables is None:
                self.output_variables = [None for o in self.output_names]
            for i in range(len(self.output_names)):  # pylint: disable=C0200
                name = self.output_names[i]
                if isinstance(name, Variable):
                    self.output_variables[i] = name
                else:
                    raise TypeError(  # pragma: no cover
                        "output_names must be a list of strings "
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
        self.external_inputs = []

    def add_external_input(self, op):
        """
        Tells a subgraph this node comes from a graph calling this one.
        """
        logger.debug("op:%s.add_external_input:%r",
                     self.__class__.__name__, op)
        self.external_inputs.append(op)

    def then_do(self, branch):
        """
        Fills attribute *then_branch*.

        :param branch: onnx graph or @see cl OnnxOperator
        :return: self
        """
        if isinstance(branch, onnx.GraphProto) and len(branch.input) > 0:
            raise RuntimeError(  # pragma: no cover
                "then_branch subgraph cannot have any input.")
        return self._add_subgraph('then_branch', branch)

    def else_do(self, branch):
        """
        Fills attribute *else_branch*.

        :param branch: onnx graph or @see cl OnnxOperator
        :return: self
        """
        if isinstance(branch, onnx.GraphProto) and len(branch.input) > 0:
            raise RuntimeError(  # pragma: no cover
                "else_branch subgraph cannot have any input.")
        return self._add_subgraph('else_branch', branch)

    def _add_subgraph(self, attribute, branch):
        """
        Fills attribute *attribute*.

        :param attribute: attribute name
        :param branch: onnx graph or @see cl OnnxOperator
        :return: self
        """
        if isinstance(branch, str):
            # branch is an input.
            OnnxIdentity = loadop('Identity')
            branch = OnnxIdentity(OnnxExisting(branch),
                                  op_version=self.op_version)
        logger.debug("op:%s:_add_subgraph:%s=type(branch)=%r",
                     self.__class__.__name__, attribute, type(branch))
        if isinstance(branch, onnx.ModelProto):
            return self._add_subgraph(attribute, branch.graph)
        if isinstance(branch, onnx.GraphProto):
            self.kwargs[attribute] = branch
            return self
        if isinstance(branch, OnnxOperator):
            self.kwargs[attribute] = branch
            branch._set_control_op(self)
            return self
        raise TypeError(  # pragma: no cover
            "Unexpected type %r for a subgraph, attribute %r "
            "and class %r." % (
                type(branch), attribute, self.__class__.__name__))

    def _set_control_op(self, op):
        """
        Sets *control_op* for every instance of @see cl OnnxExisting node.

        :param op: operator calling the subgraph.
        """
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, OnnxOperatorBase):
                logger.debug("op:%s-%d:_set_control_op:propagate-into-input:%d:p:%d",
                             self.__class__.__name__, id(self), i, id(op))
                logger.indent()
                inp._set_control_op(op)
                logger.dedent()
        if self.kwargs is None:
            return
        for k, v in self.kwargs.items():
            if isinstance(v, OnnxOperatorBase):
                logger.debug("op:%s-%d:_set_control_op:propagate-into-attribute:%s:p:%d",
                             self.__class__.__name__, id(self), k, id(op))
                logger.indent()
                v._set_control_op(op)
                logger.dedent()

    @property
    def output_names(self):
        "Returns `self.output_names_`."
        return self.output_names_

    @output_names.setter
    def output_names(self, value):
        logger.debug("op:%s:output_names:set(%r)",
                     self.__class__.__name__, value)
        if not isinstance(value, (list, OnnxOperator._InputContainer)):
            raise TypeError(  # pragma: no cover
                f"Value must be a list not {type(value)!r}.")
        res = []
        for v in value:
            if isinstance(v, (Variable, ExistingVariable)):
                res.append(v)
            elif isinstance(v, str):
                res.append(Variable(v))
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for an output_names %r."
                    "" % type(v))
        self.output_names_ = res

    def _check(self):
        input_types = (Variable, OnnxOperatorBase, numpy.ndarray,
                       TensorProto)
        for o in self.inputs:
            if not isinstance(o, input_types):
                raise TypeError(  # pragma: no cover
                    f"Wrong type for inputs {self.inputs!r}.")
        if self.output_names is not None:
            for o in self.output_names:
                if not isinstance(o, Variable):
                    raise TypeError(  # pragma: no cover
                        f"Wrong type for output_names {self.output_names!r}.")

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
                        raise RuntimeError(  # pragma: no cover
                            "Unexpected shape %r for value, it must be "
                            "an array of one element." % value.shape)
                    self.kwargs['value'] = from_array(
                        numpy.array([val], dtype=value.dtype))
                    return
                raise TypeError(  # pragma: no cover
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
            raise RuntimeError(  # pragma: no cover
                "Missing attribute 'past_version', there is "
                "no other available schema.")
        found = None
        for v in self.past_version.values():
            if v.since_version > op_version:
                continue
            if found is None or v.since_version > found.since_version:
                found = v
        if found is None:
            raise RuntimeError(  # pragma: no cover
                "Operator '{}': requested version {} < "
                "{} schema version (past_version {}).".format(
                    self.__class__.__name__,
                    op_version, self.since_version,
                    [v.since_version for v in self.past_version.values()]))
        return found

    def __repr__(self):
        """
        usual
        """
        return "{}({} in) -> {}".format(
            self.__class__.__name__,
            len(self.inputs) if self.inputs is not None else 0,
            [str(o) for o in self.output_names]
            if self.output_names is not None else "?")

    def get_output_result(self, i=0):
        """
        Returns the output name at position *i*.
        """
        return NodeResultName(self, i)

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
            raise RuntimeError(  # pragma: no cover
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
        logger.debug("op:%s-%d.add_to(builder-%d):1",
                     self.__class__.__name__, id(self), id(builder))
        inputs = builder.get_input_names(self, self.inputs)
        if self.output_names is not None:
            n_outputs = len(self.output_names)
        elif self.expected_outputs is not None:
            n_outputs = len(self.expected_outputs)
        else:
            n_outputs = self.output_range[0]
        outputs = [builder.get_unique_output_name(NodeResultName(self, i))
                   for i in range(n_outputs)]
        logger.debug("op:%s-%d.add_to(builder-%d):2:%s:%r:%r",
                     self.__class__.__name__, id(self), id(builder),
                     self.operator_name, inputs, outputs)
        logger.indent()
        builder.add_node(
            self.operator_name,
            builder.get_unique_name(
                '_' + self.operator_name.lower(), reserved=False),
            inputs, outputs, domain=self.domain, opset=self.op_version,
            **self.kwargs)
        logger.dedent()
        logger.debug("op:%s-%d.add_to(builder-%d):3",
                     self.__class__.__name__, id(self), id(builder))

    @staticmethod
    def _node_to_graph_preprocess_list(inputs):
        new_inputs = OrderedDict()
        for el in inputs:
            if isinstance(el, str):
                new_inputs[el] = Variable(el)
            elif isinstance(el, Variable):
                new_inputs[el.name] = el
            elif isinstance(el, tuple) and len(el) == 2:
                # sklearn-onnx
                new_inputs[el[0]] = Variable(
                    el[0], guess_numpy_type(el[1]), el[1].shape)
            elif isinstance(el, ValueInfoProto):
                new_inputs[el.name] = el
            else:
                raise TypeError(  # pragma: no cover
                    f"Unable to handle input type {type(el)!r} ({el!r}).")
        return new_inputs

    @staticmethod
    def _node_to_graph_process_input(processed, inputs, set_inputs, node, inp,
                                     new_inputs, new_stack, inputs_dtype,
                                     as_function=False):
        if not as_function and inputs is None and inputs_dtype is None:
            raise RuntimeError(  # pragma: no cover
                "Both inputs and inputs_dtype cannot be None at the same time "
                "for inp=%r." % (inp, ))

        if isinstance(inp, OnnxExisting):
            if inp.inputs[0].output_names is None:
                raise RuntimeError(  # pragma: no cover
                    "output_names cannot be None for OnnxExisting, "
                    "subop is %r." % (inp.inputs[0], ))
            # We need to check that this input was not already added.
            oinp = inp.inputs[0].output_names[0]
            if not new_inputs.has_input(oinp) and id(inp.inputs[0]) not in processed:
                raise RuntimeError(  # pragma: no cover
                    "This node id=%d (%r) was not added yet in the subgraph "
                    "but it must be from node %r." % (
                        id(inp.inputs[0]), inp.inputs[0], node))
        elif isinstance(inp, OnnxOperator):
            new_stack.append(inp)
            logger.debug("op:static:SG-op:processed[%d]:%s",
                         id(inp), inp.__class__.__name__)
            processed[id(inp)] = inp
        elif isinstance(inp, OnnxOperatorItem):
            new_stack.append(inp)
            logger.debug("op:static:SG-it:processed[%d]:%s",
                         id(inp), inp.__class__.__name__)
            processed[id(inp)] = inp
            new_stack.append(inp.onx_op)
            logger.debug("op:static:SG-op:processed[%d]:%s",
                         id(inp.onx_op), inp.onx_op.__class__.__name__)
            processed[id(inp.onx_op)] = inp.onx_op
        elif isinstance(inp, OnnxOperatorTuple):
            # new_stack.append(inp)
            # new_stack.append(inp.onx_op)
            raise NotImplementedError(  # pragma: no cover
                "Unable to guess inputs when one input is OnnxOperatorTuple.")
        elif isinstance(inp, Variable):
            if inp.name in set_inputs:
                return
            if inp.name == '':
                return
            logger.debug("op:static:SG-var:processed[%d]:%s",
                         id(inp), inp.__class__.__name__)
            processed[id(inp)] = inp
            set_inputs.add(inp.name)
            if inputs is None and inputs_dtype is None:
                new_inputs.append(InputDetectedVariable(node, inp))
            elif isinstance(inputs, dict):
                if inp.name in inputs:
                    var = InputDetectedVariable(
                        node, inp.copy_merge(inputs[inp.name]))
                    new_inputs.append(var)
                else:
                    raise ValueError(  # pragma: no cover
                        f"Unable to find input {inp!r} in {inputs!r}.")
            elif inputs_dtype is not None:
                new_inputs.append(
                    InputDetectedVariable(node, inp.copy_add(inputs_dtype)))
            elif isinstance(inputs, Variable):
                if inp.name == inputs.name:
                    new_inputs.append(
                        InputDetectedVariable(node, inp.copy_merge(inputs)))
                else:
                    new_inputs.append(InputDetectedVariable(node, inp))
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Unable to handle inputs={inputs!r}.")
        elif isinstance(inp, numpy.ndarray):
            pass
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected input type {type(inp)!r} in node type {type(node)!r}.")

    @staticmethod
    def _node_to_graph_get_type(node, name=None, outputs=None,
                                outputs_dtype=None):
        if outputs is None:
            return outputs_dtype, None
        if isinstance(outputs, Variable):
            if name is None:
                return (outputs.dtype or outputs_dtype, None)
            if isinstance(name, Variable):
                return (outputs.dtype or name.dtype or outputs_dtype,
                        None)
            raise RuntimeError(  # pragma: no cover
                f"Unable to handle outputs={outputs!r}.")
        if isinstance(outputs, dict):
            if name is None:
                return _infer_node_output(node, outputs)
            if isinstance(name, Variable):
                n = name.name
            else:
                n = name
            if n not in outputs:
                return None, None
            return outputs[n], None
        if isinstance(outputs, (list, OnnxOperator._InputContainer)):
            raise NotImplementedError(  # pragma: no cover
                f"Unexpected type for name={name!r}, outputs={outputs!r}.")
        if is_numpy_dtype(outputs):
            return outputs, None
        raise RuntimeError(  # pragma: no cover
            f"Unable to handle outputs={outputs!r}.")

    @staticmethod
    def _node_to_graph_reorder_by_name(new_inputs, inputs):
        memo = OrderedDict((n.name, n) for n in new_inputs)
        done = set()
        result = []
        for inp in inputs:
            if inp.name in memo:
                result.append(memo[inp.name])
                done.add(inp.name)
        for k, v in memo.items():
            if k in done:
                continue
            result.append(v)
        return result

    class _InputContainer:

        def __init__(self):
            self._c = []
            self._names = set()

        def has_input(self, inp):
            "Checks that input *inp* is part the list of names."
            if inp.name in self._names:
                return True
            return False

        def append(self, inp):
            "Append one element to the list."
            name = inp.var.name
            self._c.append(inp)
            self._names.add(name)

        def __len__(self):
            return len(self._c)

        def __repr__(self):
            return f"{'_InputContainer'}(\n {pprint.pformat(self._c)})"

        def __iter__(self):
            for inp in self._c:
                yield inp

    def _node_to_graph(self, other_outputs=None, inputs=None, outputs=None,
                       as_function=False, processed=None):
        """
        Builds a graph as a list of nodes to walk through in that order.
        """
        if processed is None:
            raise RuntimeError(  # pragma: no cover
                "processed cannot be None.")
        node_outputs = [self]
        if other_outputs is not None:
            node_outputs += other_outputs

        if inputs is not None:
            logger.debug("op:%s-%d._node_to_graph:1:inputs=%r",
                         self.__class__.__name__, id(self), inputs)
        if outputs is not None:
            logger.debug("op:%s-%d._node_to_graph:1:outputs=%r",
                         self.__class__.__name__, id(self), outputs)

        # preprocess inputs, outputs
        _keep_inputs = None
        inputs_dtype = None
        if isinstance(inputs, (list, OnnxOperator._InputContainer)):
            _keep_inputs = inputs
            inputs_dict = self._node_to_graph_preprocess_list(inputs)
        elif isinstance(inputs, dict):
            inputs_dict = inputs
        elif isinstance(inputs, Variable):
            inputs = [inputs]
            inputs_dict = self._node_to_graph_preprocess_list(inputs)
        elif is_numpy_dtype(inputs):
            inputs_dtype = inputs
            inputs_dict = None
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected type {type(inputs)!r} for inputs.")

        _keep_outputs = None
        outputs_dtype = None
        if isinstance(outputs, (list, OnnxOperator._InputContainer)):
            _keep_outputs = outputs
            outputs_dict = self._node_to_graph_preprocess_list(outputs)
        elif isinstance(outputs, dict):
            outputs_dict = outputs
        elif isinstance(outputs, Variable):
            outputs = [outputs]
            outputs_dict = self._node_to_graph_preprocess_list(outputs)
        elif is_numpy_dtype(outputs):
            outputs_dtype = outputs
            outputs_dict = None
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected type {type(outputs)!r} for outputs.")

        if inputs is not None:
            logger.debug("op:%s-%d._node_to_graph:2:inputs=%r",
                         self.__class__.__name__, id(self), inputs)
        if outputs is not None:
            logger.debug("op:%s-%d._node_to_graph:2:outputs=%r",
                         self.__class__.__name__, id(self), outputs)
        if inputs_dict is not None:
            logger.debug("op:%s-%d._node_to_graph:2:inputs_dict=%r",
                         self.__class__.__name__, id(self), inputs_dict)
        if outputs_dict is not None:
            logger.debug("op:%s-%d._node_to_graph:2:outputs_dict=%r",
                         self.__class__.__name__, id(self), outputs_dict)
        if inputs_dtype is not None:
            logger.debug("op:%s-%d._node_to_graph:2:inputs_dtype=%r",
                         self.__class__.__name__, id(self), inputs_dtype)
        if outputs_dtype is not None:
            logger.debug("op:%s-%d._node_to_graph:2:outputs_dtype=%r",
                         self.__class__.__name__, id(self), outputs_dtype)

        # walk through graph
        stack = list(node_outputs)
        new_inputs = self._InputContainer()
        set_inputs = set()
        memo = []
        while len(stack) > 0:
            logger.debug("op:%s-%d._node_to_graph:loop:len(memo)=%d",
                         self.__class__.__name__, id(self), len(memo))
            memo.extend(stack)
            new_stack = []
            for obj in stack:
                logger.debug("op:%s-%d._node_to_graph:-node=%r:external_inputs=%r",
                             self.__class__.__name__, id(self),
                             obj.__class__.__name__,
                             getattr(obj, 'external_inputs', "-"))
                if isinstance(obj, OnnxExisting):
                    pass
                elif isinstance(obj, OnnxOperatorItem):
                    # nothing to do, OnnxOperatorItem is created
                    # by OnnxOperator.__getitem__.
                    pass
                elif isinstance(obj, (OnnxOperator, OnnxOperatorTuple)):
                    if len(obj.external_inputs) > 0:
                        # external_inputs are inputs required by a subgraph
                        # but not necessarily used in the main graph.
                        # They need to be processed first.
                        for inp in obj.external_inputs:
                            self._node_to_graph_process_input(
                                processed, inputs_dict, set_inputs, obj, inp, new_inputs,
                                new_stack, inputs_dtype, as_function=as_function)
                    for inp in obj.inputs:
                        self._node_to_graph_process_input(
                            processed, inputs_dict, set_inputs, obj, inp, new_inputs,
                            new_stack, inputs_dtype, as_function=as_function)
                else:
                    raise TypeError(  # pragma: no cover
                        f"Unexpected type {type(obj)!r}.")
            stack = new_stack

        # reorder new_inputs to follow inputs initial order
        if _keep_inputs is not None:
            new_inputs = self._node_to_graph_reorder_by_name(
                new_inputs, inputs)

        logger.debug("op:%s-%d._node_to_graph:new_inputs=%r",
                     self.__class__.__name__, id(self), new_inputs)

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
                    to, shape = self._node_to_graph_get_type(
                        node, outputs=outputs_dict,
                        outputs_dtype=outputs_dtype)
                    if to is None:
                        run_shape = True
                    res = '???_%d' % i
                    var = Variable(res, added_dtype=to, shape=shape)
                    if var.name in set_names:
                        raise RuntimeError(  # pragma: no cover
                            f"Duplicated output name var={var!r}.")
                    set_names.add(var.name)
                    new_outputs.append(OutputDetectedVariable(node, var, i))
            else:
                for i, o in enumerate(node.output_names):
                    if isinstance(o, str):
                        raise TypeError(  # pragma: no cover
                            "Output %d - %r (%r) not allowed in node %r." % (
                                i, o, node.output_names, node))
                    to, shape = self._node_to_graph_get_type(
                        node, o, outputs=outputs_dict,
                        outputs_dtype=outputs_dtype)
                    if to is None:
                        run_shape = True
                    res = (o, to)
                    var = o.copy_merge(to, shape=shape)
                    if var.name in set_names:
                        raise RuntimeError(  # pragma: no cover
                            f"Duplicated output name o={o!r} var={var!r}.")
                    set_names.add(var.name)
                    new_outputs.append(OutputDetectedVariable(node, var, i))
        if len(new_outputs) == 0:
            raise RuntimeError(  # pragma: no cover
                f"No detected outputs inputs={inputs_dict!r} outputs={outputs_dict!r}.")

        # reorder new_outputs to follow outputs initial order
        if _keep_outputs is not None:
            new_outputs = self._node_to_graph_reorder_by_name(
                new_outputs, outputs)

        logger.debug("op:%s-%d._node_to_graph:new_outputs=%r",
                     self.__class__.__name__, id(self), new_outputs)

        return nodes, new_inputs, new_outputs, run_shape

    def to_onnx(self, inputs=None, outputs=None,
                other_outputs=None, target_opset=None,
                optim=True, verbose=0, run_shape=True,
                function_name=None, function_domain=None,
                fLOG=print, processed=None, check_model=True,
                return_builder=False):
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
        :param optim: optimize the model with function
            @see fn onnx_optimisations
        :param run_shape: in case output shapes are not specify,
            the function runs function :epkg:`infer_shapes`
            to guess them, False would disable that
            default behaviour
        :param verbose: prints information
        :param function_name: if not None, returns a :epkg:`FunctionProto`
        :param function_domain: in case of a function, declares the function
            as part of this domain
        :param fLOG: logging function
        :param processed: keeps track the of the processed nodes
        :param check_model: checks the output model
        :param return_builder: if True, returns the instance of @see cl GraphBuilder
            used to build the onnx graph.
        :return: ONNX stucture

        *inputs* and *outputs* parameters work the same way.
        Here is some possible walues:
            - `inputs=numpy.float32`: all inputs are dense tensors of
              unknown shapes sharing the same element type
            - `inputs={'X': numpy.float32`, 'Y': numpy.in64}`:
              input `X` is a dense tensor of float32,
              input `Y` is a dense tensor of int64,
            - `{'X': numpy.array(...)}}`: input `X` is a dense
              tensor with a precise shape
            - `inputs=[Variable('X', numpy.float32, [1, 2])]`:
              input `X` is a dense tensor of float32 with shape `[1, 2]`
            - `inputs=[Variable('X', numpy.float32, [None, 2])]`:
              input `X` is a dense tensor of float32 with a 2D tensor
              with an unknown dimension (first one)
            - see @see cl Variable

        (OnnxOperator)
        """
        # opsets
        logger.debug(
            "op:%s-%d.to_onnx(%r, %r, other_outputs=%r, target_opset=%r, as_function=%r)",
            self.__class__.__name__, id(self), inputs, outputs,
            other_outputs, target_opset, function_name)
        if isinstance(target_opset, dict):
            dom = self.domain or ''
            target_opset = target_opset.get(dom, None)
        elif isinstance(target_opset, int):
            if self.domain not in ('', None):
                # The target_opset is for the domain '' we ignore it.
                target_opset = None
        elif target_opset is not None:
            raise TypeError(  # pragma: no cover
                "target_opset must be a dictionary {domain: "
                "target_opset} not %r for operator %r." % (
                    target_opset, self.__class__.__name__))

        if self.domain in ('', None) and target_opset == 1:
            raise RuntimeError(  # pragma: no cover
                "target_opset cannot be 1.")
        if (self.op_version is not None and target_opset is not None and
                self.op_version > target_opset):
            raise RuntimeError(  # pragma: no cover
                "target_opset={} is lower than the version={} requested "
                "for this node '{}'.".format(
                    target_opset, self.op_version, self.__class__.__name__))

        # get the graph
        if processed is None:
            processed = {}
        logger.debug("op:%s-%d:SG-self:processed[%d]:SELF",
                     self.__class__.__name__, id(self), id(self))
        processed[id(self)] = self

        logger.indent()
        nodes, graph_inputs, graph_outputs, run_shape2 = self._node_to_graph(
            other_outputs, inputs, outputs, as_function=function_name is not None,
            processed=processed)
        logger.dedent()

        logger.debug("op:%s.to_onnx:graph_inputs=%r",
                     self.__class__.__name__, graph_inputs)
        logger.debug("op:%s.to_onnx:graph_outputs=%r",
                     self.__class__.__name__, graph_outputs)

        if len(nodes) == 0:
            raise RuntimeError(  # pragma: no cover
                "Node list is empty.")
        if verbose > 1:
            for i, n in enumerate(nodes):  # pragma: no cover
                fLOG("nodes[%d]=%r" % (i, n))
            for i, n in enumerate(graph_inputs):  # pragma: no cover
                fLOG("graph_inputs[%d]=%r" % (i, n))

        # creates a _GraphBuilder
        builder = _GraphBuilder()

        # reserve input names starting by the first one
        for node in reversed(nodes):
            for var in node.inputs:
                if isinstance(var, Variable):
                    logger.debug("op:%s.to_onnx:_add_name(%r)",
                                 self.__class__.__name__, var.name)
                    builder._add_name(var.name)

        # reserve output names starting by the last ones
        for node in reversed(nodes):
            builder.reserve_names(node, node.output_names)

        # adds every node to the builder
        for i, node in enumerate(nodes):
            logger.debug("op:%s-%d.to_onnx:node:%d/%d:%r",
                         self.__class__.__name__, id(self), i, len(nodes), node)

        for node in nodes:
            if isinstance(node, OnnxExisting):
                continue
            logger.indent()
            hidden = node._to_onnx_attributes(
                inputs=graph_inputs, target_opset=target_opset,
                optim=optim, verbose=verbose, run_shape=run_shape, fLOG=fLOG,
                processed=processed)
            logger.dedent()

            if len(hidden) > 0:
                logger.debug(
                    "op:%s-%d.to_onnx:to_onnx:%s-%d:hidden:%r",
                    self.__class__.__name__, id(self),
                    node.__class__.__name__, id(node), hidden)
                builder.get_input_names(node, hidden)
            node.add_to(builder)

        logger.debug(
            "op:%s-%d.to_onnx:to_onnx:a", self.__class__.__name__, id(self))

        logger.indent()
        onx = builder.to_onnx(
            inputs=graph_inputs, outputs=graph_outputs,
            target_opset=target_opset, verbose=verbose,
            optim=optim, run_shape=run_shape and run_shape2,
            function_name=function_name, function_domain=function_domain,
            check_model=check_model)
        logger.dedent()

        logger.debug(
            "op:%s-%d.to_onnx:to_onnx:b:%s:%d-nodes",
            self.__class__.__name__, id(self), type(onx).__name__,
            len(onx.graph.node) if hasattr(onx, 'graph') else onx.node)
        if return_builder:
            return onx, builder
        return onx

    def _to_onnx_attributes(self, inputs=None, target_opset=None,
                            optim=True, verbose=0, run_shape=True,
                            fLOG=print, processed=None):
        """
        Converts attributes into ONNX.
        Returns the hidden inputs.
        """
        if processed is None:
            raise RuntimeError(  # pragma: no cover
                "processed cannot be None.")
        converts = []
        for k, v in self.kwargs.items():
            if isinstance(v, OnnxOperatorBase):
                converts.append(k)
        hidden_inputs = []
        for name in converts:
            if verbose > 0:
                fLOG(  # pragma: no cover
                    '[OnnxOperator._to_onnx_attributes] process %r of type %r.'
                    '' % (name, type(self.kwargs[name])))
            model, hidden = self._to_onnx_attribute(
                self.kwargs[name], inputs=inputs, target_opset=target_opset,
                optim=optim, verbose=verbose, run_shape=run_shape, fLOG=fLOG,
                processed=processed)
            hidden_inputs.extend(hidden)
            if len(model.graph.node) == 0:
                _, hidden = self._to_onnx_attribute(
                    self.kwargs[name], inputs=inputs, target_opset=target_opset,
                    optim=False, verbose=verbose, run_shape=run_shape, fLOG=fLOG,
                    processed=processed)
                raise RuntimeError(  # pragma: no cover
                    "Conversion to graph of parameter %r from\nnode=%r "
                    "and\ninputs=%r\nis empty:\n%s\nHIDDEN\n%r" % (
                        name, self.kwargs[name], self.kwargs[name].inputs,
                        model, hidden))
            if name in {'else_branch', 'then_branck'}:
                if len(model.graph.input) > 0:
                    # else_branch, then_branch must not have any input.
                    del model.graph.input[:]
            self.kwargs[name] = model.graph
        return hidden_inputs

    def _to_onnx_attribute(self, oxop, inputs=None, target_opset=None,
                           optim=True, verbose=0, run_shape=True,
                           fLOG=print, processed=None):
        """
        Converts one subgraph into ONNX.
        Returns the ONNX graph and the hidden inputs.
        """
        if processed is None:
            raise RuntimeError(  # pragma: no cover
                "processed cannot be None.")
        if inputs is None:
            vars = None
        else:
            named_inputs = set(oxop.find_named_inputs())
            vars = []
            added = set()
            for inp in inputs:
                if inp.var.name in named_inputs and inp.var.name not in added:
                    added.add(inp.var.name)
                    vars.append(Variable(
                        inp.var.name, inp.var.dtype or inp.var.added_dtype))
            if verbose > 0:
                fLOG(  # pragma: no cover
                    f'[OnnxOperator._to_onnx_attribute] inputs={vars!r}')
            logger.debug("op:%s._to_onnx_attribute:inputs(%r)",
                         self.__class__.__name__, vars)
        logger.indent()
        onx, att_builder = oxop.to_onnx(
            inputs=vars, target_opset=target_opset, run_shape=run_shape,
            verbose=verbose, fLOG=fLOG, processed=processed, optim=False,
            check_model=False, return_builder=True)
        logger.dedent()
        hidden_inputs = att_builder.hidden_input
        if len(hidden_inputs) > 0:
            if verbose > 0:
                fLOG(  # pragma: no cover
                    f'[OnnxOperator._to_onnx_attribute] inputs={vars!r}')
            logger.debug("op:%s._to_onnx_attribute:inputs:hidden:%r",
                         self.__class__.__name__, att_builder.hidden_input)
        if len(onx.graph.node) == 0:
            raise RuntimeError(  # pragma: no cover
                "Empty graph (class=%r, optim=%r) from\nnode=%r "
                "and\ninputs=%r\nis empty:\n%s" % (
                    type(oxop), optim, oxop, vars, onx))
        shaped_onx = infer_shapes(onx)
        return shaped_onx, hidden_inputs

    def predecessors(self):
        """
        Returns the list of predecessors.

        :return: list of @see cl OnnxOperator
        """
        stack = [self]
        last = 0
        while True:
            end = len(stack)
            if end == last:
                break
            for i in range(last, end):
                node = stack[i]
                for inp in node.inputs:
                    if isinstance(inp, OnnxOperatorBase):
                        stack.append(inp)
            last = end
        return stack

    def __call__(self, *args, function_name=None, function_domain=None,
                 **kwargs):
        """
        Creates an instance of class @see cl OnnxOperatorFunction.
        Equivalent to `OnnxOperatorFunction(proto, *args, **kwargs)`.

        :param args: see @see cl OnnxOperatorFunction
        :param function_name: name to be given to the function
        :param function_domain: function domain, if None,
            it is given a default value
        :param kwargs: see @see cl OnnxOperatorFunction
        :return: instance of type @see cl OnnxOperatorFunction
        """
        if function_name is None:
            def clean(name):
                if name.startswith("Onnx"):
                    name = name[4:]
                return name

            pred = self.predecessors()
            cls = [clean(p.__class__.__name__) for p in pred]
            function_name = "".join(cls)
        onx = self.to_onnx(function_name=function_name,
                           function_domain=function_domain)
        return OnnxOperatorFunction(onx, *args, **kwargs)

    def find_named_inputs(self):
        """
        Retrieves all named inputs in this graph.
        """
        unique = set()
        found = []
        for inp in self.inputs:
            if isinstance(inp, str):
                if inp not in unique:
                    found.append(inp)
                    unique.add(inp)
            elif isinstance(inp, Variable):
                if inp.name not in unique:
                    found.append(inp.name)
                    unique.add(inp.name)
            elif isinstance(inp, OnnxOperatorBase):
                f = inp.find_named_inputs()
                for n in f:
                    if n not in unique:
                        found.append(n)
                        unique.add(n)
            elif isinstance(inp, numpy.ndarray):
                pass
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Unexpected input type {type(inp)!r}.")
        return found

    def to_onnx_this(self, evaluated_inputs):
        """
        Returns a simple ONNX graph corresponding to this node.

        :param evaluated_inputs: inputs as a list
        :return: ONNX graph
        """
        logger.debug('op:%s-%d.to_onnx_this:%r',
                     self.__class__.__name__, id(self),
                     evaluated_inputs)
        inputs_names = ['I%d' % i for i in range(len(evaluated_inputs))]
        if self.output_names is None:
            if self.expected_outputs is None:
                raise NotImplementedError(  # pragma: no cover
                    "expected_outputs and output_names are not defined.")
            output_names = [o[0] for o in self.expected_outputs]
        else:
            output_names = [o.name for o in self.output_names]
        node = make_node(self.op_type, inputs_names, output_names,
                         domain=self.domain, name="f", **self.kwargs)
        onx_inputs = [Variable(name, a.dtype).make_value_info()
                      for name, a in zip(inputs_names, evaluated_inputs)]
        onx_outputs = [make_value_info(name, make_tensor_type_proto(0, []))
                       for name in output_names]
        graph = make_graph([node], 'f', onx_inputs, onx_outputs)
        model = make_model(
            graph, opset_imports=[make_operatorsetid(
                self.domain or '', self.since_version)])
        return model

    def run(self, *inputs, verbose=0, fLOG=None, clear_cache=False, runtime=None):
        """
        Other name for
        `OnnxInference.f <mlprodict.onnxrt.onnx_inference.OnnxInference.f>`_.
        """
        return self.f(*inputs, verbose=verbose, fLOG=fLOG,
                      clear_cache=clear_cache, runtime=runtime)

    def f(self, *inputs, verbose=0, fLOG=None,  # pylint: disable=W0221
          clear_cache=False, runtime=None):
        """
        Computes the predictions for this node.
        Similar to an eager evaluation.

        :param inputs: inputs as dictionary or a list of inputs
            (see below)
        :param verbose: display information while predicting
        :param fLOG: logging function if *verbose > 0*
        :param clear_cache: onnx graph is created once unless
            this parameter is True
        :param runtime: runtime to use for the evaluation,
            see @see cl OnnxInference
        :return: outputs as a dictionary if the input were given as a
            dictionary or a single result or a tuple otherwise

        The inputs refer to the inputs of the graph.
        The method walks through all inputs and finds inputs defined as
        string. It replaces them by the value found in the dictionary.
        If the inputs are specified in a list, the function retrieves the
        list of inputs defined as a string and assigns them a value.
        Logging function can be used to get more insight about it.
        During the evaluation every node is independently converted
        into ONNX. The ONNX graph is cached in the class itself.
        """
        # input evaluation
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            dict_inputs = inputs[0]
            as_dict = True
        elif not isinstance(inputs, (tuple, list, OnnxOperator._InputContainer)):
            raise TypeError(  # pragma: no cover
                f"inputs must be a list not {type(inputs)!r}.")
        elif len(inputs) > 0 and isinstance(inputs[0], OnnxOperator):
            raise TypeError(  # pragma: no cover
                f"Unexpected type for inputs[0]: {type(inputs[0])!r}.")
        else:
            as_dict = False
            if verbose > 0:
                fLOG(  # pragma: no cover
                    "[OnnxOperator.f] retrieves named inputs")
            if hasattr(self, "feval_named_inputs_"):
                named_inputs = self.feval_named_inputs_  # pylint: disable=E0203
            else:
                named_inputs = self.find_named_inputs()
                self.feval_named_inputs_ = named_inputs
            if len(named_inputs) != len(inputs):
                raise RuntimeError(
                    "Mismatch between the number of found inputs (%d) and "
                    "the number of given inputs (%d) (found %r)."
                    "" % (
                        len(named_inputs), len(inputs), named_inputs))
            dict_inputs = {
                name: value for name, value in zip(named_inputs, inputs)}
            if verbose > 0:
                fLOG(  # pragma: no cover
                    f"[OnnxOperator.f] found inputs: {named_inputs!r}")

        # conversion
        evaluated_inputs = []
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, str):
                evaluated_inputs.append(dict_inputs[inp])
            elif isinstance(inp, Variable):
                evaluated_inputs.append(dict_inputs[inp.name])
            elif isinstance(inp, OnnxOperatorBase):
                if verbose > 0:
                    fLOG(  # pragma: no cover
                        "[OnnxOperator.f] evaluate input %d (op_type=%r)" % (
                            i, self.__class__.op_type))
                out = inp.f(dict_inputs, verbose=verbose, fLOG=fLOG)
                if isinstance(out, dict):
                    if len(out) == 1:
                        evaluated_inputs.append(out.popitem()[1])
                    else:
                        raise NotImplementedError(  # pragma: no cover
                            "Not yet implemented in case when there are multiple "
                            "outputs (%r)." % list(out))
                elif isinstance(out, (list, OnnxOperator._InputContainer)):
                    evaluated_inputs.extend(out)
                else:
                    evaluated_inputs.append(out)
            elif isinstance(inp, numpy.ndarray):
                evaluated_inputs.append(inp)
            else:
                raise RuntimeError(  # pragma: no cover
                    "Unexpected type %r for input %d." % (type(inp), i))

        # conversion to ONNX
        if not hasattr(self, 'feval_onnx_'):
            self.feval_onnx_ = {}
        key = tuple((m.dtype, m.shape) for m in evaluated_inputs)
        if key not in self.feval_onnx_ or clear_cache:
            if verbose > 0:
                fLOG(
                    f"[OnnxOperator.f] creating node {self.op_type!r}, inputs={key!r}")
            from ..onnxrt import OnnxInference
            model = self.to_onnx_this(evaluated_inputs)
            oinf = OnnxInference(model, runtime=runtime)
            self.feval_onnx_[key] = oinf
        else:
            oinf = self.feval_onnx_[key]

        # execution
        if verbose > 0:
            fLOG(f"[OnnxOperator.f] execute node {self.op_type!r}")
        got = oinf.run({k: v for k, v in
                        zip(oinf.input_names, evaluated_inputs)})
        if as_dict:
            return got
        if len(got) == 1:
            return got.popitem()[1]
        return [got[n] for n in oinf.output_names]

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
        elif isinstance(n2, OnnxOperatorTuple):
            raise NotImplementedError(  # pragma: no cover
                "_merge_op_version is not implemented when n2 "
                "is OnnxOperatorTuple.")
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

    def astype(self, to):
        """
        Automatically adds operator `OnnxCast` to the graph.

        :param ov: onnx node
        :return: `OnnxCast(self, ov, to=to)`
        """
        OnnxCast = loadop('Cast')
        return OnnxCast(self, to=to, op_version=self.op_version)


class OnnxOperatorFunction(OnnxOperator):
    """
    This operator is used to insert existing ONNX function into
    the ONNX graph being built.

    :param function_proto: instance of type :epkg:`FunctionProto`
    :param inputs: inputs
    :param output_names: output names
    :param sub_functions: functions called by this one
    """

    domain = 'mlprodict'
    since_version = 1
    expected_inputs = None
    expected_outputs = None
    input_range = [1, 1e9]
    output_range = [1, 1e9]
    op_type = 'Function'
    domain = 'mlprodict.xop'

    @staticmethod
    def attribute_to_value(att):
        """
        Converts an attribute into a value using python structures.
        """
        if isinstance(att, onnx.AttributeProto):
            dtype = att.type
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Unable to copy attribute type {type(att)!r}.")
        if dtype == 1:  # .f
            value = att.f
        elif dtype == 2:  # .i
            value = att.i
        elif dtype == 3:  # .s
            value = att.s
        elif dtype == 4:  # .t
            value = att.t
        elif dtype == 6:  # .floats
            value = list(att.floats)
        elif dtype == 7:  # .ints
            value = list(att.ints)
        elif dtype == 8:  # .strings
            value = list(att.strings)
        elif dtype == 11:  # .double_data
            value = list(att.double_data)
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Unable to copy attribute type {dtype!r} ({att!r}).")
        return value

    def __init__(self, function_proto, *inputs, output_names=None,
                 sub_functions=None):
        logger.debug("op:Function(ONNX, %d in, output_names=%r)",
                     len(inputs), output_names)
        if function_proto is None:
            raise ValueError(
                "function_proto cannot be None.")  # pragma: no cover
        if not isinstance(function_proto, onnx.FunctionProto):
            raise TypeError(  # pragma: no cover
                "function_proto must be of type FunctionProto not %r." %
                type(function_proto))
        if len(inputs) > len(function_proto.input):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of inputs %r > expected %r." % (
                    len(inputs), len(function_proto.input)))
        if (output_names is not None and
                len(output_names) != len(function_proto.output)):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of outputs %r != expected %r." % (
                    len(output_names), len(function_proto.output)))
        OnnxOperator.__init__(self, *inputs, output_names=output_names)
        self.model = function_proto
        self.sub_functions = sub_functions

    def __repr__(self):
        "usual"
        atts = {}
        for att in ['output_names']:
            value = getattr(self, att, None)
            if value is not None:
                atts[att] = value
        atts.update(self.kwargs)
        if self.sub_functions is not None and len(self.sub_functions) > 0:
            atts["sub_functions"] = list(range(len(self.sub_functions)))
        msg = ", ".join(f"{k}={v!r}" for k, v in atts.items())
        if len(atts) > 0:
            msg = ", " + msg
        return f"{self.__class__.__name__}(...{msg})"

    def add_to(self, builder):
        """
        Adds to graph builder.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        logger.debug("op:Function.add_to(builder)")
        inputs = builder.get_input_names(self, self.inputs)
        n_outputs = len(self.model.output)
        outputs = [builder.get_unique_output_name(NodeResultName(self, i))
                   for i in range(n_outputs)]

        # linking inputs
        logger.indent()
        if self.sub_functions is not None:
            for sub in self.sub_functions:
                builder.add_function(sub)
        builder.add_function(self.model)
        builder.add_node(
            self.model.name, builder.get_unique_name(
                '_fct_' + self.model.name, reserved=False),
            inputs, outputs, domain=self.model.domain)
        logger.dedent()


class _GraphBuilder:
    """
    Graph builder. It takes a graph structure made with
    instances of @see cl OnnxOperatorBase.
    The main method is `to_onnx`.

    * `initializer`: list of initializers to add to the ONNX graph
    * `node`: list of nodes to add to the ONNX graph
    * `input`: list of inputs to add to the ONNX graph
    * `output`: list of inputs to add to the ONNX graph
    * `opsets`: opsets of the ONNX graph
    * `input_names`: dictionary of input names
        `{name: InputDetectedVariable}`
    * `node_output_names`: memorizes a name for a node output
        when the user did not specify any
        `{(id(node), index): OutputDetectedVariable}`
    * `reserved_names`: dictionary `{ name : (node, index) }`,
        name which should remain unchanged in the ONNX graph
    * `names`: list of uniques names
    * `functions`: dictionary `{ domain, name: function_proto }`
    * `function_hashes`: dictionary `{ domain, name: hash of function_proto }`
    """

    def __init__(self):
        self.initializer = []
        self.node = []
        self.input = []
        self.output = []
        self.opsets = {}
        self.input_names = {}
        self.node_output_names = {}
        self.reserved_names = {}
        self.names = set()
        self.functions = {}
        self.function_hashes = {}
        logger.debug('_GraphBuilder-%d:new', id(self))

    def _add_domain(self, domain, version):
        if domain not in self.opsets:
            self.opsets[domain] = version
        else:
            self.opsets[domain] = max(version, self.opsets[domain])

    def _add_name(self, name):
        self.names.add(name)

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

    def reserve_names(self, node, output_names):
        """
        Adds names to the list of reserved names.
        All must be unique.

        :param node: node or None for an input
        :param output_names: names of the output
        """
        if output_names is None:
            return
        for index, var in enumerate(output_names):
            if not isinstance(var, (Variable, ExistingVariable)):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type {type(var)!r} for {var!r}.")
            self.reserve_name(node, var.name, index)

    def reserve_name(self, node, name, index):
        """
        Reserves a name so that it cannot be changed.

        :param node: node or None for an input
        :param name: name
        :param index: input index
        """
        if not isinstance(name, str):
            raise TypeError(  # pragma: no cover
                f"Name {name!r} is not a string.")
        if name in self.reserved_names:
            raise RuntimeError(  # pragma: no cover
                "Name %r is already reserved from node %r, index=%d." % (
                    name, node, index))
        logger.debug("_GraphBuilder-%d.reserve_name([%s-%d], %r, %r)",
                     id(self), node.__class__.__name__, id(node),
                     name, index)
        self.reserved_names[name] = (node, index)
        self._add_name(name)

    def get_unique_output_name(self, result):
        """
        Returns a unique output_name for a NodeResultName.

        :param result: instance of @see cl NodeResultName
        """
        if not isinstance(result, NodeResultName):
            raise TypeError(  # pragma: no cover
                "Result must be of type NodeResultName not %r (%r)." % (
                    type(result), result))
        if result.node is None:
            key = None, result.index
        else:
            key = id(result.node), result.index
        if key in self.node_output_names:
            return self.node_output_names[key]
        name = result.get_name()
        if name in self.reserved_names:
            unique = name
        else:
            unique = self.get_unique_name(name)
        self.node_output_names[key] = unique
        return unique

    def get_unique_name(self, name, reserved=True):
        """
        Returns a unique name to name an output.

        :param name: name
        :param reserved: bypass if the name is a reserved one
        :return: unique name, may be the same if not taken already
        """
        if not isinstance(name, str):
            raise TypeError(  # pragma: no cover
                f"name must be a string not {type(name)!r}.")
        if reserved and name in self.reserved_names:
            logger.debug(  # pragma: no cover
                "_GraphBuilder-%d.get_unique_name(%r) 1-> %r",
                id(self), name, name)
            return name
        if name not in self.names:
            self._add_name(name)
            logger.debug("_GraphBuilder-%d.get_unique_name(%r) 2-> %r",
                         id(self), name, name)
            return name
        i = 1
        new_name = f"{name}_{self.number2alpha(i)}"
        while new_name in self.names:
            i += 1
            new_name = f"{name}_{self.number2alpha(i)}"
        self._add_name(new_name)
        logger.debug("_GraphBuilder-%d.get_unique_name(%r) 3-> %r",
                     id(self), name, new_name)
        return new_name

    def get_input_names(self, node, inputs):
        """
        Returns input names for node *node* and inputs *inputs*.

        :param node: node
        :param inputs: inputs
        :return: name
        """
        logger.debug(
            "_GraphBuilder-%d.get_input_names:1:%s-%d:%r",
            id(self), node.__class__.__name__, id(node), inputs)
        names = []
        for i in inputs:
            if isinstance(i, (Variable, ExistingVariable)):
                self._add_name(i.name)
                names.append(i.name)
                if i.name in self.input_names:
                    if isinstance(i, Variable):
                        self.input_names[i.name] = InputDetectedVariable(
                            None, i)
                        logger.debug(
                            "_GraphBuilder-%d.get_input_names:2:a:%d:+input_names:%s",
                            id(self), id(node), i.name)
                    else:
                        logger.debug(  # pragma: no cover
                            "_GraphBuilder-%d.get_input_names:2:a:%d:=input_names:%s",
                            id(self), id(node), i.name)
                else:
                    self.input_names[i.name] = InputDetectedVariable(None, i)
                    logger.debug(
                        "_GraphBuilder-%d.get_input_names:2:b:%d:+input_names:%s",
                        id(self), id(node), i.name)
            elif isinstance(i, InputDetectedVariable):
                self._add_name(i.name)
                names.append(i.name)
                if i.name in self.input_names:
                    logger.debug(  # pragma: no cover
                        "_GraphBuilder-%d.get_input_names:2:c:%d:=input_names:%s",
                        id(self), id(node), i.name)
                else:
                    self.input_names[i.name] = i
                    logger.debug(
                        "_GraphBuilder-%d.get_input_names:2:c:%d:+input_names:%s",
                        id(self), id(node), i.name)
            elif isinstance(i, OnnxExisting):
                inp = i.inputs[0]
                n = inp.output_names[0]
                self._add_name(n.name)
                names.append(n.name)
                if n.name in self.input_names:
                    if isinstance(inp, Variable):
                        self.input_names[n.name] = InputDetectedVariable(
                            None, n)
                        logger.debug(  # pragma: no cover
                            "_GraphBuilder-%d.get_input_names:2:d:%d:+input_names:%s",
                            id(self), id(node), n.name)
                    else:
                        logger.debug(
                            "_GraphBuilder-%d.get_input_names:2:d:%d:=input_names:%s",
                            id(self), id(node), n.name)
                else:
                    self.input_names[n.name] = InputDetectedVariable(None, n)
                    logger.debug(
                        "_GraphBuilder-%d.get_input_names:2:d:%d:+input_names:%s",
                        id(self), id(node), n.name)
            elif isinstance(i, OnnxOperator):
                key = id(i), 0
                try:
                    name = self.node_output_names[key]
                except KeyError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to find key %r for input "
                        "(type(i) is %r, type(node) is %r) "
                        "%r in node %r among %r." % (
                            key, type(i), type(node), i, node,
                            list(self.node_output_names))) from e
                names.append(name)
            elif isinstance(i, OnnxOperatorItem):
                if isinstance(i.onx_op, OnnxOperatorTuple):
                    if i.onx_op.values is None:
                        key = id(i.onx_op.unique), i.index
                    else:
                        key = id(i.onx_op[i.index]), 0
                elif isinstance(i.onx_op, OnnxOperator):
                    key = id(i.onx_op), i.index
                else:
                    raise TypeError(  # pragma: no cover
                        f"Unexpected type for OnnxOperatorItem: {type(i.onx_op)!r}.")
                try:
                    name = self.node_output_names[key]
                except KeyError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to find key %r for input %r in node %r." % (
                            key, i, node)) from e
                names.append(name)
            elif isinstance(i, OnnxOperatorTuple):
                raise NotImplementedError()  # pragma: no cover
            elif isinstance(i, numpy.ndarray):
                # Adding an initializer
                name = self.get_unique_name('init', reserved=False)
                init = from_array(i, name)
                self.initializer.append(init)
                names.append(name)
            else:
                raise TypeError(  # pragma: no cover
                    f"Unexpected type for an input {type(i)!r}.")
        logger.debug(
            "_GraphBuilder-%d.get_input_names:3:%r", id(self), names)
        return names

    def add_initializer(self, name, init):
        """
        Adds an initializer to the graph.

        :param name: initializer name
        :param init: initializer to copy
        :return: created intializer
        """
        if isinstance(init, onnx.TensorProto):
            tensor = to_array(init)
            val = from_array(tensor, name)
            logger.debug("_GraphBuilder.add_initializer:1(%r, %r, %r)",
                         name, tensor.dtype, tensor.shape)
        elif isinstance(init, numpy.ndarray):
            value = to_array(init)
            val = from_array(value, name)
            logger.debug("_GraphBuilder.add_initializer:2(%r, %r, %r)",
                         name, init.dtype, init.shape)
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Unsupported initializer type {type(init)!r}.")
        self.initializer.append(val)
        return val

    def add_function(self, function_proto,
                     raise_if_exist=False, check_unique=True,
                     opset=1):
        """
        Adds a function to the graph.

        :param function_proto: instance of type :epkg:`FunctionProto`
        :param raise_if_exist: raises an exception if a function of the
            same name was already added
        :param check_unique: checks if a function was added twice,
            it is the same
        :param opset: opset for the domain the function belongs to
        """
        def _hash(p):
            m = hashlib.sha256()
            m.update(p.SerializeToString())
            return m.hexdigest()[:64]

        key = function_proto.domain, function_proto.name
        if key in self.functions:
            if raise_if_exist:
                raise RuntimeError(  # pragma: no cover
                    f"Function {key!r} is added for the second time.")
            if check_unique:
                hs = _hash(function_proto)
                if hs != self.function_hashes[key]:
                    raise RuntimeError(  # pragma: no cover
                        "Function %r is added for the second time "
                        "and the content is not the same." % (key, ))
                return
        self.functions[key] = function_proto
        self.function_hashes[key] = _hash(function_proto)
        self._add_domain(function_proto.domain, opset)

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
        if domain is None:
            domain = ''
        logger.debug("_GraphBuilder-%d.add_node(%r, %r, "
                     "inputs=%r, outputs=%r, domain=%r, opset=%r)",
                     id(self), op_type, name, inputs, outputs, domain, opset)
        if not isinstance(inputs, (list, OnnxOperator._InputContainer)):
            raise TypeError(  # pragma: no cover
                f"inputs must be a list not {type(inputs)!r}.")
        if not isinstance(outputs, (list, OnnxOperator._InputContainer)):
            raise TypeError(  # pragma: no cover
                f"inputs must be a list not {type(outputs)!r}.")
        if any(map(lambda x: not isinstance(x, str), inputs)):
            raise TypeError(  # pragma: no cover
                f"inputs must be all strings not {inputs!r}.")
        if any(map(lambda x: not isinstance(x, str), outputs)):
            raise TypeError(  # pragma: no cover
                f"outputs must be all strings not {outputs!r}.")
        if opset is not None:
            self._add_domain(domain, opset)
        node = make_node(op_type, inputs, outputs, name=name,
                         domain=domain, **attributes)
        self.node.append(node)
        return node

    def _process_io(self, inputs, input_names_):
        logger.debug("_GraphBuilder-%d._process_io:1:inputs=%r",
                     id(self), inputs)
        logger.debug("_GraphBuilder-%d._process_io:1:input_names_=%r",
                     id(self), input_names_)
        if input_names_ is None:
            input_names = None
        else:
            input_names = []
            for inp in input_names_:
                if inp.var.name == '':
                    continue
                input_names.append(inp)

        if inputs is None:
            logger.debug(  # pragma: no cover
                "_GraphBuilder-%d._process_io:return:%r",
                id(self), self.input_names)
            return [
                make_tensor_value_info(
                    'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
                for name in self.input_names], None

        if not isinstance(inputs, (list, OnnxOperator._InputContainer)):
            if is_numpy_dtype(inputs):
                inputs = [inputs]

        logger.debug("_GraphBuilder-%d._process_io:2:input_names=%r",
                     id(self), input_names)
        if input_names is None:
            # outputs
            set_names = set()
            input_names = []
            new_inputs = []
            for inp in inputs:
                if isinstance(inp, OutputDetectedVariable):
                    if inp.name in set_names:
                        raise ValueError(  # pragma: no cover
                            f"Names already taken {inp.name!r} in {inputs!r}.")
                    set_names.add(inp.name)
                    if isinstance(inp.node, OnnxExisting):
                        raise NotImplementedError(  # pragma: no cover
                            f"Unexpected name {inp.name!r} type {type(inp.node)!r}.")
                        # continue
                    key = id(inp.node), inp.index
                    if key in self.node_output_names:
                        new_name = self.node_output_names[key]
                        new_var = OutputDetectedVariable(
                            inp.node, inp.var.copy_name(new_name), inp.index)
                        input_names.append(new_var)
                        new_inputs.append(new_var)
                    else:
                        raise RuntimeError(  # pragma: no cover
                            "Key %r is ambiguous or defined in "
                            "two nodes %r, id(node)=%d, index=%d." % (
                                key, inp, id(inp.node), inp.index))
                else:
                    raise TypeError(  # pragma: no cover
                        "Unexpected type %r (it should be "
                        "OutputDetectedVariable) in %r." % (inp, inputs))
            inputs = new_inputs
            if len(input_names) == 0:
                raise RuntimeError(  # pragma: no cover
                    "Unable to cross %r and %r or %r (set_names=%r)." % (
                        inputs, self.output_names_rev,
                        self.node_output_names_rev, set_names))
        elif not isinstance(input_names, (list, OnnxOperator._InputContainer)):
            raise RuntimeError(  # pragma: no cover
                f"Unexpected type for input_names {type(input_names)!r}.")
        else:
            # inputs
            pass

        # common parts
        logger.debug("_GraphBuilder-%d._process_io:3:input_names:%r",
                     id(self), input_names)
        logger.debug("_GraphBuilder-%d._process_io:3:inputs:%r",
                     id(self), inputs)
        no_exists_names = [c for c in input_names if not isinstance(
            c.var, (ExistingVariable, OnnxExisting))]
        no_exists = [c for c in inputs if not isinstance(
            c.var, (ExistingVariable, OnnxExisting))]

        if isinstance(input_names, (list, OnnxOperator._InputContainer)):
            d_input_names = {}
            for inp in input_names:
                if inp.name in d_input_names:
                    raise ValueError(  # pragma: no cover
                        f"Duplicated name {inp.name!r} in {input_names!r}.")
                d_input_names[inp.name] = inp
        elif isinstance(input_names, dict):
            d_input_names = input_names
        else:
            raise TypeError(  # pragma: no cover
                "Unexpected type for input_names %r (%r)." % (
                    type(input_names), input_names))

        logger.debug("_GraphBuilder-%d._process_io:4:no_exists_names:%r",
                     id(self), no_exists_names)
        logger.debug("_GraphBuilder-%d._process_io:4:no_exists:%r",
                     id(self), no_exists)

        # mapping
        res = []
        for inp in no_exists:
            if not isinstance(inp, DetectedVariable):
                raise TypeError(  # pragma: no cover
                    f"inp not DetectedVariable but {type(inp)!r} ({inp!r}).")
            if inp.name.startswith('???'):
                raise RuntimeError(  # pragma: no cover
                    f"Issue with variable {inp!r}.")
            var = d_input_names[inp.name]
            if not isinstance(var, DetectedVariable):
                raise TypeError(  # pragma: no cover
                    f"var not Variable but {type(var)!r} ({var!r}).")

            # inp: Variable
            # var: str
            if isinstance(var.var, ExistingVariable):
                # It may be an input referenced in a subgraph and not used in the
                # main graph.
                if inp.var.name != var.var.name:
                    raise RuntimeError(  # pragma: no cover
                        f"Unexpected {inp!r} != {var!r}.")
            elif inp.var != var.var:
                raise RuntimeError(  # pragma: no cover
                    f"Unexpected {inp!r} != {var!r}.")

            if isinstance(inp.var, ExistingVariable):
                # The type of ExistingVariable must be known
                # to build the subgraph. Let's try unknown.
                res.append(make_tensor_value_info(inp.name, 0, None))
            else:
                res.append(make_tensor_value_info(
                    inp.name, inp.var.proto_added_type,
                    inp.var.proto_added_shape))

        hidden = [c for c in input_names if isinstance(
            c.var, (ExistingVariable, OnnxExisting))]
        logger.debug("_GraphBuilder-%d._process_io:4:return:res:%r",
                     id(self), [n.name for n in res])
        logger.debug("_GraphBuilder-%d._process_io:4:return:hidden:%r",
                     id(self), hidden)
        return res, hidden

    def to_onnx(self, inputs=None, outputs=None,
                target_opset=None, run_shape=False,
                optim=True, function_name=None,
                function_domain=None, verbose=0,
                check_model=True):
        """
        Converts this operator into an ONNX graph.

        :param inputs: specific inputs (as a dictionary) or
            default inputs if not specified
        :param outputs: specific outputs
        :param target_opset: dictionary with target opset per domain,
            None for the default one
        :param run_shape: run shape inference before returning the model
        :param optim: optimize the model with function
            @see fn onnx_optimisations
        :param function_name: if not None builds a :epkg:`FunctionProto`
            use this name
        :param function_domain: in case of a function, declares the function
            as part of this domain, `'mlprodict'` if None
        :param verbose: prints information
        :param check_model: checks the output model
        :return: onnx graph

        (_GraphBuilder)
        """
        logger.debug("_GraphBuilder-%d.to_onnx:#####:%s",
                     id(self), str(function_name))
        logger.debug("_GraphBuilder-%d.to_onnx(%r, %r, target_opset=%r)",
                     id(self), inputs, outputs, target_opset)
        # inputs and outputs
        if not all(map(lambda x: isinstance(x, InputDetectedVariable), inputs)):
            raise TypeError(  # pragma: no cover
                "One of the input is not InputDetectedVariable.")
        if not all(map(lambda x: isinstance(x, OutputDetectedVariable), outputs)):
            raise TypeError(  # pragma: no cover
                "One of the outputs is not OutputDetectedVariable.")
        logger.indent()
        self.input, self.hidden_input = self._process_io(
            inputs, list(self.input_names.values()))
        logger.dedent()
        logger.debug("_GraphBuilder-%d.to_onnx:hidden_input:%r",
                     id(self), self.hidden_input)
        logger.indent()
        self.output, self.hidden_output = self._process_io(outputs, None)
        logger.dedent()
        if len(self.hidden_output) > 0:
            raise RuntimeError(  # pragma: no cover
                f"Unexpected hidden output {self.hidden_output!r}.")
        logger.debug("_GraphBuilder-%d.to_onnx:self.input=%r",
                     id(self), [i.name for i in self.input])
        if len(self.hidden_input) > 0:
            logger.debug("_GraphBuilder-%d.to_onnx:self.hidden_input=%r",
                         id(self), [i.name for i in self.hidden_input])
        logger.debug("_GraphBuilder-%d.to_onnx:self.output=%r",
                     id(self), [i.name for i in self.output])
        logger.debug("_GraphBuilder-%d.to_onnx:build:n_inputs=%r n_inits=%r n_nodes=%r "
                     "n_outputs=%r",
                     id(self), len(self.input), len(self.initializer),
                     len(self.node), len(self.output))

        if function_name is not None:
            if function_domain is None:
                function_domain = 'mlprodict'
            if len(self.initializer) > 0:
                nodes = []
                for init in self.initializer:
                    nodes.append(
                        make_node('Constant', [], [init.name], value=init,
                                  name=f'_init_{init.name}'))
                nodes.extend(self.node)
            else:
                nodes = self.node
            fct = make_function(
                function_domain, function_name,
                [_.name for _ in self.input],
                [_.name for _ in self.output],
                nodes,
                [make_opsetid(k, v) for k, v in self.opsets.items()])
            if check_model:
                check_onnx(fct)
            if optim:
                from ..onnx_tools.optim import onnx_optimisations
                fct = onnx_optimisations(fct)
                if check_model:
                    check_onnx(fct)
            logger.debug("_GraphBuilder-%d:fct:.to_onnx() -> done", id(self))
            logger.debug("_GraphBuilder-%d:fct:to_onnx:#####", id(self))
            return fct
        else:
            graph = make_graph(
                self.node, 'XOP', self.input, self.output, self.initializer)
            onnx_model = make_model(
                graph, functions=list(self.functions.values()))
            opv = self.opsets.get('', max_supported_opset())
            opset2ir = _default_OPSET_TO_IR_VERSION()
            irv = opset2ir.get(opv, max(opset2ir.values()))
            onnx_model.ir_version = irv

            logger.debug("_GraphBuilder-%d.to_onnx:2onnx:n_inputs=%r n_inits=%r "
                         "n_nodes=%r n_outputs=%r",
                         id(self), len(onnx_model.graph.input),
                         len(onnx_model.graph.initializer),
                         len(onnx_model.graph.node),
                         len(onnx_model.graph.output))

            del onnx_model.opset_import[:]  # pylint: disable=E1101
            seen_opset = set()
            for k, v in self.opsets.items():
                if (k or '') in seen_opset:
                    raise RuntimeError(  # pragma: no cover
                        f"Duplicated opset ({k!r}, {v!r}).")
                op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
                op_set.domain = k or ''
                op_set.version = v
                seen_opset.add(op_set.domain)

            # optimisation, remove redundant constant, unnecessary
            # identity nodes.
            if check_model:
                check_onnx(onnx_model)
            if optim:
                from ..onnx_tools.optim import onnx_optimisations
                onnx_model = onnx_optimisations(onnx_model)
                if check_model:
                    logger.debug(
                        "_GraphBuilder-%d.to_onnx:check_onnx", id(self))
                    check_onnx(onnx_model)

            logger.debug("_GraphBuilder-%d.to_onnx:optim:n_inputs=%r n_inits=%r "
                         "n_nodes=%r n_outputs=%r",
                         id(self), len(onnx_model.graph.input),
                         len(onnx_model.graph.initializer),
                         len(onnx_model.graph.node),
                         len(onnx_model.graph.output))

            if run_shape:
                logger.debug("_GraphBuilder-%d.to_onnx:infer_shapes", id(self))
                with_shape = infer_shapes(onnx_model)
                logger.debug("_GraphBuilder-%d.to_onnx:shape:n_inputs=%r "
                             "n_inits=%r n_nodes=%r n_outputs=%r",
                             id(self), len(with_shape.graph.input),
                             len(with_shape.graph.initializer),
                             len(with_shape.graph.node),
                             len(with_shape.graph.output))
                return with_shape

            logger.debug("_GraphBuilder-%d.to_onnx:mod -> done", id(self))
            logger.debug("_GraphBuilder-%d.to_onnx:mod:#####", id(self))
            return onnx_model


class _StaticVariables:
    """
    Holds static variables.
    """

    def __init__(self):
        self._all_schemas_ = None
        self._all_schemas_versions_ = None
        self._all_domains_ = None
        self._all_classes_ = None

    @property
    def all_schemas(self):
        "Returns all schemas."
        self.populate()
        return self._all_schemas_

    @property
    def all_classes(self):
        "Returns all operators wrapped in classes."
        self.populate()
        return self._all_classes_

    @property
    def all_schemas_versions(self):
        "Returns all operators, domains, versions."
        self.populate()
        return self._all_schemas_versions_

    @property
    def all_domains(self):
        "Returns all domains."
        self.populate()
        return self._all_domains_

    def populate(self):
        "Populates static variables."
        if self._all_schemas_ is not None:
            return
        (self._all_schemas_, self._all_schemas_versions_,
         self._all_domains_) = _populate_schemas()
        self._all_classes_ = {}


_S = _StaticVariables()
onnx_load_factory = Xop = OnnxLoadFactory()


class OnnxExisting(OnnxOperator):
    """
    Wrapper around OnnxIdentity to specify this operator is
    not part of the subgraph it is used in.
    """

    _unique_names = set()

    expected_inputs = ['X']
    expected_outputs = ['Y']
    operator_name = 'Existing'
    input_range = [1, 1]
    output_range = [1, 1]
    domain = ''
    is_deprecated = False
    since_version = 1
    past_version = []
    attr_names = []
    op_type = 'Existing'
    __module__ = __name__

    @staticmethod
    def get_unique_name(var):
        """
        Returns a unique variable name.

        :param var: an instance of OnnxOperator.
        :return: unique variable name
        """
        if isinstance(var, OnnxOperator):
            name = "%s_%s" % ((var.domain or "").lower().replace(".", ""),
                              var.op_type.lower())
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected type {type(var)!r} for var.")
        i = 0
        new_name = "_exist_%s_%d" % (name, i)
        while new_name in OnnxExisting._unique_names:
            i += 1
            new_name = "_exist_%s_%d" % (name, i)
        OnnxExisting._unique_names.add(new_name)
        return new_name

    def __init__(self, *args, **kwargs):  # pylint: disable=W0231
        # OnnxIdentity.__init__(self, *args, **kwargs)  # pylint: disable=W0233
        OnnxOperator.__init__(self, *args, **kwargs)  # pylint: disable=W0233
        self.control_ops_ = None
        if len(self.inputs) != 1:
            raise RuntimeError(  # pragma: no cover
                f"Unexpected number of inputs {len(self.inputs)}.")
        if isinstance(self.inputs[0], Variable):
            # It is one input
            new_names = [
                ExistingVariable(self.inputs[0].name, self.inputs[0])]
            logger.debug("op:OnnxExisting-%d.__init__:set-input:1:%r",
                         id(self), new_names)
            self.inputs[0].output_names = new_names
        else:
            if not isinstance(self.inputs[0], OnnxOperatorBase):
                raise TypeError(  # pragma: no cover
                    f"Only input should a node not {type(self.inputs[0])!r}.")
            if self.inputs[0].output_names is None:
                new_names = [
                    ExistingVariable(OnnxExisting.get_unique_name(self.inputs[0]),
                                     self.inputs[0])]
                logger.debug("op:OnnxExisting-%d.__init__:set-input:2:%r",
                             id(self), new_names)
                self.inputs[0].output_names = new_names

    def __repr__(self):
        """
        usual
        """
        return "{}({}) -> {}".format(
            self.__class__.__name__,
            self.inputs[0].output_names,
            [str(o) for o in self.output_names]
            if self.output_names is not None else "?")

    def find_named_inputs(self):
        """
        Retrieves all named inputs in this graph.
        """
        res = []
        for i, inp in enumerate(self.inputs[0].output_names):
            if not isinstance(inp, (Variable, ExistingVariable)):
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for input %r in node type %r."
                    "" % (type(inp), i, type(self)))
            res.append(inp.name)
        return res

    def f(self, *inputs, verbose=0, fLOG=None,  # pylint: disable=W0221
          clear_cache=False, runtime=None):
        "For the eager mode."
        raise NotImplementedError()  # pragma: no cover

    def _set_control_op(self, op):
        if op is None:
            raise RuntimeError(  # pragma: no cover
                "op cannot be None in _set_control_op.")
        logger.debug("op:%s-%d:_set_control_op:found:p:%d:%r",
                     self.__class__.__name__, id(self), id(op),
                     self.inputs[0].output_names)
        if self.control_ops_ is None:
            self.control_ops_ = []
        self.control_ops_.append(op)
        op.add_external_input(self.inputs[0])
