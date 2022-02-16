"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import os
import numpy
from scipy.sparse.coo import coo_matrix
import onnx
from ._cache import cache_folder
from .xop_variable import Variable
from .xop_auto import get_rst_doc


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
    from .xop import OnnxOperator, OnnxOperatorItem

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


def _populate_schemas():
    """
    Populates all schemas.
    """
    res = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.support_level == schema.SupportType.EXPERIMENTAL:
            # Skips experimental operators.
            continue
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + '_' + str(schema.since_version)] = schema
    return res


def _dynamic_class_creation(operator_names=None, cache=False, verbose=0, fLOG=print):
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
        operator_names = list(_all_schemas)

    res = _all_schemas
    cls = {}
    set_names = dict()
    set_skip = set()
    for pos, op_name in enumerate(operator_names):
        set_names[op_name] = pos
        if '_' in op_name:
            n = op_name.split('_')[0]
            if n.startswith('Onnx'):
                set_skip.add(n)
            else:
                set_skip.add('Onnx' + n)
            if n not in set_names:
                set_names[n] = -1

    if verbose > 1 and fLOG is not None:
        fLOG("[_dynamic_class_creation] set_names=%r" % set_names)
        fLOG("[_dynamic_class_creation] set_skip=%r" % set_skip)

    returned_classes = []
    positions = {}

    for op_name, position in set_names.items():
        cl_name = op_name if op_name.startswith('Onnx') else 'Onnx' + op_name
        if verbose > 3 and fLOG is not None:
            fLOG('[_dynamic_class_creation] cl_name=%r op_name=%r (in=%d)' % (
                cl_name, op_name, 1 if cl_name in _all_classes else 0))
        if cl_name in _all_classes:
            if cl_name not in set_skip:
                if position >= 0:
                    returned_classes.append((position, _all_classes[cl_name]))
            continue
        if verbose > 0 and fLOG is not None:
            fLOG("[_dynamic_class_creation] op_name=%r, cl_name=%r" % (
                op_name, cl_name))

        name = op_name[4:] if op_name.startswith('Onnx') else op_name
        try:
            schema = res[name]
        except KeyError as e:
            raise ValueError(
                "Operator %r (or %r) does not exists." % (
                    name, op_name)) from e
        inputs = [_c(o, 'I', i) for i, o in enumerate(schema.inputs)]
        outputs = [_c(o, 'O', i) for i, o in enumerate(schema.outputs)]
        args = [p for p in schema.attributes]

        if '_' in op_name:
            class_name = "Onnx" + name
        else:
            class_name = "Onnx" + schema.name

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


_all_schemas = _populate_schemas()
_all_classes = {}


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
