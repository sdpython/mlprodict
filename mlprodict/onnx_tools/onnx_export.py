"""
@file
@brief Exports an ONNX graph in a way it can we created again
with a python script. It relies on :epkg:`jinja2` and :epkg:`autopep8`.

.. versionadded:: 0.7
"""
import numpy
from jinja2 import Template
import autopep8
import onnx
from onnx import numpy_helper
from .onnx2py_helper import (
    _var_as_dict, guess_proto_dtype, guess_proto_dtype_name)
from .onnx_export_templates import (
    _onnx_templates, _tf2onnx_templates, _numpy_templates)


def make_tf2onnx_code(opset, name=None, op_type=None, domain='',
                      inputs=None, outputs=None, attributes=None,
                      used=None, context=None, mark_inits=None, indent=8,
                      **unused):
    """
    Converts an ONNX operators into :epkg:`tf2onnx` code.

    :param opset: target opset for the conversion (usually unused)
    :param name: node name
    :param op_type: operator type
    :param domain: domain
    :param inputs: inputs
    :param outputs: outputs
    :param attributes: attributes
    :param used: dictionary `{k: v}`,
        list of nodes taking *k* as input
    :param context: whole context
    :param mark_inits: marks initializer as replaced
    :param indent: number of spaces to add on the second
        and following rows
    :return: code as str
    """
    def simplify(name, kind, force=True):
        value = None
        if (used is not None and name in used and
                len(used[name]) == 1 and context is not None):
            inits = context['initializers_dict']
            if name in inits:
                v = inits[name]
                if v.dtype == numpy.int64 and v.size < 10:
                    value = v
                    if name not in mark_inits:
                        mark_inits[name] = []
                    mark_inits[name].append(v)

        if value is None and force:
            inits = context['initializers_dict']
            value = inits[name]
        if kind == 'list':
            if value is None:
                return name
            if len(value.shape) == 0:
                return str(value)
            return str(list(value))
        raise NotImplementedError(
            "Unknown scenario to simplify (%r)." % kind)

    rows = []
    if op_type == 'Unsqueeze':
        if len(inputs) == 2:
            rows.append(
                "node = GraphBuilder(ctx).make_unsqueeze("
                "{'data': varx[%r], 'axes': %s}, return_node=True)"
                "" % (inputs[0], simplify(inputs[1], 'list')))
        else:
            raise NotImplementedError(  # pragma: no cover
                "Unable to create code for operator %r (opset <= 12)"
                "." % op_type)
    else:
        if len(attributes) > 0:
            attributes_str = ", ".join("%s=%s" % (k, v) for k, v in attributes)
            attr = ", attr=dict(%s)" % attributes_str
        else:
            attr = ""
        rows.append(
            "inputs = [%s]" % ", ".join("varx[%r]" % n for n in inputs))
        sdomain = '' if domain == '' else ("domain=%r, " % domain)
        rows.append(
            "node = ctx.make_node(%r, inputs=inputs%s, %s"
            "name=make_name(%r))" % (
                op_type, attr, sdomain, name))
    for i, n in enumerate(outputs):
        rows.append("varx[%r] = node.output[%d]" % (n, i))
    if indent > 0:
        sind = " " * indent
        for i in range(1, len(rows)):
            rows[i] = sind + rows[i]
    return "\n".join(rows)


def make_numpy_code(opset, name=None, op_type=None, domain='',
                    inputs=None, outputs=None, attributes=None,
                    used=None, context=None, mark_inits=None,
                    **unused):
    """
    Converts an ONNX operators into :epkg:`numpy` code.

    :param opset: target opset for the conversion (usually unused)
    :param name: node name
    :param op_type: operator type
    :param domain: domain
    :param inputs: inputs
    :param outputs: outputs
    :param attributes: attributes
    :param used: dictionary `{k: v}`,
        list of nodes taking *k* as input
    :param context: whole context
    :param mark_inits: marks initializer as replaced
    :return: code as str
    """
    def make_sure_inputs(n, m=None):
        if m is None:
            m = n
        if len(inputs) < n:
            raise RuntimeError(  # pragma: no cover
                "Expecting at least %d inputs for operator %r not %r." % (
                    n, op_type, inputs))
        if len(inputs) > m:
            raise RuntimeError(  # pragma: no cover
                "Expecting at most %d inputs for operator %r not %r." % (
                    m, op_type, inputs))

    def make_sure_opsets(mi, ma=None):
        if mi is not None and opset < mi:
            raise RuntimeError(  # pragma: no cover
                "Cannot convert operator type %d, opset %d < %d." % (
                    op_type, opset, mi))
        if ma is not None and opset > ma:
            raise RuntimeError(  # pragma: no cover
                "Cannot convert operator type %d, opset %d > %d." % (
                    op_type, opset, mi))

    def getat(name, defval=None):
        for n, val in attributes:
            if name == n:
                return val
        return defval

    def simplify(name, kind):
        value = None
        if (used is not None and name in used and
                len(used[name]) == 1 and context is not None):
            inits = context['initializers_dict']
            if name in inits:
                v = inits[name]
                if v.dtype == numpy.int64 and v.size < 10:
                    value = v
                    if name not in mark_inits:
                        mark_inits[name] = []
                    mark_inits[name].append(v)

        if kind == 'tuple':
            if value is None:
                return "tuple(%s)" % name
            if value.size == 1:
                return str(tuple(value)[0])
            return str(tuple(value))
        elif kind == 'list':
            if value is None:
                return name
            if len(value.shape) == 0:
                return str(value)
            return str(list(value))
        raise NotImplementedError(
            "Unknown scenario to simplify (%r)." % kind)

    def make_tuple(val):
        if isinstance(val, tuple):
            return val
        if isinstance(val, list):
            return tuple(val)
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            return tuple(map(int, val.strip('()[]').replace(" ", "").split(",")))
        raise NotImplementedError(
            "Unable to convert %r into tuple." % val)

    if domain != '':
        raise NotImplementedError(
            "Unable to convert any operator from domain %r." % domain)

    binary_ops = dict(Add='+', Sub='-', Div='/', Mul='*', MatMul='@',
                      Pow='**')
    unary_ops = dict(Neg='-')
    unary_ops_ = dict(Sqrt='** 0.5')

    outs = ", ".join(outputs)

    if op_type in binary_ops:
        make_sure_inputs(2)
        return "%s = %s %s %s" % (outs, inputs[0], binary_ops[op_type], inputs[1])

    if op_type in unary_ops:
        make_sure_inputs(1)
        return "%s = %s %s" % (outs, unary_ops[op_type], inputs[0])

    if op_type in unary_ops_:
        make_sure_inputs(1)
        return "%s = %s %s" % (outs, inputs[0], unary_ops_[op_type])

    if op_type == 'ArgMin':
        make_sure_opsets(12)
        make_sure_inputs(1)
        axis = getat('axis', 0)
        keepdims = getat('keepdims', 1)
        select_last_index = getat('keepdims', 0)
        return (
            "%s = argmin_use_numpy_select_last_index("
            "%s, axis=%s, keepdims=%s, select_last_index=%s)" % (
                outs, inputs[0], axis, keepdims, select_last_index))

    if op_type == 'Concat':
        axis = getat('axis', 0)
        return "%s = numpy.concatenate([%s], %s)" % (outs, ", ".join(inputs), axis)

    if op_type == 'Max':
        return "%s = numpy.maximum(%s)" % (outs, ", ".join(inputs))

    if op_type == 'Gather':
        make_sure_opsets(11)
        make_sure_inputs(2)
        axis = getat('axis', 0)
        return "%s = numpy.take(%s, %s, axis=%s)" % (
            outs, inputs[0], simplify(inputs[1], 'list'), axis)

    if op_type == 'Gemm':
        make_sure_inputs(2, 3)
        alpha = getat('alpha', 0.)
        transA = getat('transA', 0)
        transB = getat('transB', 0)
        ta = ".T" if transA in ('1', 1, True) else ""
        tb = ".T" if transB in ('1', 1, True) else ""
        if len(inputs) == 2:
            return "%s = %s%s @ %s%s * %s" % (
                outs, inputs[0], ta, inputs[1], tb, alpha)
        beta = getat('beta', 0.)
        return "%s = %s%s @ %s%s * %s + %s * %s" % (
            outs, inputs[0], ta, inputs[1], tb, alpha, inputs[2], beta)

    if op_type == 'Identity':
        return "%s = %s" % (outs, inputs[0])

    if op_type == 'ReduceProd':
        make_sure_inputs(1)
        axes = getat('axes', "[0]")
        keepdims = getat('keepdims', 0)
        return "%s = %s.prod(axis=tuple(%s), keepdims=%s)" % (
            outs, inputs[0], axes, keepdims)

    if op_type == 'ReduceSum':
        make_sure_opsets(11)
        make_sure_inputs(2)
        keepdims = getat('keepdims', 0)
        return "%s = %s.sum(axis=%s, keepdims=%s)" % (
            outs, inputs[0], simplify(inputs[1], 'tuple'), keepdims)

    if op_type == 'ReduceSumSquare':
        make_sure_inputs(1)
        axes = getat('axes', "[0]")
        keepdims = getat('keepdims', 0)
        return "%s = (%s ** 2).sum(axis=tuple(%s), keepdims=%s)" % (
            outs, inputs[0], axes, keepdims)

    if op_type == 'Reshape':
        make_sure_inputs(2)
        return "%s = %s.reshape(%s)" % (
            outs, inputs[0], simplify(inputs[1], 'tuple'))

    if op_type == 'Shape':
        make_sure_inputs(1)
        return "%s = numpy.array(%s.shape, dtype=numpy.int64)" % (outs, inputs[0])

    if op_type == 'Slice':
        return "%s = make_slice(%s)" % (outs, ", ".join(inputs))

    if op_type == 'Squeeze':
        make_sure_opsets(13)
        make_sure_inputs(2)
        return "%s = numpy.squeeze(%s, axis=%s)" % (
            outs, inputs[0], simplify(inputs[1], 'tuple'))

    if op_type == 'Transpose':
        make_sure_inputs(1)
        perm = getat('perm', None)
        return "%s = numpy.transpose(%s, axes=%s)" % (
            outs, inputs[0], make_tuple(perm))

    if op_type == 'Unsqueeze':
        make_sure_opsets(13)
        make_sure_inputs(2)
        return "%s = numpy.expand_dims(%s, axis=%s)" % (
            outs, inputs[0], simplify(inputs[1], 'tuple'))

    raise NotImplementedError(
        "Unable to convert operator type %r name=%r." % (op_type, name))


def export_template(model_onnx, templates, opset=None, verbose=True, name=None,
                    rename=False, use_onnx_tensor=False,
                    autopep_options=None):
    """
    Exports an ONNX model to the onnx syntax.

    :param model_onnx: string or ONNX graph
    :param templates: exporting templates
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: insert prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :param use_onnx_tensor: when an attribute is an array
        and its name is `'value'`, it converts that array into an
        ONNX tensor to avoid type mismatch, (operator *ConstantOfShape*, ...)
    :param autopep_options: :epkg:`autopep8` options
    :return: python code
    """
    def number2name(n):
        n += 1
        seq = []
        while n >= 1:
            r = n % 26
            seq.append(r)
            n = (n - r) // 26
        return "".join(chr(65 + i) for i in reversed(seq))

    def rename_name(name):
        if len(name) == 0:
            raise ValueError(  # pragma: no cover
                "name is empty.")
        if name in dict_names:
            return dict_names[name]
        if rename:
            i = 0
            new_name = number2name(i)
            while new_name in dict_names:
                i += 1
                new_name = number2name(i)
            if len(new_name) == 0:
                raise ValueError(  # pragma: no cover
                    "Unable to rename name=%r i=%d." % (name, i))
            dict_names[name] = new_name
            dict_names[new_name] = new_name
            return new_name
        return name

    # containers
    context = {}
    used = {}

    # opset
    opsets = {}
    for oimp in model_onnx.opset_import:
        if oimp.domain == '' and opset is None:
            opsets[oimp.domain] = oimp.version
            opset = oimp.version
        else:
            opsets[oimp.domain] = opset
    context['opsets'] = opsets
    context['target_opset'] = opset

    dict_names = {}
    if rename:
        for o in model_onnx.graph.input:
            dict_names[o.name] = o.name
        for o in model_onnx.graph.output:
            dict_names[o.name] = o.name

    # inits
    initializers = []
    for init in model_onnx.graph.initializer:
        init_name = rename_name(init.name)
        value = numpy_helper.to_array(init)
        initializers.append((init_name, value))
    context['initializers'] = initializers
    context['initializers_dict'] = {k: v for k, v in initializers}

    # inputs
    inputs = []
    for inp in model_onnx.graph.input:
        t = inp.type.tensor_type
        dims = tuple(t.shape.dim)
        if len(dims) == 0:
            dims = None
        inputs.append((inp.name, t.elem_type, dims))
    context['inputs'] = inputs

    # outputs
    outputs = []
    for inp in model_onnx.graph.output:
        t = inp.type.tensor_type
        dims = tuple(t.shape.dim)
        if len(dims) == 0:
            dims = None
        outputs.append((inp.name, t.elem_type, dims))
    context['outputs'] = outputs

    # node
    nodes = []
    for node in model_onnx.graph.node:
        for i in node.input:
            if i not in used:
                used[i] = []
            used[i].append(node)
        attributes = []
        for at in node.attribute:
            temp = _var_as_dict(at)
            value = temp['value']
            if use_onnx_tensor:
                if node.op_type == 'Cast' and at.name == 'to':
                    attributes.append(
                        (at.name, guess_proto_dtype_name(int(value))))
                    continue
            if isinstance(value, str):
                attributes.append((at.name, "%r" % value))
            else:
                if isinstance(value, numpy.ndarray):
                    if use_onnx_tensor and at.name == 'value':
                        onnx_dtype = guess_proto_dtype_name(
                            guess_proto_dtype(value.dtype))
                        value = (
                            'make_tensor("value", %s, dims=%r, vals=%r)'
                            '' % (onnx_dtype, list(value.shape),
                                  value.tolist()))
                        attributes.append((at.name, value))
                    else:
                        attributes.append((at.name, repr(value.tolist())))
                else:
                    attributes.append((at.name, repr(value)))

        attributes_str = ", ".join("%s=%s" % (k, v) for k, v in attributes)
        d = dict(name=node.name, op_type=node.op_type,
                 domain=node.domain,
                 inputs=[rename_name(n) for n in node.input],
                 outputs=[rename_name(n) for n in node.output],
                 attributes=attributes, attributes_str=attributes_str)
        nodes.append(d)
    context['nodes'] = nodes

    # graph
    context['name'] = name or model_onnx.graph.name
    context['ir_version'] = model_onnx.ir_version
    context['producer_name'] = model_onnx.producer_name
    context['domain'] = model_onnx.domain
    context['model_version'] = model_onnx.model_version
    context['doc_string'] = model_onnx.doc_string
    context['metadata'] = {p.key: p.value for p in model_onnx.metadata_props}
    context['skip_inits'] = {}
    mark_inits = {}

    # final
    template = Template(templates)
    final = template.render(
        enumerate=enumerate, sorted=sorted, len=len,
        make_numpy_code=lambda *args, **kwargs: make_numpy_code(
            *args, context=context, used=used, mark_inits=mark_inits,
            **kwargs),
        make_tf2onnx_code=lambda *args, **kwargs: make_tf2onnx_code(
            *args, context=context, used=used, mark_inits=mark_inits,
            **kwargs),
        **context)

    skip_inits = set()
    for k, v in mark_inits.items():
        if len(v) == len(used[k]):
            # One initializers was removed.
            skip_inits.add(k)

    if len(skip_inits) > 0:
        context['skip_inits'] = skip_inits
        # Again with skip_inits.
        final = template.render(
            enumerate=enumerate, sorted=sorted, len=len,
            make_numpy_code=lambda *args, **kwargs: make_numpy_code(
                *args, context=context, used=used, mark_inits=mark_inits,
                **kwargs),
            make_tf2onnx_code=lambda *args, **kwargs: make_tf2onnx_code(
                *args, context=context, used=used, mark_inits=mark_inits,
                **kwargs),
            **context)

    final += "\n"
    if not verbose:
        rows = final.split("\n")
        final = "\n".join(_ for _ in rows if not _.endswith("#  verbose"))
    return autopep8.fix_code(final, options=autopep_options)


def export2onnx(model_onnx, opset=None, verbose=True, name=None, rename=False,
                autopep_options=None):
    """
    Exports an ONNX model to the :epkg:`onnx` syntax.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :param autopep_options: :epkg:`autopep8` options
    :return: python code

    The following example shows what a python code creating a graph
    implementing the KMeans would look like.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from mlprodict.onnx_tools.onnx_export import export2onnx

        X = numpy.arange(20).reshape(10, 2).astype(numpy.float32)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = to_onnx(tr, X, target_opset=14)
        code = export2onnx(onx)

        print(code)
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    code = export_template(model_onnx, templates=_onnx_templates,
                           opset=opset, verbose=verbose, name=name,
                           rename=rename, use_onnx_tensor=True,
                           autopep_options=autopep_options)
    return code


def export2tf2onnx(model_onnx, opset=None, verbose=True, name=None,
                   rename=False, autopep_options=None):
    """
    Exports an ONNX model to the :epkg:`tensorflow-onnx` syntax.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :param autopep_options: :epkg:`autopep8` options
    :return: python code

    .. runpython::
        :showcode:

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from mlprodict.onnx_tools.onnx_export import export2tf2onnx

        X = numpy.arange(20).reshape(10, 2).astype(numpy.float32)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = to_onnx(tr, X, target_opset=14)
        code = export2tf2onnx(onx)

        print(code)
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    code = export_template(model_onnx, templates=_tf2onnx_templates,
                           opset=opset, verbose=verbose, name=name,
                           rename=rename, use_onnx_tensor=True,
                           autopep_options=autopep_options)
    code = code.replace("], ]", "]]")
    return code


def export2numpy(model_onnx, opset=None, verbose=True, name=None,
                 rename=False, autopep_options=None):
    """
    Exports an ONNX model to the :epkg:`numpy` syntax.
    The exports does not work with all operators.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :param autopep_options: :epkg:`autopep8` options
    :return: python code

    .. runpython::
        :showcode:

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from mlprodict.onnx_tools.onnx_export import export2numpy

        X = numpy.arange(20).reshape(10, 2).astype(numpy.float32)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = to_onnx(tr, X, target_opset=14)
        code = export2numpy(onx)

        print(code)

    This can be applied to the decomposition of an einsum
    equation into simple matrix operations.

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum import decompose_einsum_equation
        from mlprodict.onnx_tools.onnx_export import export2numpy

        x1 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)
        x2 = numpy.arange(4).reshape(2, 2).astype(numpy.float32)
        r = numpy.einsum("bac,cd->ad", x1, x2)

        seq_clean = decompose_einsum_equation(
            "bac,cd->ad", strategy='numpy', clean=True)
        onx = seq_clean.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
        code = export2numpy(onx, name="einsum")
        print(code)
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    code = export_template(model_onnx, templates=_numpy_templates,
                           opset=opset, verbose=verbose, name=name,
                           rename=rename, autopep_options=autopep_options)
    for i in range(-6, 6):
        code = code.replace("axis=tuple([%d])" % i, "axis=%d" % i)
        code = code.replace("tuple([%d])" % i, "(%d, )" % i)
    return code
