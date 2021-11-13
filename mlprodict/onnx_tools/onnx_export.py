"""
@file
@brief Exports an ONNX graph in a way it can we created again
with a python script. It relies on :epkg:`jinja2` and :epkg:`autopep8`.

.. versionadded:: 0.7
"""
import numpy
import onnx
from onnx import numpy_helper
from .onnx2py_helper import (
    _var_as_dict, guess_proto_dtype, guess_proto_dtype_name)
from .onnx_export_templates import (
    get_onnx_template, get_tf2onnx_template, get_numpy_template)
from .exports.numpy_helper import make_numpy_code
from .exports.tf2onnx_helper import make_tf2onnx_code


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
    # delayed import to avoid raising an exception if not installed.
    import autopep8

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
        dims = []
        for d in t.shape.dim:
            dd = d.dim_value
            if dd == 0:
                dd = None
            dims.append(dd)
        if len(dims) == 0:
            dims = None
        if 'dim_value' in str(dims):
            raise RuntimeError(  # pragma: no cover
                "Unexpected issue in %r - %r." % (dims, t))
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
        for i_raw_name in node.input:
            i = rename_name(i_raw_name)
            if i not in used:
                used[i] = []
            used[i].append(node)
        attributes = []
        for at in node.attribute:
            temp = _var_as_dict(at)
            value = temp['value']
            if node.op_type in {'Scan', 'Loop', 'If'}:
                raise NotImplementedError(
                    "Subgraph are not yet implemented (operator=%r)."
                    "" % node.op_type)
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
    context['name'] = context['name'].replace("(", "_").replace(")", "")
    context['ir_version'] = model_onnx.ir_version
    context['producer_name'] = model_onnx.producer_name
    context['domain'] = model_onnx.domain
    context['model_version'] = model_onnx.model_version
    context['doc_string'] = model_onnx.doc_string
    context['metadata'] = {p.key: p.value for p in model_onnx.metadata_props}
    context['skip_inits'] = {}
    mark_inits = {}

    # First rendering to detect any unused or replaced initializer.
    from jinja2 import Template  # delayed import
    template = Template(templates)
    final = template.render(
        enumerate=enumerate, sorted=sorted, len=len,
        make_numpy_code=lambda *args, **kwargs: make_numpy_code(
            *args, context=context, used=used, mark_inits=mark_inits,
            **kwargs),
        make_tf2onnx_code=lambda *args, **kwargs: make_tf2onnx_code(
            *args, context=context, used=used, mark_inits=mark_inits,
            **kwargs),
        verbose=verbose, **context)

    skip_inits = set()
    for k, v in mark_inits.items():
        if len(v) == len(used[k]):
            # One initializers was removed.
            skip_inits.add(k)

    if len(skip_inits) > 0:
        # Second rendering if needed when an initializer was replaced
        # or removed.
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
            verbose=verbose, **context)

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
        :process:

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

    code = export_template(model_onnx, templates=get_onnx_template(),
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
        :process:

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

    code = export_template(model_onnx, templates=get_tf2onnx_template(),
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
        :process:

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
        :process:

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

    code = export_template(model_onnx, templates=get_numpy_template(),
                           opset=opset, verbose=verbose, name=name,
                           rename=rename, autopep_options=autopep_options)
    for i in range(-6, 6):
        code = code.replace("axis=tuple([%d])" % i, "axis=%d" % i)
        code = code.replace("tuple([%d])" % i, "(%d, )" % i)
    return code
