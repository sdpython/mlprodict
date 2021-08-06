"""
@file
@brief Exports an ONNX graph in a way it can we created again
with a python script. It relies on :epkg:`jinja2` and :epkg:`autopep8`.

.. versionadded:: 0.7
"""
from textwrap import dedent
import numpy
from jinja2 import Template
import autopep8
import onnx
from onnx import numpy_helper
from .onnx2py_helper import _var_as_dict


_onnx_templates = dedent("""
    import numpy
    from onnx import numpy_helper
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor, make_graph,
        make_tensor_value_info)


    def create_model():
        # containers
        print('[containers]')   # verbose
        initializers = []
        nodes = []
        inputs = []
        outputs = []

        # opsets
        print('[opsets]')   # verbose
        opsets = {{ opsets }}
        target_opset = {{ target_opset }}

        # initializers
        print('[initializers]')   # verbose
        {% for name, value in initializers: %}
        {% if len(value.shape) == 0: %}
        value = numpy.array({{ value }}, dtype=numpy.{{ value.dtype }})
        {% else %}
        list_value = {{ value.ravel().tolist() }}
        value = numpy.array(list_value, dtype=numpy.{{ value.dtype }}){% if len(value.shape) > 1: %}.reshape({{ value.shape }}){% endif %}
        {% endif %}
        tensor = numpy_helper.from_array(value, name='{{ name }}')
        initializers.append(tensor)
        {% endfor %}

        # inputs
        print('[inputs]')   # verbose
        {% for name, type, shape in inputs: %}
        value = make_tensor_value_info('{{ name }}', {{ type }}, {{ shape }})
        inputs.append(value)
        {% endfor %}

        # outputs
        print('[outputs]')   # verbose
        {% for name, type, shape in outputs: %}
        value = make_tensor_value_info('{{ name }}', {{ type }}, {{ shape }})
        outputs.append(value)
        {% endfor %}

        # nodes
        print('[nodes]')   # verbose
        {% for node in nodes: %}
        node = make_node(
            '{{ node['op_type'] }}',
            {{ node['inputs'] }},
            {{ node['outputs'] }},
            {% if node['name']: %}name='{{ node['name'] }}',{% endif %}
            {%- for name, value in node['attributes']: -%}
            {{ name }}={{ value }},
            {%- endfor -%}
            domain='{{ node['domain'] }}')
        nodes.append(node)
        {% endfor %}

        # graph
        print('[graph]')   # verbose
        graph = make_graph(nodes, '{{ name }}', inputs, outputs, initializers)
        onnx_model = make_model(graph)
        onnx_model.ir_version = {{ ir_version }}
        onnx_model.producer_name = '{{ producer_name }}'
        onnx_model.producer_version = '{{ producer_version }}'
        onnx_model.domain = '{{ domain }}'
        onnx_model.model_version = {{ model_version }}
        onnx_model.doc_string = '{{ doc_string }}'
        set_model_props(onnx_model, {{ metadata }})

        # opsets
        print('[opset]')   # verbose
        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for dom, value in opsets.items():
            op_set = onnx_model.opset_import.add()
            op_set.domain = dom
            op_set.version = value

        return onnx_model


    onnx_model = create_model()
""")


_tf2onnx_templates = dedent("""
    import inspect
    import collections
    import numpy
    from onnx import AttributeProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor, make_graph,
        make_tensor_value_info)
    try:
        from utils import make_name
    except ImportError:

        _make_name_id = 0


        def make_name(name):
            global _make_name_id
            name = "%s_%d" % (name, _make_name_id)
            _make_name_id += 1
            return name


    class tf_op:
        _OPSETS = collections.OrderedDict()

        def __init__(self, name, domain='', **kwargs):
            if not isinstance(name, list):
                name = [name]
            self.names = name
            self.domain = domain
            self.kwargs = kwargs

        def __call__(self, func):
            for ke, va in inspect.getmembers(func, inspect.ismethod):
                if ke.startswith("version_"):
                    version = int(ke.replace("version_", ""))
                    self.register_handler(va, version, self.names, self.domain, self.kwargs)
            return func

        def register_handler(self, func, version, names, domain, kwargs):
            opset = tf_op._OPSETS.get(domain)
            if not opset:
                opset = []
                tf_op._OPSETS[domain] = opset
            while version >= len(opset):
                opset.append({})
            opset_dict = opset[version]
            for name in names:
                opset_dict[name] = (func, kwargs)


    @tf_op("{{ name }}")
    class Convert{{ name }}Op:

        supported_dtypes = [
            numpy.float32,
        ]

        @classmethod
        def any_version(cls, opset, ctx, node, **kwargs):
            '''
            Converter for ``{{ name }}``.

            * producer: {{ producer_name }}
            * version: {{ model_version }}
            * description: {{ doc_string }}
            {%- for key, val in sorted(metadata.items()): -%}
            * {{ key }}: {{ val }}
            {%- endfor %}
            '''
            oldnode = node
            input_name = node.input[0]
            onnx_dtype = ctx.get_dtype(input_name)
            utils.make_sure(onnx_dtype in Convert{{ name }}Op.supported_dtypes, "Unsupported input type.")
            shape = ctx.get_shape(input_name)
            vars = {}

            # initializers
            print('[initializers]')   # verbose
            {% for name, value in initializers: %}
            {% if len(value.shape) == 0: %}
            value = numpy.array({{ value }}, dtype=numpy.{{ value.dtype }})
            {% else %}
            list_value = {{ value.ravel().tolist() }}
            value = numpy.array(list_value, dtype=numpy.{{ value.dtype }}){% if len(value.shape) > 1: %}.reshape({{ value.shape }}){% endif %}
            {% endif %}
            r_{{ name }} = ctx.make_const(name=make_name('init_{{ name }}'), np_val=value)
            vars['{{ name }}'] = r_{{ name }}.name
            {% endfor %}

            # nodes
            print('[nodes]')   # verbose
            {% for node in nodes: %}
            attr = dict(
                {%- for name, value in node['attributes']: -%}
                {{ name }}={{ value }},
                {%- endfor -%})
            inputs = [{% for name in node['inputs']: -%}vars['{{ name }}'], {%- endfor %}]
            node = ctx.make_node(
                '{{ node['op_type'] }}', inputs=inputs, attr=attr,{% if node['domain']: -%} domain='{{ node['domain'] }}', {% endif %}
                name=make_name('{{ node['name'] }}'))
            {% for i, name in enumerate(node['outputs']): -%}
            vars['{{ name }}'] = node.output[{{ i }}]
            {%- endfor %}
            {% endfor %}

            # finalize
            ctx.replace_all_inputs(oldnode.output[0], node.output[0])
            ctx.remove_node(oldnode.name)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        return cls.any_version(13, ctx, node, **kwargs)


    def create_model():
        inputs = []
        outputs = []

        # inputs
        print('[inputs]')   # verbose
        {% for name, type, shape in inputs: %}
        value = make_tensor_value_info('{{ name }}', {{ type }}, {{ shape }})
        inputs.append(value)
        {% endfor %}

        # outputs
        print('[outputs]')   # verbose
        {% for name, type, shape in outputs: %}
        value = make_tensor_value_info('{{ name }}', {{ type }}, {{ shape }})
        outputs.append(value)
        {% endfor %}

        inames = [i.name for i in inputs]
        onames = [i.name for i in outputs]
        node = make_node('{{ name }}', inames, onames, name='{{ name }}')

        # graph
        print('[graph]')   # verbose
        graph = make_graph([node], '{{ name }}', inputs, outputs)
        onnx_model = make_model(graph)
        onnx_model.ir_version = {{ ir_version }}
        onnx_model.producer_name = '{{ producer_name }}'
        onnx_model.producer_version = '{{ producer_version }}'
        onnx_model.domain = '{{ domain }}'
        onnx_model.model_version = {{ model_version }}
        onnx_model.doc_string = '{{ doc_string }}'
        set_model_props(onnx_model, {{ metadata }})

        # opsets
        print('[opset]')   # verbose
        opsets = {{ opsets }}
        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for dom, value in opsets.items():
            op_set = onnx_model.opset_import.add()
            op_set.domain = dom
            op_set.version = value

        return onnx_model


    class Rewrite:

        def __init__(self, onnx_model, tf_op):
            self._onnx_model = onnx_model        
            self._nodes = list(onnx_model.graph.node)
            self._tf_op = tf_op
        
        def make_node(self, op_type, inputs, attr=None, outputs=None,
                      name=None, domain=''):
            if attr is None:
                attr = {}
            if name is None:
                name = make_name(op_type)

            if outputs is None:
                outputs = [name + ":" + str(i) for i in range(output_count)]

            output_count = len(outputs)
            raw_attr = {}
            onnx_attrs = []
            for a, v in attr.items():
                if isinstance(v, AttributeProto):
                    onnx_attrs.append(v)
                else:
                    raw_attr[a] = v

            n = self.get_node_by_name(name)

            for o in outputs:
                n = self.get_node_by_output_in_current_graph(o)

            onnx_node = make_node(
                op_type, inputs, outputs, name=name, domain=domain, **raw_attr)

            self._nodes.append(onnx_node)
            return node

        def rewrite(self):
            print('[rewrite]')   # verbose
            done = {}
            modif = 1
            while modif > 0:
                modif = 0
                for node in self._nodes:
                    if done.get(node.name, False):
                        continue
                    domain = node.domain
                    if domain not in self._tf_op._OPSETS:
                        continue
                    rews = self._tf_op._OPSETS[domain]
                    # look for an opset
                    # call the rewriter
            
        

    onnx_model = create_model()
    onnx_rewritten = Rewrite(onnx_model, tf_op).rewrite()
""")


def export_template(model_onnx, templates, opset=None, verbose=True, name=None):
    """
    Exports an ONNX model to the onnx syntax.

    :param model_onnx: string or ONNX graph
    :param templates: exporting templates
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: insert prints
    :param name: to overwrite onnx name
    :return: python code
    """
    # containers
    context = {}

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

    # inits
    initializers = []
    for init in model_onnx.graph.initializer:
        value = numpy_helper.to_array(init)
        initializers.append((init.name, value))
    context['initializers'] = initializers

    # inputs
    inputs = []
    for inp in model_onnx.graph.input:
        t = inp.type.tensor_type
        dims = tuple(t.shape.dim)
        inputs.append((inp.name, t.elem_type, dims))
    context['inputs'] = inputs

    # outputs
    outputs = []
    for inp in model_onnx.graph.output:
        t = inp.type.tensor_type
        dims = tuple(t.shape.dim)
        outputs.append((inp.name, t.elem_type, dims))
    context['outputs'] = outputs

    # node
    nodes = []
    for node in model_onnx.graph.node:
        attributes = []
        for at in node.attribute:
            temp = _var_as_dict(at)
            value = temp['value']
            if isinstance(value, str):
                attributes.append((at.name, "%r" % value))
            else:
                if isinstance(value, numpy.ndarray):
                    attributes.append((at.name, repr(value.tolist())))
                else:
                    attributes.append((at.name, repr(value)))
        d = dict(name=node.name, op_type=node.op_type,
                 domain=node.domain, inputs=node.input,
                 outputs=node.output, attributes=attributes)
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

    # final
    template = Template(templates)
    final = template.render(
        enumerate=enumerate, sorted=sorted, len=len,
        **context)

    if not verbose:
        rows = final.split("\n")
        final = "\n".join(_ for _ in rows if not _.endswith("#  verbose"))
    return autopep8.fix_code(final)


def export2onnx(model_onnx, opset=None, verbose=True, name=None):
    """
    Exports an ONNX model to the :epkg:`onnx` syntax.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :return: python code
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    return export_template(model_onnx, templates=_onnx_templates,
                           opset=opset, verbose=verbose, name=name)


def export2tf2onnx(model_onnx, opset=None, verbose=True, name=None):
    """
    Exports an ONNX model to the e:pkg:`tensorflow-onnx` syntax.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :return: python code
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    return export_template(model_onnx, templates=_tf2onnx_templates,
                           opset=opset, verbose=verbose, name=name)
