"""
@file
@brief Templates to export an ONNX graph in a way it can we created again
with a python script.

.. versionadded:: 0.7
"""
from textwrap import dedent

_onnx_templates = dedent("""
    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor, make_graph,
        make_tensor_value_info)


    def create_model():
        '''
        Converted ``__A__ name __B__``.

        * producer: __A__ producer_name __B__
        * version: __A__ model_version __B__
        * description: __A__ doc_string __B__
        {%- for key, val in sorted(metadata.items()): -%}
        * __A__ key __B__: __A__ val __B__
        {%- endfor %}
        '''
        # containers
        print('[containers]')   # verbose
        initializers = []
        nodes = []
        inputs = []
        outputs = []

        # opsets
        print('[opsets]')   # verbose
        opsets = __A__ opsets __B__
        target_opset = __A__ target_opset __B__

        # initializers
        print('[initializers]')   # verbose
        {% for name, value in initializers: %}
        {% if len(value.shape) == 0: %}
        value = numpy.array(__A__ value __B__, dtype=numpy.__A__ value.dtype __B__)
        {% else %}
        list_value = __A__ value.ravel().tolist() __B__
        value = numpy.array(list_value, dtype=numpy.__A__ value.dtype __B__){% if len(value.shape) > 1: %}.reshape(__A__ value.shape __B__){% endif %}
        {% endif %}
        tensor = numpy_helper.from_array(value, name='__A__ name __B__')
        initializers.append(tensor)
        {% endfor %}

        # inputs
        print('[inputs]')   # verbose
        {% for name, type, shape in inputs: %}
        value = make_tensor_value_info('__A__ name __B__', __A__ type __B__, __A__ shape __B__)
        inputs.append(value)
        {% endfor %}

        # outputs
        print('[outputs]')   # verbose
        {% for name, type, shape in outputs: %}
        value = make_tensor_value_info('__A__ name __B__', __A__ type __B__, __A__ shape __B__)
        outputs.append(value)
        {% endfor %}

        # nodes
        print('[nodes]')   # verbose
        {% for node in nodes: %}
        node = make_node(
            '__A__ node['op_type'] __B__',
            __A__ node['inputs'] __B__,
            __A__ node['outputs'] __B__,
            {% if node['name']: %}name='__A__ node['name'] __B__',{% endif %}
            {%- for name, value in node['attributes']: -%}
            __A__ name __B__=__A__ value __B__,
            {%- endfor -%}
            domain='__A__ node['domain'] __B__')
        nodes.append(node)
        {% endfor %}

        # graph
        print('[graph]')   # verbose
        graph = make_graph(nodes, '__A__ name __B__', inputs, outputs, initializers)
        onnx_model = make_model(graph)
        onnx_model.ir_version = __A__ ir_version __B__
        onnx_model.producer_name = '__A__ producer_name __B__'
        onnx_model.producer_version = '__A__ producer_version __B__'
        onnx_model.domain = '__A__ domain __B__'
        onnx_model.model_version = __A__ model_version __B__
        onnx_model.doc_string = '__A__ doc_string __B__'
        set_model_props(onnx_model, __A__ metadata __B__)

        # opsets
        print('[opset]')   # verbose
        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for dom, value in opsets.items():
            op_set = onnx_model.opset_import.add()
            op_set.domain = dom
            op_set.version = value

        return onnx_model


    onnx_model = create_model()
""".replace("__A__", "{{").replace("__B__", "}}"))


_tf2onnx_templates = dedent("""
    import inspect
    import collections
    import numpy
    from onnx import AttributeProto, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor, make_graph,
        make_tensor_value_info)
    # from tf2onnx.utils import make_name, make_sure, map_onnx_to_numpy_type
    from mlprodict.onnx_tools.exports.tf2onnx_helper import (
        make_name, make_sure, map_onnx_to_numpy_type)
    # from tf2onnx.handler import tf_op
    # from tf2onnx.graph_builder import GraphBuilder
    from mlprodict.onnx_tools.exports.tf2onnx_helper import (
        tf_op, Tf2OnnxConvert, GraphBuilder)


    @tf_op("__A__ name __B__")
    class Convert__A__ name __B__Op:

        supported_dtypes = [
            numpy.float32,
        ]

        @classmethod
        def any_version(cls, opset, ctx, node, **kwargs):
            '''
            Converter for ``__A__ name __B__``.

            * producer: __A__ producer_name __B__
            * version: __A__ model_version __B__
            * description: __A__ doc_string __B__
            {%- for key, val in sorted(metadata.items()): -%}
            * __A__ key __B__: __A__ val __B__
            {%- endfor %}
            '''
            oldnode = node
            input_name = node.input[0]
            onnx_dtype = ctx.get_dtype(input_name)
            np_dtype = map_onnx_to_numpy_type(onnx_dtype)
            make_sure(np_dtype in Convert__A__ name __B__Op.supported_dtypes, "Unsupported input type.")
            shape = ctx.get_shape(input_name)
            varx = {x: x for x in node.input}

            # initializers
            if getattr(ctx, 'verbose', False):
                print('[initializers] %r' % cls)
            {% for name, value in initializers: %}
            {% if len(value.shape) == 0: -%}
            value = numpy.array(__A__ value __B__, dtype=numpy.__A__ value.dtype __B__)
            {%- else -%}
            {% if value.size > 5: -%}
            list_value = __A__ value.ravel().tolist() __B__
            value = numpy.array(list_value, dtype=numpy.__A__ value.dtype __B__){% if len(value.shape) > 1: %}.reshape(__A__ value.shape __B__){% endif %}
            {%- else -%}
            value = numpy.array(__A__ value.ravel().tolist() __B__, dtype=numpy.__A__ value.dtype __B__){%-
                if len(value.shape) > 1: %}.reshape(__A__ value.shape __B__){% endif %}
            {%- endif -%}{%- endif %}
            varx['__A__ name __B__'] = ctx.make_const(name=make_name('init___A__ name __B__'), np_val=value).name
            {% endfor %}

            # nodes
            if getattr(ctx, 'verbose', False):
                print('[nodes] %r' % cls)
            {% for node in nodes: %}
            __A__ make_tf2onnx_code(target_opset, **node) __B__
            {% endfor %}

            # finalize
            if getattr(ctx, 'verbose', False):
                print('[replace_all_inputs] %r' % cls)
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
        value = make_tensor_value_info('__A__ name __B__', __A__ type __B__, __A__ shape __B__)
        inputs.append(value)
        {% endfor %}

        # outputs
        print('[outputs]')   # verbose
        {% for name, type, shape in outputs: %}
        value = make_tensor_value_info('__A__ name __B__', __A__ type __B__, __A__ shape __B__)
        outputs.append(value)
        {% endfor %}

        inames = [i.name for i in inputs]
        onames = [i.name for i in outputs]
        node = make_node('__A__ name __B__', inames, onames, name='__A__ name __B__')

        # graph
        print('[graph]')   # verbose
        graph = make_graph([node], '__A__ name __B__', inputs, outputs)
        onnx_model = make_model(graph)
        onnx_model.ir_version = __A__ ir_version __B__
        onnx_model.producer_name = '__A__ producer_name __B__'
        onnx_model.producer_version = '__A__ producer_version __B__'
        onnx_model.domain = '__A__ domain __B__'
        onnx_model.model_version = __A__ model_version __B__
        onnx_model.doc_string = '__A__ doc_string __B__'
        set_model_props(onnx_model, __A__ metadata __B__)

        # opsets
        print('[opset]')   # verbose
        opsets = __A__ opsets __B__
        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for dom, value in opsets.items():
            op_set = onnx_model.opset_import.add()
            op_set.domain = dom
            op_set.version = value

        return onnx_model


    onnx_raw = create_model()
    onnx_model = Tf2OnnxConvert(onnx_raw, tf_op, target_opset=__A__ opsets __B__).run()
""".replace("__A__", "{{").replace("__B__", "}}"))


_numpy_templates = dedent("""
    import numpy
    from mlprodict.onnx_tools.exports.numpy_helper import (
        argmin_use_numpy_select_last_index,
        make_slice)

    def numpy___A__name__B__(__A__ inputs[0][0] __B__{% for i in inputs[1:]: %}, __A__ i[0] __B__{% endfor %}):
        '''
        Numpy function for ``__A__ name __B__``.

        * producer: __A__ producer_name __B__
        * version: __A__ model_version __B__
        * description: __A__ doc_string __B__
        {%- for key, val in sorted(metadata.items()): -%}
        * __A__ key __B__: __A__ val __B__
        {%- endfor %}
        '''
        # initializers
        {% for name, value in initializers: -%}
        {% if name not in skip_inits: -%}
        {% if len(value.shape) == 0: -%}
        __A__ name __B__ = numpy.array(__A__ value __B__, dtype=numpy.__A__ value.dtype __B__)
        {%- else %}{% if value.size < 10: %}
        __A__ name __B__ = numpy.array(__A__ value.ravel().tolist() __B__, dtype=numpy.__A__ value.dtype __B__)
        {%- if len(value.shape) > 1: -%}.reshape(__A__ value.shape __B__){%- endif %}
        {% else %}
        list_value = __A__ value.ravel().tolist() __B__
        __A__ name __B__ = numpy.array(list_value, dtype=numpy.__A__ value.dtype __B__){% if len(value.shape) > 1: %}.reshape(__A__ value.shape __B__){% endif %}
        {% endif %}{% endif %}{% endif %}
        {%- endfor %}

        # nodes
        {% for node in nodes: %}
        __A__ make_numpy_code(target_opset, **node) __B__{% endfor %}

        return __A__ outputs[0][0] __B__{% for o in outputs[1:]: %}, __A__ o[0] __B__{% endfor %}
""".replace("__A__", "{{").replace("__B__", "}}"))
