"""
@file
@brief
"""
from io import BytesIO
import json
import warnings
import numpy
from onnx import load, load_model, checker, shape_inference
from onnx import onnx_pb as onnx_proto
from onnx import numpy_helper
from .onnx_inference_node import OnnxInferenceNode


class OnnxInference:
    """
    Loads an :epkg:`ONNX` file or object or stream.
    Computes the output of the :epkg:`ONNX` graph.
    """

    def __init__(self, onnx_or_bytes_or_stream, runtime=None, skip_run=False):
        """
        @param      onnx_or_bytes_or_stream     :epkg:`onnx` object,
                                                bytes, or filename or stream
        @param      runtime                     runtime options
        @param      skip_run                    do not build the runtime
        """
        if isinstance(onnx_or_bytes_or_stream, bytes):
            self.obj = load_model(BytesIO(onnx_or_bytes_or_stream))
        elif isinstance(onnx_or_bytes_or_stream, BytesIO):
            self.obj = load_model(onnx_or_bytes_or_stream)
        elif isinstance(onnx_or_bytes_or_stream, str):
            self.obj = load(onnx_or_bytes_or_stream)
        elif hasattr(onnx_or_bytes_or_stream, 'graph'):
            self.obj = onnx_or_bytes_or_stream
        else:
            raise TypeError("Unable to handle type {}.".format(
                type(onnx_or_bytes_or_stream)))
        self.runtime = runtime
        self.skip_run = skip_run
        self._init()

    def __getstate__(self):
        """
        To pickle the object.
        """
        return {'onnx': self.obj.SerializeToString(),
                'runtime': self.runtime,
                'skip_run': self.skip_run}

    def __setstate__(self, state):
        """
        To unpickle the object.
        """
        onx = state['onnx']
        self.obj = load_model(BytesIO(onx))
        self.runtime = state['runtime']
        self.skip_run = state['skip_run']
        self._init()

    def _init(self):
        """
        Prepares the instance to deliver predictions.
        """
        self.graph_ = self.to_sequence()
        self.outputs_ = self.graph_['outputs']
        if not self.skip_run:
            if self.runtime == 'onnxruntime-whole':
                # Loads the onnx with onnxruntime as a single file.
                del self.graph_
                from .ops_whole.session import OnnxWholeSession
                self._whole = OnnxWholeSession(self.obj, self.runtime)
                self._run = self._run_whole_runtime
            else:
                self.sequence_ = self.graph_['sequence']
                self.inits_ = self.graph_['inits']
                variables = self.inits_.copy()
                for node in self.sequence_:
                    node.setup_runtime(self.runtime, variables)
                    for k, v in node.ops_.typed_outputs_:
                        variables[k] = v
                self._run = self._run_sequence_runtime

    def __str__(self):
        """
        usual
        """
        return str(self.obj)

    def __repr__(self):
        """
        usual
        """
        return "OnnxInference(...)"

    def check_model(self):
        """
        Checks the model follow :epkg:`ONNX` conventions.
        """
        checker.check_model(self.obj)

    def shape_inference(self):
        """
        Infers the shape of the outputs.

        @return     A new :epkg:`ONNX` graph which defined outputs.
        """
        return shape_inference.infer_shapes(self.obj)

    @staticmethod
    def _elem_type_as_str(elem_type):
        if elem_type == onnx_proto.TensorProto.FLOAT:  # pylint: disable=E1101
            return 'float'
        if elem_type == onnx_proto.TensorProto.BOOL:  # pylint: disable=E1101
            return 'bool'
        if elem_type == onnx_proto.TensorProto.DOUBLE:  # pylint: disable=E1101
            return 'double'
        if elem_type == onnx_proto.TensorProto.STRING:  # pylint: disable=E1101
            return 'str'
        if elem_type == onnx_proto.TensorProto.INT64:  # pylint: disable=E1101
            return 'int64'
        if elem_type == onnx_proto.TensorProto.INT32:  # pylint: disable=E1101
            return 'int32'

        # The following code should be refactored.
        selem = str(elem_type)

        if selem.startswith("tensor_type"):
            this = elem_type.tensor_type
            et = OnnxInference._elem_type_as_str(this.elem_type)
            shape = this.shape
            dim = shape.dim
            dims = [d.dim_value for d in dim]
            if len(dims) == 0:
                dims = '?'
            return {'kind': 'tensor', 'elem': et, 'shape': shape}

        if selem.startswith("map_type"):
            this = elem_type.map_type
            kt = OnnxInference._elem_type_as_str(this.key_type)
            vt = OnnxInference._elem_type_as_str(this.value_type)
            return {'kind': 'map', 'key': kt, 'value': vt}

        import pprint
        raise NotImplementedError("elem_type {} is unknown\nfieds:\n{}.".format(
            elem_type, pprint.pformat(dir(elem_type))))

    @staticmethod
    def _var_as_dict(var):
        """
        Converts a protobuf object into something readable.
        The current implementation relies on :epkg:`json`.
        That's not the most efficient way.
        """
        if hasattr(var, 'type'):
            # variable
            if var.type is not None:
                if hasattr(var.type, 'tensor_type') and var.type.tensor_type.elem_type > 0:
                    t = var.type.tensor_type
                    elem_type = OnnxInference._elem_type_as_str(t.elem_type)
                    shape = t.shape
                    dim = shape.dim
                    dims = [d.dim_value for d in dim]
                    if len(dims) == 0:
                        dims = '?'
                    dtype = dict(kind='tensor', elem=elem_type,
                                 shape=tuple(dims))
                elif hasattr(var.type, 'real'):
                    dtype = dict(kind='real', elem=var.type.real)
                elif hasattr(var.type, "sequence_type") and var.type.sequence_type is not None:
                    t = var.type.sequence_type
                    elem_type = OnnxInference._elem_type_as_str(t.elem_type)
                    dtype = dict(kind='sequence', elem=elem_type)
                else:
                    import pprint
                    raise NotImplementedError("Unable to convert a type into a dictionary for {}. "
                                              "Available fields: {}.".format(var.type, pprint.pformat(dir(var.type))))
            else:
                import pprint
                raise NotImplementedError("Unable to convert variable into a dictionary for {}. "
                                          "Available fields: {}.".format(var, pprint.pformat(dir(var.type))))

            res = dict(name=var.name, type=dtype)

            if hasattr(var, 'floats') and dtype.get('elem', None) == 6:
                res['value'] = numpy.array(var.floats)
            elif hasattr(var, 'strings') and dtype.get('elem', None) == 8:
                res['value'] = numpy.array(var.strings)
            elif hasattr(var, 'ints') and dtype.get('elem', None) == 7:
                res['value'] = numpy.array(var.ints)
            elif hasattr(var, 'f') and dtype.get('elem', None) == 1:
                res['value'] = var.f
            elif hasattr(var, 's') and dtype.get('elem', None) == 3:
                res['value'] = var.s
            elif hasattr(var, 'i') and dtype.get('elem', None) == 2:
                res['value'] = var.i
            elif "'value'" in str(var):
                warnings.warn("No value: {} -- {}".format(
                    dtype, str(var).replace("\n", "").replace(" ", "")))
            return res

        elif hasattr(var, 'op_type'):
            if hasattr(var, 'attribute'):
                atts = {}
                for att in var.attribute:
                    atts[att.name] = OnnxInference._var_as_dict(att)
            return dict(name=var.name, op_type=var.op_type,
                        domain=var.domain, atts=atts)

        elif hasattr(var, 'dims') and len(var.dims) > 0:
            # initializer
            dims = [d for d in var.dims]
            if var.data_type == 1 and var.float_data is not None:
                data = numpy.array(var.float_data, copy=False).reshape(dims)
            elif var.data_type == 6 and var.int32_data is not None:
                data = numpy.array(var.int32_data, copy=False).reshape(dims)
            elif var.data_type == 7 and var.int64_data is not None:
                data = numpy.array(var.int64_data, copy=False).reshape(dims)
            else:
                raise NotImplementedError(
                    "Iniatilizer {} cannot be converted into a dictionary.".format(var))
            return dict(name=var.name, value=data)

        elif var.data_type > 0:
            if var.data_type == 1 and var.float_data is not None:
                data = numpy.array(var.float_data, copy=False)
            elif var.data_type == 6 and var.int32_data is not None:
                data = numpy.array(var.int32_data, copy=False)
            elif var.data_type == 7 and var.int64_data is not None:
                data = numpy.array(var.int64_data, copy=False)
            else:
                raise NotImplementedError(
                    "Iniatilizer {} cannot be converted into a dictionary.".format(var))
            return dict(name=var.name, value=data)

        else:
            raise NotImplementedError(
                "Unable to guess which object it is.\n{}".format(var))

    @staticmethod
    def _type_to_string(dtype):
        """
        Converts a type into a readable string.
        """
        if not isinstance(dtype, dict):
            dtype_ = OnnxInference._var_as_dict(dtype)
        else:
            dtype_ = dtype
        if dtype_["kind"] == 'tensor':
            return "{0}({1})".format(dtype_['elem'], dtype_['shape'])
        if dtype_['kind'] == 'sequence':
            return "[{0}]".format(OnnxInference._type_to_string(dtype_['elem']))
        if dtype_["kind"] == 'map':
            return "{{{0}, {1}}}".format(dtype_['key'], dtype_['value'])
        raise NotImplementedError(
            "Unable to convert into string {} or {}.".format(dtype, dtype_))

    def to_dot(self, **params):
        """
        Produces a :epkg:`DOT` language string for the graph.

        @param      params      additional params to draw the graph
        @return                 string

        Default options for the graph are:

        ::

            options = {
                'orientation': 'portrait',
                'ranksep': '0.25',
                'nodesep': '0.05',
                'width': '0.5',
                'height': '0.1',
            }

        One example:

        .. exref::
            :title: Convert ONNX into DOT

            An example on how to convert an :epkg:`ONNX`
            graph into :epkg:`DOT`.

            .. runpython::
                :showcode:

                import numpy
                from skl2onnx.algebra.onnx_ops import OnnxLinearRegressor
                from skl2onnx.common.data_types import FloatTensorType
                from mlprodict.onnxrt import OnnxInference

                pars = dict(coefficients=numpy.array([1., 2.]),
                            intercepts=numpy.array([1.]),
                            post_transform='NONE')
                onx = OnnxLinearRegressor('X', output_names=['Y'], **pars)
                model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                        outputs=[('Y', FloatTensorType([1]))])
                oinf = OnnxInference(model_def)
                print(oinf.to_dot())

            See an example of representation in notebook
            :ref:`onnxvisualizationrst`.
        """
        options = {
            'orientation': 'portrait',
            'ranksep': '0.25',
            'nodesep': '0.05',
            'width': '0.5',
            'height': '0.1',
        }
        options.update(params)

        inter_vars = {}
        exp = ["digraph{"]
        for opt in {'orientation', 'pad', 'nodesep', 'ranksep'}:
            if opt in options:
                exp.append("  {}={};".format(opt, options[opt]))
        fontsize = 10

        # inputs
        exp.append("")
        for obj in self.obj.graph.input:
            dobj = OnnxInference._var_as_dict(obj)
            exp.append('  {0} [shape=box color=red label="{0}\\n{1}" fontsize={2}];'.format(
                dobj['name'], OnnxInference._type_to_string(dobj['type']), fontsize))
            inter_vars[obj.name] = obj

        # outputs
        exp.append("")
        for obj in self.obj.graph.output:
            dobj = OnnxInference._var_as_dict(obj)
            exp.append('  {0} [shape=box color=green label="{0}\\n{1}" fontsize={2}];'.format(
                dobj['name'], OnnxInference._type_to_string(dobj['type']), fontsize))
            inter_vars[obj.name] = obj

        # initializer
        exp.append("")
        for obj in self.obj.graph.initializer:
            dobj = OnnxInference._var_as_dict(obj)
            val = dobj['value']
            flat = val.flatten()
            if flat.shape[0] < 9:
                st = str(val)
            else:
                st = str(val)
                if len(st) > 30:
                    st = st[:30] + '...'
            st = st.replace('\n', '\\n')
            kind = ""
            exp.append('  {0} [shape=box label="{0}\\n{4}{1}({2})\\n{3}" fontsize={5}];'.format(
                dobj['name'], dobj['value'].dtype,
                dobj['value'].shape, st, kind, fontsize))
            inter_vars[obj.name] = obj

        # nodes
        for node in self.obj.graph.node:
            exp.append("")
            for out in node.output:
                if out not in inter_vars:
                    inter_vars[out] = out
                    exp.append(
                        '  {0} [shape=box label="{0}" fontsize={1}];'.format(out, fontsize))

            dobj = OnnxInference._var_as_dict(node)
            if dobj['name'].strip() == '':
                raise RuntimeError(
                    "Issue with a node\n{}\n----\n{}".format(dobj, node))

            atts = []
            if 'atts' in dobj:
                for k, v in sorted(dobj['atts'].items()):
                    val = None
                    if 'value' in v:
                        val = str(v['value'])
                        sl = max(30 - len(k), 10)
                        if len(val) > sl:
                            val = val[:sl] + "..."
                    if val is not None:
                        atts.append('{}={}'.format(k, val))
            satts = "" if len(atts) == 0 else ("\\n" + "\\n".join(atts))
            exp.append('  {1} [shape=box style="filled,rounded" color=orange label="{0}\\n({1}){2}" fontsize={3}];'.format(
                dobj['op_type'], dobj['name'], satts, fontsize))

            for inp in node.input:
                exp.append("  {} -> {};".format(inp, node.name))
            for out in node.output:
                exp.append("  {} -> {};".format(node.name, out))

        exp.append('}')
        return "\n".join(exp)

    def to_json(self, indent=2):
        """
        Converts an :epkg:`ONNX` model into :epkg:`JSON`.

        @param      indent      indentation
        @return                 string

        .. exref::
            :title: Convert ONNX into JSON

            An example on how to convert an :epkg:`ONNX`
            graph into :epkg:`JSON`.

            .. runpython::
                :showcode:

                import numpy
                from skl2onnx.algebra.onnx_ops import OnnxLinearRegressor
                from skl2onnx.common.data_types import FloatTensorType
                from mlprodict.onnxrt import OnnxInference

                pars = dict(coefficients=numpy.array([1., 2.]),
                            intercepts=numpy.array([1.]),
                            post_transform='NONE')
                onx = OnnxLinearRegressor('X', output_names=['Y'], **pars)
                model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                        outputs=[('Y', FloatTensorType([1]))])
                oinf = OnnxInference(model_def)
                print(oinf.to_json())
        """

        def _to_json(obj):
            s = str(obj)
            rows = ['{']
            leave = None
            for line in s.split('\n'):
                if line.endswith("{"):
                    rows.append('"%s": {' % line.strip('{ '))
                elif ':' in line:
                    spl = line.strip().split(':')
                    if len(spl) != 2:
                        raise RuntimeError(
                            "Unable to interpret line '{}'.".format(line))

                    if spl[0].strip() in ('type', ):
                        st = spl[1].strip()
                        if st in {'INT', 'INTS', 'FLOAT', 'FLOATS', 'STRING', 'STRINGS'}:
                            spl[1] = '"{}"'.format(st)

                    if spl[0] in ('floats', 'ints'):
                        if leave:
                            rows.append("{},".format(spl[1]))
                        else:
                            rows.append('"{}": [{},'.format(
                                spl[0], spl[1].strip()))
                            leave = spl[0]
                    elif leave:
                        rows[-1] = rows[-1].strip(',')
                        rows.append('],')
                        rows.append('"{}": {},'.format(
                            spl[0].strip(), spl[1].strip()))
                        leave = None
                    else:
                        rows.append('"{}": {},'.format(
                            spl[0].strip(), spl[1].strip()))
                elif line.strip() == "}":
                    rows[-1] = rows[-1].rstrip(",")
                    rows.append(line + ",")
                elif line:
                    raise RuntimeError(
                        "Unable to interpret line '{}'.".format(line))
            rows[-1] = rows[-1].rstrip(',')
            rows.append("}")
            js = "\n".join(rows)

            try:
                content = json.loads(js)
            except json.decoder.JSONDecodeError as e:
                js2 = "\n".join("%04d %s" % (i + 1, line)
                                for i, line in enumerate(js.split("\n")))
                raise RuntimeError(
                    "Unable to parse JSON\n{}".format(js2)) from e
            return content

        # meta data
        final_obj = {}
        for k in {'ir_version', 'producer_name', 'producer_version',
                  'domain', 'model_version', 'doc_string'}:
            if hasattr(self.obj, k):
                final_obj[k] = getattr(self.obj, k)

        # inputs
        inputs = []
        for obj in self.obj.graph.input:
            st = _to_json(obj)
            inputs.append(st)
        final_obj['inputs'] = inputs

        # outputs
        outputs = []
        for obj in self.obj.graph.output:
            st = _to_json(obj)
            outputs.append(st)
        final_obj['outputs'] = outputs

        # init
        inits = {}
        for obj in self.obj.graph.initializer:
            value = numpy_helper.to_array(obj).tolist()
            inits[obj.name] = value
        final_obj['initializers'] = inits

        # nodes
        nodes = []
        for obj in self.obj.graph.node:
            node = dict(name=obj.name, op_type=obj.op_type, domain=obj.domain,
                        inputs=[str(_) for _ in obj.input],
                        outputs=[str(_) for _ in obj.output],
                        attributes={})
            for att in obj.attribute:
                st = _to_json(att)
                node['attributes'][st['name']] = st
                del st['name']
            nodes.append(node)
        final_obj['nodes'] = nodes

        return json.dumps(final_obj, indent=indent)

    def to_sequence(self):
        """
        Produces a graph to facilitate the execution.

        One example:

        .. exref::
            :title: Convert ONNX into graph

            An example on how to convert an :epkg:`ONNX`
            graph into a graph.

            .. runpython::
                :showcode:

                import pprint
                import numpy
                from skl2onnx.algebra.onnx_ops import OnnxLinearRegressor
                from skl2onnx.common.data_types import FloatTensorType
                from mlprodict.onnxrt import OnnxInference

                pars = dict(coefficients=numpy.array([1., 2.]),
                            intercepts=numpy.array([1.]),
                            post_transform='NONE')
                onx = OnnxLinearRegressor('X', output_names=['Y'], **pars)
                model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                        outputs=[('Y', FloatTensorType([1]))])
                oinf = OnnxInference(model_def)
                pprint.pprint(oinf.to_sequence())

            See an example of representation in notebook
            :ref:`onnxvisualizationrst`.
        """
        inits = {}
        variables = {}
        outputs = {}
        nodes = {}

        # inputs
        for obj in self.obj.graph.input:
            variables[obj.name] = OnnxInference._var_as_dict(obj)

        # outputs
        for obj in self.obj.graph.output:
            outputs[obj.name] = OnnxInference._var_as_dict(obj)

        # initializer
        for obj in self.obj.graph.initializer:
            init_obj = OnnxInference._var_as_dict(obj)
            if init_obj is None:
                raise RuntimeError(
                    "Unable to convert an initializer\n{}".format(obj))
            inits[obj.name] = init_obj
            if 'value' not in inits[obj.name]:
                raise RuntimeError("One initializer has no value: '{}'\n{}\n{}".format(
                    obj.name, inits[obj.name], obj))

        # nodes
        for node in self.obj.graph.node:
            dobj = OnnxInference._var_as_dict(node)
            if dobj is None:
                raise RuntimeError("Unable to convert a node\n{}".format(node))
            if 'atts' in dobj:
                atts = dobj['atts']
                for k, v in atts.items():
                    if not isinstance(v, dict) or 'value' not in v:
                        raise RuntimeError("A parameter has no value '{}' for node '{}'\nv={}\ndobj=[{}]".format(
                            k, node.name, v, node))
            nodes[node.name] = OnnxInferenceNode(node, dobj)

        # names
        names = {}
        for k, v in inits.items():
            names[k] = ('C', v)
        for k, v in variables.items():
            names[k] = ('I', v)
        for k, v in outputs.items():
            names[k] = ('O', v)
        for k, v in nodes.items():
            names[k] = ('N', v)

        # ordering
        order = {}
        modif = 1
        intermediate = {}
        while modif > 0:
            modif = 0
            for k, v in names.items():
                if k in order:
                    continue
                if v[0] in {'I', 'C'}:
                    order[k] = len(order)
                    modif += 1
                elif v[0] == 'O':
                    continue
                else:
                    if all(inp in order for inp in v[1].inputs):
                        order[k] = len(order)
                        modif += 1
                        for o in v[1].outputs:
                            if o in order:
                                raise RuntimeError(
                                    "Two nodes share the same output '{}'.".format(o))
                            order[o] = len(order)
                            intermediate[o] = None
                            modif += 1

        # compute
        rev = [(v, k) for k, v in order.items()]
        rev.sort()
        sequence = []
        for _, name in rev:
            if name not in nodes:
                continue
            node = nodes[name]
            node.set_order(len(sequence))
            sequence.append(node)

        # defines where an intermediare output is not needed
        last_used = {}
        for node in sequence:
            for inp in node.inputs:
                last_used[inp] = node.order
        for k, ord in last_used.items():
            sequence[ord].add_variable_to_clean(k)

        return dict(inits=inits, inputs=variables, outputs=outputs,
                    nodes=nodes, sequence=sequence, intermediate=intermediate)

    def run(self, inputs, clean_right_away=False, verbose=0, fLOG=None):
        """
        Computes the predictions for this :epkg:`onnx` graph.

        @param      inputs              inputs as dictionary
        @param      clean_right_away    clean the intermediate outputs
                                        as soon as they are not needed
        @param      verbose             display information while predicting
        @param      fLOG                logging function if *verbose > 0*
        @return                         outputs as dictionary

        .. exref::
            :title: Computes predictions with any runtime

            The following example compares predictions
            between :epkg:`scikit-learn` and this runtime
            for the python runtime.

            .. runpython::
                :showcode:

                import numpy
                from sklearn.linear_model import LinearRegression
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from skl2onnx import to_onnx
                from mlprodict.onnxrt import OnnxInference

                iris = load_iris()
                X, y = iris.data, iris.target
                X_train, X_test, y_train, _ = train_test_split(X, y)
                clr = LinearRegression()
                clr.fit(X_train, y_train)

                exp = clr.predict(X_test[:5])
                print(exp)

                model_def = to_onnx(clr, X_train.astype(numpy.float32))
                oinf = OnnxInference(model_def)
                y = oinf.run({'X': X_test[:5]})
                print(y)
        """
        return self._run(inputs, clean_right_away=False, verbose=verbose, fLOG=fLOG)

    def _run_sequence_runtime(self, inputs, clean_right_away=False, verbose=0, fLOG=None):
        if clean_right_away:
            raise NotImplementedError("clean_right_away=true not implemented.")
        values = inputs.copy()
        for k, v in self.inits_.items():
            values[k] = v['value']
        if verbose == 0 or fLOG is None:
            for node in self.sequence_:
                node.run(values)
        else:
            if verbose >= 2:
                for k in sorted(values):
                    fLOG("-k='{}' shape={} dtype={}".format(
                        k, values[k].shape, values[k].dtype))
            keys = set(values)
            for node in self.sequence_:
                if verbose >= 1:
                    fLOG(node)
                node.run(values)
                for k in sorted(values):
                    if k not in keys:
                        if isinstance(values[k], numpy.ndarray):
                            fLOG("+k='{}': {} (dtype={})".format(
                                k, values[k].shape, values[k].dtype))
                        else:
                            fLOG("+k='{}': {}".format(
                                k, type(values[k])))
                keys = set(values)

        return {k: values[k] for k in self.outputs_}

    def _run_whole_runtime(self, inputs, clean_right_away=False, verbose=0, fLOG=None):
        if verbose != 0:
            raise NotImplementedError("verbose option not implemented.")
        if clean_right_away:
            raise RuntimeError(
                "clean_right_away=true does not wrok with this runtime.")
        res = self._whole.run(inputs)
        return {k: v for k, v in zip(self.outputs_, res)}
