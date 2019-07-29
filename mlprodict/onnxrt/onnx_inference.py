"""
@file
@brief Implements a class able to compute the predictions
from on an :epkg:`ONNX` model.
"""
from collections import OrderedDict
from io import BytesIO
import json
from time import perf_counter
import warnings
import numpy
from onnx import load, load_model, checker, shape_inference
from onnx import onnx_pb as onnx_proto
from onnx import numpy_helper
from onnx.helper import make_model
from .onnx_inference_node import OnnxInferenceNode
from .onnx_inference_manipulations import select_model_inputs_outputs, enumerate_model_node_outputs
from .onnx2py_helper import _var_as_dict, _type_to_string
from .sklearn_helper import enumerate_fitted_arrays, pairwise_array_distances


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
        elif isinstance(onnx_or_bytes_or_stream, onnx_proto.GraphProto):
            self.obj = make_model(onnx_or_bytes_or_stream,
                                  producer_name='mlprodict')
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
        self.target_opset_ = self.graph_['targets'].get('', None)
        if not self.skip_run:
            if self.runtime == 'onnxruntime1':
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
                    node.setup_runtime(self.runtime, variables, self.__class__,
                                       target_opset=self.target_opset_)
                    if hasattr(node, 'ops_') and hasattr(node.ops_, 'typed_outputs_'):
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

    @property
    def input_names(self):
        """
        Returns the names of all inputs.
        """
        return [_.name for _ in self.obj.graph.input]

    @property
    def output_names(self):
        """
        Returns the names of all outputs.
        """
        return [_.name for _ in self.obj.graph.output]

    def to_dot(self, recursive=False, prefix='', **params):
        """
        Produces a :epkg:`DOT` language string for the graph.

        @param      params      additional params to draw the graph
        @param      recursive   also show subgraphs inside operator like
                                @see cl Scan
        @param      prefix      prefix for every node name
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
            dobj = _var_as_dict(obj)
            exp.append('  {3}{0} [shape=box color=red label="{0}\\n{1}" fontsize={2}];'.format(
                dobj['name'], _type_to_string(dobj['type']), fontsize, prefix))
            inter_vars[obj.name] = obj

        # outputs
        exp.append("")
        for obj in self.obj.graph.output:
            dobj = _var_as_dict(obj)
            exp.append('  {3}{0} [shape=box color=green label="{0}\\n{1}" fontsize={2}];'.format(
                dobj['name'], _type_to_string(dobj['type']), fontsize, prefix))
            inter_vars[obj.name] = obj

        # initializer
        exp.append("")
        for obj in self.obj.graph.initializer:
            dobj = _var_as_dict(obj)
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
            exp.append('  {6}{0} [shape=box label="{0}\\n{4}{1}({2})\\n{3}" fontsize={5}];'.format(
                dobj['name'], dobj['value'].dtype,
                dobj['value'].shape, st, kind, fontsize, prefix))
            inter_vars[obj.name] = obj

        # nodes
        for node in self.obj.graph.node:
            exp.append("")
            for out in node.output:
                if out not in inter_vars:
                    inter_vars[out] = out
                    exp.append(
                        '  {2}{0} [shape=box label="{0}" fontsize={1}];'.format(out, fontsize, prefix))

            dobj = _var_as_dict(node)
            if dobj['name'].strip() == '':
                raise RuntimeError(
                    "Issue with a node\n{}\n----\n{}".format(dobj, node))

            atts = []
            if 'atts' in dobj:
                for k, v in sorted(dobj['atts'].items()):
                    val = None
                    if 'value' in v:
                        val = str(v['value']).replace(
                            "\n", "\\n").replace('"', "'")
                        sl = max(30 - len(k), 10)
                        if len(val) > sl:
                            val = val[:sl] + "..."
                    if val is not None:
                        atts.append('{}={}'.format(k, val))
            satts = "" if len(atts) == 0 else ("\\n" + "\\n".join(atts))

            if recursive and node.op_type in {'Scan'}:
                # creates the subgraph
                body = dobj['atts']['body']['value']
                oinf = OnnxInference(
                    body, runtime=self.runtime, skip_run=self.skip_run)
                subprefix = prefix + "B_"
                subdot = oinf.to_dot(recursive=recursive, prefix=subprefix)
                lines = subdot.split("\n")
                start = 0
                for i, line in enumerate(lines):
                    if '[' in line:
                        start = i
                        break
                subgraph = "\n".join(lines[start:])

                # connecting the subgraph
                exp.append("  subgraph cluster_{}{} {{".format(
                    node.op_type, id(node)))
                exp.append('    label="{0}\\n({1}){2}";'.format(
                    dobj['op_type'], dobj['name'], satts))
                exp.append('    fontsize={0};'.format(fontsize))
                exp.append('    color=black;')
                exp.append(
                    '\n'.join(map(lambda s: '  ' + s, subgraph.split('\n'))))

                for inp1, inp2 in zip(node.input, body.input):
                    exp.append(
                        "  {0}{1} -> {2}{3};".format(prefix, inp1, subprefix, inp2.name))
                for out1, out2 in zip(body.output, node.output):
                    exp.append(
                        "  {0}{1} -> {2}{3};".format(subprefix, out1.name, prefix, out2))

            else:
                exp.append('  {4}{1} [shape=box style="filled,rounded" color=orange label="{0}\\n({1}){2}" fontsize={3}];'.format(
                    dobj['op_type'], dobj['name'], satts, fontsize, prefix))

                for inp in node.input:
                    exp.append(
                        "  {0}{1} -> {0}{2};".format(prefix, inp, node.name))
                for out in node.output:
                    exp.append(
                        "  {0}{1} -> {0}{2};".format(prefix, node.name, out))

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
        targets = {}
        for o in self.obj.opset_import:
            targets[o.domain] = o.version

        # inputs
        for obj in self.obj.graph.input:
            variables[obj.name] = _var_as_dict(obj)

        # outputs
        for obj in self.obj.graph.output:
            if hasattr(obj, 'type') and str(obj.type) != '':
                outputs[obj.name] = _var_as_dict(obj)
            else:
                outputs[obj.name] = {'name': obj.name}

        # initializer
        for obj in self.obj.graph.initializer:
            init_obj = _var_as_dict(obj)
            if init_obj is None:
                raise RuntimeError(
                    "Unable to convert an initializer\n{}".format(obj))
            inits[obj.name] = init_obj
            if 'value' not in inits[obj.name]:
                raise RuntimeError("One initializer has no value: '{}'\n{}\n{}".format(
                    obj.name, inits[obj.name], obj))

        # nodes
        for node in self.obj.graph.node:
            dobj = _var_as_dict(node)
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
                    nodes=nodes, sequence=sequence, intermediate=intermediate,
                    targets=targets)

    def run(self, inputs, clean_right_away=False,
            intermediate=False, verbose=0, node_time=False,
            fLOG=None):
        """
        Computes the predictions for this :epkg:`onnx` graph.

        @param      inputs              inputs as dictionary
        @param      clean_right_away    clean the intermediate outputs
                                        as soon as they are not needed
        @param      intermediate        returns a dictionary of intermediate
                                        variables instead of the results only
        @param      verbose             display information while predicting
        @param      node_time           measure time of each node
        @param      fLOG                logging function if *verbose > 0*
        @return                         outputs as dictionary
                                        and a second dictionary of the time spent
                                        in each node if *node_time* is True

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
                from mlprodict.onnxrt import OnnxInference, to_onnx

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

        The function returns all intermediate outputs
        if *intermediate* is True. In case of runtime
        *onnxruntime1*, if intermediate is True,
        the first class builds all :epkg:`ONNX` cut out
        to keep the one output and converted into
        *OnnxInference*.
        """
        return self._run(inputs, clean_right_away=False, intermediate=intermediate,
                         verbose=verbose, node_time=node_time, fLOG=fLOG)

    def _run_sequence_runtime(self, inputs, clean_right_away=False,
                              intermediate=False, verbose=0, node_time=False,
                              fLOG=None):
        if clean_right_away:
            raise NotImplementedError("clean_right_away=true not implemented.")
        values = OrderedDict(inputs)
        for k, v in self.inits_.items():
            values[k] = v['value']

        mtime = []
        if verbose == 0 or fLOG is None:
            if node_time:
                for i, node in enumerate(self.sequence_):
                    t = perf_counter()
                    node.run(values)
                    t2 = perf_counter()
                    mtime.append(dict(i=i, name=node.onnx_node.name,
                                      op_type=node.onnx_node.op_type,
                                      time=t2 - t))
            else:
                for node in self.sequence_:
                    node.run(values)
        else:
            def dispsimple(arr):
                if hasattr(arr, 'shape'):
                    if len(arr.shape) == 1:
                        threshold = 8
                    else:
                        threshold = min(
                            50, min(50 // arr.shape[1], 8) * arr.shape[1])
                    fLOG(numpy.array2string(arr, max_line_width=120,
                                            suppress_small=True,
                                            threshold=threshold))
                else:
                    s = str(arr)
                    if len(s) > 50:
                        s = s[:50] + "..."
                    fLOG(s)

            if verbose >= 2:
                for k in sorted(values):
                    fLOG("-k='{}' shape={} dtype={} min={} max={}".format(
                        k, values[k].shape, values[k].dtype,
                        numpy.min(values[k]), numpy.max(values[k])))
            keys = set(values)
            if verbose >= 1:
                fLOG("-- OnnxInference: run {} nodes".format(len(self.sequence_)))
            for i, node in enumerate(self.sequence_):
                if verbose >= 1:
                    fLOG(node)
                if node_time:
                    t = perf_counter()
                    node.run(values)
                    t2 = perf_counter()
                    mtime.append(dict(i=i, name=node.onnx_node.name,
                                      op_type=node.onnx_node.op_type,
                                      time=t2 - t))
                else:
                    node.run(values)
                for k in sorted(values):
                    if k not in keys:
                        if isinstance(values[k], numpy.ndarray):
                            fLOG("+k='{}': {} (dtype={} min={} max={})".format(
                                k, values[k].shape, values[k].dtype,
                                numpy.min(values[k]), numpy.max(values[k])))
                            if verbose >= 3:
                                dispsimple(values[k])
                        else:
                            fLOG("+k='{}': {}".format(
                                k, type(values[k])))
                            if verbose >= 3:
                                dispsimple(values[k])
                keys = set(values)

        if intermediate:
            return (values, mtime) if node_time else values
        else:
            try:
                res = {k: values[k] for k in self.outputs_}
            except KeyError as e:
                raise RuntimeError("Unable to find one output in [{}].".format(
                    ", ".join(sorted(values)))) from e
            return (res, mtime) if node_time else res

    def build_intermediate(self):
        """
        Builds every possible :epkg:`ONNX` file
        which computes one specific intermediate output
        from the inputs.

        @return         :epkg:`*py:collections:OrderedDict`
        """
        ord = OrderedDict()
        for output in enumerate_model_node_outputs(self.obj):
            subonx = select_model_inputs_outputs(self.obj, output)
            ord[output] = OnnxInference(subonx, runtime=self.runtime,
                                        skip_run=self.skip_run)
        return ord

    def _run_whole_runtime(self, inputs, clean_right_away=False,
                           intermediate=False, verbose=0, node_time=False,
                           fLOG=None):
        # node_time is unused
        if clean_right_away:
            raise RuntimeError(
                "clean_right_away=true does not work with this runtime.")
        if intermediate:
            if hasattr(self, "intermediate_onnx_inference_"):
                inter_run = self.intermediate_onnx_inference_  # pylint: disable=E0203
            else:
                if verbose > 0:
                    fLOG("-- OnnxInference: build intermediate")
                inter_run = self.build_intermediate()
                self.intermediate_onnx_inference_ = inter_run
                graph = self.to_sequence()
                self.inits_ = graph['inits']

            if verbose >= 1:
                fLOG("-- OnnxInference: run {} nodes".format(
                    len(self.intermediate_onnx_inference_)))
            values = OrderedDict(inputs)
            for k, v in self.inits_.items():
                values[k] = v['value']
            if verbose >= 2:
                for k in sorted(values):
                    fLOG("-k='{}' shape={} dtype={}".format(
                        k, values[k].shape, values[k].dtype))
            for node, oinf in self.intermediate_onnx_inference_.items():
                if verbose >= 1:
                    fLOG(node)
                    if verbose >= 2:
                        fLOG(oinf.obj)
                output = oinf.run(inputs)[node]
                values[node] = output
                if verbose >= 1:
                    if isinstance(output, numpy.ndarray):
                        fLOG("+k='{}': {} (dtype={})".format(
                            k, output.shape, output.dtype))
                    else:
                        fLOG("+k='{}': {}".format(
                            k, type(output)))
            return values
        else:
            if verbose != 0:
                warnings.warn(
                    "verbose option not implemented if runtime is 'onnxruntime1'")
            res = self._whole.run(inputs)
            return {k: v for k, v in zip(self.outputs_, res)}

    def __getitem__(self, item):
        """
        Returns the ONNX verions of a node.
        """
        if isinstance(item, tuple):
            node_name, att_name = item
        else:
            node_name = item
            att_name = None

        node_ = None
        for node in self.obj.graph.node:
            if node.name == node_name:
                node_ = node
                break

        if node_ is None:
            raise IndexError("Unable to node name '{}'.".format(node_name))

        if att_name is None:
            return node_

        for att in node_.attribute:
            if att.name == att_name:
                return att

        raise IndexError("Unable to find attribute '{}' from node '{}'.".format(
            att_name, node_name))

    def switch_initializers_dtype(self, model=None,
                                  dtype_in=numpy.float32,
                                  dtype_out=numpy.float64):
        """
        Switches all initializers to ``numpy.float64``. If *model*
        is None, a simple cast is done. Otherwise, the function assumes
        the model is a :epkg:`scikit-learn` pipeline.
        This only works if the runtime is ``'python'``.

        @param      model       :epkg:`scikit-learn` model or None
        @param      dtype_in    previous type
        @param      dtype_out   next type
        @return                 done operations
        """
        if self.runtime != 'python':
            raise RuntimeError("Initializers can be casted only if the "
                               "runtime is 'python' not '{}'.".format(self.runtime))

        # first pass: simple cast
        done = []
        initializer = self.inits_
        for k, v in initializer.items():
            if isinstance(v['value'], numpy.ndarray):
                if v['value'].dtype == dtype_in:
                    v['value'] = v['value'].astype(dtype_out)
                    done.append(("pass1", "+", "init", k, v['value']))
                else:
                    done.append(("pass1", "-", "init", k, v['value']))
        for k, v in self.graph_['nodes'].items():
            res = v.switch_initializers_dtype(dtype_in=dtype_in,
                                              dtype_out=dtype_out)
            for r in res:
                done.append(("pass1", "node", k) + r)
        for k, v in self.graph_['intermediate'].items():
            if v is None:
                continue
            res = v.switch_initializers_dtype(dtype_in=dtype_in,
                                              dtype_out=dtype_out)
            for r in res:
                done.append(("pass1", "sub", k) + r)

        if model is not None:
            # Second pass, we compare all arrays from the model
            # to the arrays in the converted models.
            def dist(a):
                cast = a.astype(dtype_in).astype(dtype_out)
                d = pairwise_array_distances([cast], [a])[0, 0]
                return d

            done_ = [(c, c[-1]) for c in done]
            moda_ = [(a, a[-2][-1]) for a in enumerate_fitted_arrays(model)
                     if dist(a[-2][-1]) > 0]
            aconv = [_[-1] for _ in done_]
            amoda = [_[-1] for _ in moda_]
            distances = pairwise_array_distances(aconv, amoda)

            for i in range(distances.shape[0]):
                j = numpy.argmin(distances[i])
                d = distances[i, j]
                if d < 0.1:
                    numpy.copyto(aconv[i], amoda[j])
                    done.append(("pass2", d) + done_[i][0])

        return done
