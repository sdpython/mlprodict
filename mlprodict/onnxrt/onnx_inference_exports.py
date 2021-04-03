"""
@file
@brief Extensions to class @see cl OnnxInference.
"""
import os
import json
import re
from io import BytesIO
import pickle
import textwrap
from onnx import numpy_helper
from ..tools.onnx2py_helper import _var_as_dict, _type_to_string


class OnnxInferenceExport:
    """
    Implements methods to export a instance of
    @see cl OnnxInference into :epkg:`json` or :epkg:`dot`.
    """

    def __init__(self, oinf):
        """
        @param      oinf    @see cl OnnxInference
        """
        self.oinf = oinf

    def to_dot(self, recursive=False, prefix='',  # pylint: disable=R0914
               add_rt_shapes=False, use_onnx=False, **params):
        """
        Produces a :epkg:`DOT` language string for the graph.

        :param params: additional params to draw the graph
        :param recursive: also show subgraphs inside operator like
            @see cl Scan
        :param prefix: prefix for every node name
        :param add_rt_shapes: adds shapes infered from the python runtime
        :param use_onnx: use :epkg:`onnx` dot format instead of this one
        :return: string

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
                :warningout: DeprecationWarning

                import numpy
                from skl2onnx.algebra.onnx_ops import OnnxLinearRegressor
                from skl2onnx.common.data_types import FloatTensorType
                from mlprodict.onnxrt import OnnxInference

                pars = dict(coefficients=numpy.array([1., 2.]),
                            intercepts=numpy.array([1.]),
                            post_transform='NONE')
                onx = OnnxLinearRegressor('X', output_names=['Y'], **pars)
                model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                        outputs=[('Y', FloatTensorType([1]))],
                                        target_opset=12)
                oinf = OnnxInference(model_def)
                print(oinf.to_dot())

            See an example of representation in notebook
            :ref:`onnxvisualizationrst`.
        """
        clean_label_reg1 = re.compile("\\\\x\\{[0-9A-F]{1,6}\\}")
        clean_label_reg2 = re.compile("\\\\p\\{[0-9P]{1,6}\\}")

        def dot_name(text):
            return text.replace("/", "_").replace(":", "__")

        def dot_label(text):
            for reg in [clean_label_reg1, clean_label_reg2]:
                fall = reg.findall(text)
                for f in fall:
                    text = text.replace(f, "_")  # pragma: no cover
            return text

        options = {
            'orientation': 'portrait',
            'ranksep': '0.25',
            'nodesep': '0.05',
            'width': '0.5',
            'height': '0.1',
        }
        options.update(params)

        if use_onnx:
            from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

            pydot_graph = GetPydotGraph(
                self.oinf.obj.graph, name=self.oinf.obj.graph.name,
                rankdir=params.get('rankdir', "TB"),
                node_producer=GetOpNodeProducer(
                    "docstring", fillcolor="orange", style="filled",
                    shape="box"))
            return pydot_graph.to_string()

        inter_vars = {}
        exp = ["digraph{"]
        for opt in {'orientation', 'pad', 'nodesep', 'ranksep'}:
            if opt in options:
                exp.append("  {}={};".format(opt, options[opt]))
        fontsize = 10

        shapes = {}
        if add_rt_shapes:
            if not hasattr(self.oinf, 'shapes_'):
                raise RuntimeError(  # pragma: no cover
                    "No information on shapes, check the runtime '{}'.".format(self.oinf.runtime))
            for name, shape in self.oinf.shapes_.items():
                va = shape.evaluate().to_string()
                shapes[name] = va
                if name in self.oinf.inplaces_:
                    shapes[name] += "\\ninplace"

        # inputs
        exp.append("")
        for obj in self.oinf.obj.graph.input:
            dobj = _var_as_dict(obj)
            sh = shapes.get(dobj['name'], '')
            if sh:
                sh = "\\nshape={}".format(sh)
            exp.append(
                '  {3}{0} [shape=box color=red label="{0}\\n{1}{4}" fontsize={2}];'.format(
                    dot_name(dobj['name']), _type_to_string(dobj['type']),
                    fontsize, prefix, dot_label(sh)))
            inter_vars[obj.name] = obj

        # outputs
        exp.append("")
        for obj in self.oinf.obj.graph.output:
            dobj = _var_as_dict(obj)
            sh = shapes.get(dobj['name'], '')
            if sh:
                sh = "\\nshape={}".format(sh)
            exp.append(
                '  {3}{0} [shape=box color=green label="{0}\\n{1}{4}" fontsize={2}];'.format(
                    dot_name(dobj['name']), _type_to_string(dobj['type']),
                    fontsize, prefix, dot_label(sh)))
            inter_vars[obj.name] = obj

        # initializer
        exp.append("")
        for obj in self.oinf.obj.graph.initializer:
            dobj = _var_as_dict(obj)
            val = dobj['value']
            flat = val.flatten()
            if flat.shape[0] < 9:
                st = str(val)
            else:
                st = str(val)
                if len(st) > 50:
                    st = st[:50] + '...'
            st = st.replace('\n', '\\n')
            kind = ""
            exp.append(
                '  {6}{0} [shape=box label="{0}\\n{4}{1}({2})\\n{3}" fontsize={5}];'.format(
                    dot_name(dobj['name']), dobj['value'].dtype,
                    dobj['value'].shape, dot_label(st), kind, fontsize, prefix))
            inter_vars[obj.name] = obj

        # nodes
        fill_names = {}
        for node in self.oinf.obj.graph.node:
            exp.append("")
            for out in node.output:
                if out not in inter_vars:
                    inter_vars[out] = out
                    sh = shapes.get(out, '')
                    if sh:
                        sh = "\\nshape={}".format(sh)
                    exp.append(
                        '  {2}{0} [shape=box label="{0}{3}" fontsize={1}];'.format(
                            dot_name(out), fontsize, dot_name(prefix), dot_label(sh)))

            dobj = _var_as_dict(node)
            if dobj['name'].strip() == '':  # pragma: no cover
                name = node.op_type
                iname = 1
                while name in fill_names:
                    name = "%s%d" % (name, iname)
                    iname += 1
                dobj['name'] = name
                node.name = name

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

            if recursive and node.op_type in {'Scan', 'Loop', 'If'}:
                fields = (['then_branch', 'else_branch']
                          if node.op_type == 'If' else ['body'])
                for field in fields:
                    if field not in dobj['atts']:
                        continue  # pragma: no cover

                    # creates the subgraph
                    body = dobj['atts'][field]['value']
                    oinf = self.oinf.__class__(
                        body, runtime=self.oinf.runtime, skip_run=self.oinf.skip_run)
                    subprefix = prefix + "B_"
                    subdot = oinf.to_dot(recursive=recursive, prefix=subprefix,
                                         add_rt_shapes=add_rt_shapes)
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
                        dobj['op_type'], dot_name(dobj['name']), satts))
                    exp.append('    fontsize={0};'.format(fontsize))
                    exp.append('    color=black;')
                    exp.append(
                        '\n'.join(map(lambda s: '  ' + s, subgraph.split('\n'))))

                    for inp1, inp2 in zip(node.input, body.input):
                        exp.append(
                            "  {0}{1} -> {2}{3};".format(
                                dot_name(prefix), dot_name(inp1),
                                dot_name(subprefix), dot_name(inp2.name)))
                    for out1, out2 in zip(body.output, node.output):
                        exp.append(
                            "  {0}{1} -> {2}{3};".format(
                                dot_name(subprefix), dot_name(out1.name),
                                dot_name(prefix), dot_name(out2)))

            else:
                exp.append('  {4}{1} [shape=box style="filled,rounded" color=orange label="{0}\\n({1}){2}" fontsize={3}];'.format(
                    dobj['op_type'], dot_name(dobj['name']), satts, fontsize,
                    dot_name(prefix)))

                for inp in node.input:
                    exp.append(
                        "  {0}{1} -> {0}{2};".format(
                            dot_name(prefix), dot_name(inp), dot_name(node.name)))
                for out in node.output:
                    exp.append(
                        "  {0}{1} -> {0}{2};".format(
                            dot_name(prefix), dot_name(node.name), dot_name(out)))

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
                :warningout: DeprecationWarning

                import numpy
                from skl2onnx.algebra.onnx_ops import OnnxLinearRegressor
                from skl2onnx.common.data_types import FloatTensorType
                from mlprodict.onnxrt import OnnxInference

                pars = dict(coefficients=numpy.array([1., 2.]),
                            intercepts=numpy.array([1.]),
                            post_transform='NONE')
                onx = OnnxLinearRegressor('X', output_names=['Y'], **pars)
                model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                        outputs=[('Y', FloatTensorType([1]))],
                                        target_opset=12)
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
                        raise RuntimeError(  # pragma: no cover
                            "Unable to interpret line '{}'.".format(line))

                    if spl[0].strip() in ('type', ):
                        st = spl[1].strip()
                        if st in {'INT', 'INTS', 'FLOAT', 'FLOATS',
                                  'STRING', 'STRINGS', 'TENSOR'}:
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
                    raise RuntimeError(  # pragma: no cover
                        "Unable to interpret line '{}'.".format(line))
            rows[-1] = rows[-1].rstrip(',')
            rows.append("}")
            js = "\n".join(rows)

            try:
                content = json.loads(js)
            except json.decoder.JSONDecodeError as e:  # pragma: no cover
                js2 = "\n".join("%04d %s" % (i + 1, line)
                                for i, line in enumerate(js.split("\n")))
                raise RuntimeError(
                    "Unable to parse JSON\n{}".format(js2)) from e
            return content

        # meta data
        final_obj = {}
        for k in {'ir_version', 'producer_name', 'producer_version',
                  'domain', 'model_version', 'doc_string'}:
            if hasattr(self.oinf.obj, k):
                final_obj[k] = getattr(self.oinf.obj, k)

        # inputs
        inputs = []
        for obj in self.oinf.obj.graph.input:
            st = _to_json(obj)
            inputs.append(st)
        final_obj['inputs'] = inputs

        # outputs
        outputs = []
        for obj in self.oinf.obj.graph.output:
            st = _to_json(obj)
            outputs.append(st)
        final_obj['outputs'] = outputs

        # init
        inits = {}
        for obj in self.oinf.obj.graph.initializer:
            value = numpy_helper.to_array(obj).tolist()
            inits[obj.name] = value
        final_obj['initializers'] = inits

        # nodes
        nodes = []
        for obj in self.oinf.obj.graph.node:
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

    def to_python(self, prefix="onnx_pyrt_", dest=None, inline=True):
        """
        Converts the ONNX runtime into independant python code.
        The function creates multiple files starting with
        *prefix* and saved to folder *dest*.

        @param  prefix      file prefix
        @param  dest        destination folder
        @param  inline      constant matrices are put in the python file itself
                            as byte arrays
        @return             file dictionary

        The function does not work if the chosen runtime
        is not *python*.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning

            import numpy
            from skl2onnx.algebra.onnx_ops import OnnxAdd
            from mlprodict.onnxrt import OnnxInference

            idi = numpy.identity(2)
            onx = OnnxAdd('X', idi, output_names=['Y'],
                          op_version=12)
            model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                    target_opset=12)
            X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
            oinf = OnnxInference(model_def, runtime='python')
            res = oinf.to_python()
            print(res['onnx_pyrt_main.py'])
        """

        def clean_args(args):
            new_args = []
            for v in args:
                # remove python keywords
                if v.startswith('min='):
                    av = 'min_=' + v[4:]
                elif v.startswith('max='):
                    av = 'max_=' + v[4:]
                else:
                    av = v
                new_args.append(av)
            return new_args

        if self.oinf.runtime != 'python':
            raise ValueError(
                "The runtime must be 'python' not '{}'.".format(
                    self.oinf.runtime))

        # metadata
        obj = {}
        for k in {'ir_version', 'producer_name', 'producer_version',
                  'domain', 'model_version', 'doc_string'}:
            if hasattr(self.oinf.obj, k):
                obj[k] = getattr(self.oinf.obj, k)
        code_begin = ["# coding: utf-8",
                      "'''",
                      "Python code equivalent to an ONNX graph.",
                      "It was was generated by module *mlprodict*.",
                      "'''"]
        code_imports = ["from io import BytesIO",
                        "import pickle",
                        "from numpy import array, float32, ndarray"]
        code_lines = ["class OnnxPythonInference:", "",
                      "    def __init__(self):",
                      "        self._load_inits()", "",
                      "    @property",
                      "    def metadata(self):",
                      "        return %r" % obj, ""]

        # inputs
        inputs = [obj.name for obj in self.oinf.obj.graph.input]
        code_lines.extend([
            "    @property", "    def inputs(self):",
            "        return %r" % inputs,
            ""
        ])

        # outputs
        outputs = [obj.name for obj in self.oinf.obj.graph.output]
        code_lines.extend([
            "    @property", "    def outputs(self):",
            "        return %r" % outputs,
            ""
        ])

        # init
        code_lines.extend(["    def _load_inits(self):",
                           "        self._inits = {}"])
        file_data = {}
        for obj in self.oinf.obj.graph.initializer:
            value = numpy_helper.to_array(obj)
            bt = BytesIO()
            pickle.dump(value, bt)
            name = '{1}{0}.pkl'.format(obj.name, prefix)
            if inline:
                code_lines.extend([
                    "        iocst = %r" % bt.getvalue(),
                    "        self._inits['{0}'] = pickle.loads(iocst)".format(
                        obj.name)
                ])
            else:
                file_data[name] = bt.getvalue()
                code_lines.append(
                    "        self._inits['{0}'] = pickle.loads('{1}')".format(
                        obj.name, name))
        code_lines.append('')

        # inputs, outputs
        inputs = self.oinf.input_names

        # nodes
        code_lines.extend(['    def run(self, %s):' % ', '.join(inputs)])
        ops = {}
        code_lines.append('        # constant')
        for obj in self.oinf.obj.graph.initializer:
            code_lines.append(
                "        {0} = self._inits['{0}']".format(obj.name))
        code_lines.append('')
        code_lines.append('        # graph code')
        for node in self.oinf.sequence_:
            fct = 'pyrt_' + node.name
            if fct not in ops:
                ops[fct] = node
            args = []
            args.extend(node.inputs)
            margs = node.modified_args
            if margs is not None:
                args.extend(clean_args(margs))
            code_lines.append("        {0} = {1}({2})".format(
                ', '.join(node.outputs), fct, ', '.join(args)))
        code_lines.append('')
        code_lines.append('        # return')
        code_lines.append('        return %s' % ', '.join(outputs))
        code_lines.append('')

        # operator code
        code_nodes = []
        for name, op in ops.items():
            inputs_args = clean_args(op.inputs_args)

            code_nodes.append('def {0}({1}):'.format(
                name, ', '.join(inputs_args)))
            imps, code = op.to_python(op.python_inputs)
            if imps is not None:
                if not isinstance(imps, list):
                    imps = [imps]
                code_imports.extend(imps)
            code_nodes.append(textwrap.indent(code, '    '))
            code_nodes.extend(['', ''])

        # end
        code_imports = list(sorted(set(code_imports)))
        code_imports.extend(['', ''])
        file_data[prefix + 'main.py'] = "\n".join(
            code_begin + code_imports + code_nodes + code_lines)

        # saves as files
        if dest is not None:
            for k, v in file_data.items():
                ext = os.path.splitext(k)[-1]
                kf = os.path.join(dest, k)
                if ext == '.py':
                    with open(kf, "w", encoding="utf-8") as f:
                        f.write(v)
                elif ext == '.pkl':  # pragma: no cover
                    with open(kf, "wb") as f:
                        f.write(v)
                else:
                    raise NotImplementedError(  # pragma: no cover
                        "Unknown extension for file '{}'.".format(k))
        return file_data
