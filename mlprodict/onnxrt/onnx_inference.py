# pylint: disable=C0302
"""
@file
@brief Implements a class able to compute the predictions
from on an :epkg:`ONNX` model.
"""
from collections import OrderedDict
from io import BytesIO
from time import perf_counter
import warnings
import textwrap
import numpy
from scipy.sparse import coo_matrix
from onnx import load, load_model, checker, shape_inference
from onnx import onnx_pb as onnx_proto
from onnx.helper import make_model
from ..tools.code_helper import make_callable
from ..onnx_tools.onnx2py_helper import (
    _var_as_dict, numpy_min, numpy_max, guess_numpy_type_from_string)
from ..onnx_tools.onnx_manipulations import (
    select_model_inputs_outputs, enumerate_model_node_outputs)
from ..onnx_tools.optim import onnx_remove_node_unused
from .onnx_inference_node import OnnxInferenceNode
from .onnx_inference_exports import OnnxInferenceExport
from .shape_object import ShapeObject


class OnnxInference:
    """
    Loads an :epkg:`ONNX` file or object or stream.
    Computes the output of the :epkg:`ONNX` graph.
    Several runtimes are available.

    * ``'python'``: the runtime implements every onnx operator
      needed to run a :epkg:`scikit-learn` model by using :epkg:`numpy`
      or C++ code.
    * ``'python_compiled'``: it is the same runtime than the previous
      one except every operator is called from a compiled function
      (@see me _build_compile_run) instead for a method going through
      the list of operator
    * ``'onnxruntime1'``: uses :epkg:`onnxruntime`
    * ``'onnxruntime2'``: this mode is mostly used to debug as
      python handles calling every operator but :epkg:`onnxruntime`
      is called for every of them, this process may fail due to
      wrong inference type specially of the graph includes
      custom nodes, in that case, it is better to compute the output
      of intermediates nodes. It is much slower as fo every output, every
      node is computed but more robust.

    :param onnx_or_bytes_or_stream: :epkg:`onnx` object,
        bytes, or filename or stream
    :param runtime: runtime options
    :param skip_run: do not build the runtime
    :param inplace: use inplace computation as much as possible
    :param input_inplace: the computation is allowed
        to overwrite the input, see :meth:`_guess_inplace
        <mlprodict.onnxrt.onnx_inference.OnnxInference._guess_inplace>`
    :param ir_version: if not None, overwrite the default version
    :param target_opset: used to overwrite *target_opset*
    :param runtime_options: specific options for the runtime

    Among the possible runtime_options, there are:
    * *enable_profiling*: enables profiling for :epkg:`onnxruntime`
    * *session_options*: an instance of *SessionOptions* from
        :epkg:`onnxruntime`
    * *ir_version*: change ir_version
    """

    def __init__(self, onnx_or_bytes_or_stream, runtime=None,
                 skip_run=False, inplace=True,
                 input_inplace=False, ir_version=None,
                 target_opset=None, runtime_options=None,
                 session_options=None):
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
            raise TypeError("Unable to handle type {}.".format(  # pragma: no cover
                type(onnx_or_bytes_or_stream)))
        if ir_version is not None:
            self.obj.ir_version = ir_version
        self.runtime = runtime
        self.skip_run = skip_run
        self.input_inplace = input_inplace
        self.inplace = inplace
        self.force_target_opset = target_opset
        self.runtime_options = runtime_options
        self._init()

    def __getstate__(self):
        """
        To pickle the object.
        """
        return {'onnx': self.obj.SerializeToString(),
                'runtime': self.runtime,
                'runtime_options': self.runtime_options,
                'skip_run': self.skip_run,
                'input_inplace': self.input_inplace,
                'inplace': self.inplace,
                'force_target_opset': self.force_target_opset}

    def __setstate__(self, state):
        """
        To unpickle the object.
        """
        onx = state['onnx']
        self.obj = load_model(BytesIO(onx))
        self.runtime = state['runtime']
        self.runtime_options = state['runtime_options']
        self.skip_run = state['skip_run']
        self.input_inplace = state['input_inplace']
        self.inplace = state['inplace']
        self.force_target_opset = state['force_target_opset']
        self._init()

    def _init(self):
        """
        Prepares the instance to deliver predictions.
        """
        self.graph_ = self.to_sequence()
        if len(self.graph_['sequence']) == 0:
            raise RuntimeError(  # pragma: no cover
                "No runnable nodes was found in the ONNX graph.")
        self.outputs_ = self.graph_['outputs']
        self.inputs_ = self.graph_['inputs']

        for ino in [self.obj.graph.input, self.obj.graph.output]:
            for xy in ino:
                shape = xy.type.tensor_type.shape
                for d in shape.dim:
                    if d.dim_value == 0 and "0" in str(d):
                        # d.dim_value returns 0 whether is is 0 or empty.
                        raise RuntimeError(
                            "Wrong ONNX file, one input or output has an empty shape: "
                            "{}.".format(xy))

        self.target_opset_ = self.graph_['targets']
        if self.force_target_opset is not None:
            if isinstance(self.force_target_opset, dict):
                self.target_opset_ = self.force_target_opset  # pragma: no cover
            else:
                self.target_opset_ = {'': self.force_target_opset}
        self.ir_version_ = self.graph_['ir_version']
        if not self.skip_run:
            if self.runtime == 'onnxruntime1':
                # Loads the onnx with onnxruntime as a single file.
                del self.graph_
                from .ops_whole.session import OnnxWholeSession
                self._whole = OnnxWholeSession(
                    self.obj, self.runtime, self.runtime_options)
                self._run = self._run_whole_runtime
            else:
                self.sequence_ = self.graph_['sequence']
                self.inits_ = self.graph_['inits']
                dtype = self._guess_input_dtype()
                variables = self.inits_.copy()
                for node in self.sequence_:
                    domain = node.onnx_node.domain
                    target_opset = self.target_opset_.get(domain, None)
                    if self.runtime in ('onnxruntime2', 'empty'):
                        node.setup_runtime(self.runtime, variables, self.__class__,
                                           target_opset=target_opset, dtype=dtype,
                                           domain=domain, ir_version=self.ir_version_,
                                           runtime_options=self.runtime_options)
                    else:
                        node.setup_runtime(self.runtime, variables, self.__class__,
                                           target_opset=target_opset, domain=domain,
                                           ir_version=self.ir_version_,
                                           runtime_options=self.runtime_options)
                    if hasattr(node, 'ops_') and hasattr(node.ops_, 'typed_outputs_'):
                        for k, v in node.ops_.typed_outputs_:
                            variables[k] = v
                self._run = self._run_sequence_runtime
        if not self.skip_run and self.runtime in ('python', None):
            self.shapes_ = self._set_shape_inference_runtime()
            if self.inplace:
                self.inplaces_ = self._guess_inplace(self.input_inplace)
        self.exporters_ = OnnxInferenceExport(self)
        self.to_json = self.exporters_.to_json
        self.to_dot = self.exporters_.to_dot
        self.to_python = self.exporters_.to_python

        if self.runtime in ('python_compiled', 'python_compiled_debug'):
            # switch the inference method to the compiled one
            _, fct, code = self._build_compile_run('debug' in self.runtime)
            setattr(self, '_run_compiled', fct)
            setattr(self, '_run_compiled_code', code)
            self._run = self._run_sequence_runtime_compiled

    def _run_sequence_runtime_compiled(
            self, inputs, clean_right_away=False, intermediate=False,
            verbose=0, node_time=False, fLOG=None):
        """
        Executes a compiled version of @see me _run_sequence_runtime,
        compiled with method @see me _build_compile_run.
        Every parameter with a default value is ignored.
        Switch to ``runtime='python'`` to enable those.
        """
        return self._run_compiled(inputs)  # pylint: disable=E1101

    def _guess_input_dtype(self):
        for _, v in self.graph_['inputs'].items():
            if 'type' not in v:
                continue  # pragma: no cover
            t = v['type']
            if 'elem' not in t:
                continue
            if t['elem'] == 'double':
                return numpy.float64
        return numpy.float32

    def __str__(self):
        """
        usual
        """
        rows = ['OnnxInference(...)']
        if hasattr(self, '_run_compiled_code'):
            rows.append(
                textwrap.indent(
                    self._run_compiled_code, '    '))  # pylint: disable=E1101
        else:
            rows.append(textwrap.indent(str(self.obj), '    '))
        return "\n".join(rows)

    def __repr__(self):
        """
        usual
        """
        return "OnnxInference(...)"  # pragma: no cover

    def check_model(self):
        """
        Checks the model follow :epkg:`ONNX` conventions.
        """
        checker.check_model(self.obj)

    def shape_inference(self):
        """
        Infers the shape of the outputs
        with :epkg:`onnx` package.

        @return     A new :epkg:`ONNX` graph which defined outputs.
        """
        return shape_inference.infer_shapes(self.obj)

    @property
    def input_names(self):
        """
        Returns the names of all inputs.
        It does not include the optional inputs.

        .. versionchanged:: 0.6
            The list does not include optional inputs anymore.
        """
        inits = set(_.name for _ in self.obj.graph.initializer)
        return [_.name for _ in self.obj.graph.input if _.name not in inits]

    @property
    def input_names_shapes(self):
        """
        Returns the names and shapes of all inputs.
        This method assumes all inputs are tensors.
        It does not include the optional inputs.

        .. versionchanged:: 0.6
            The list does not include optional inputs anymore.
        """
        names = set(self.input_names)
        return [(_.name, _var_as_dict(_)['type']['shape'])
                for _ in self.obj.graph.input if _.name in names]

    @property
    def input_names_shapes_types(self):
        """
        Returns the names, shapes, types of all inputs.
        This method assumes all inputs are tensors.
        It does not include the optional inputs.

        .. versionchanged:: 0.6
            The list does not include optional inputs anymore.
        """
        names = set(self.input_names)
        return [(_.name, _var_as_dict(_)['type']['shape'],
                 'tensor(%s)' % _var_as_dict(_)['type']['elem'])
                for _ in self.obj.graph.input if _.name in names]

    @property
    def output_names(self):
        """
        Returns the names of all outputs.
        """
        return [_.name for _ in self.obj.graph.output]

    @property
    def output_names_shapes(self):
        """
        Returns the names and shapes of all outputs.
        This method assumes all inputs are tensors.
        """
        return [(_.name, _var_as_dict(_)['type'].get('shape', None))
                for _ in self.obj.graph.output]

    def global_index(self, name):
        """
        Maps every name to one integer to avoid using dictionaries
        when running the predictions.

        @param      name        outputs name
        @return                 integer
        """
        if not hasattr(self, '_global_index'):
            self._global_index = {}
        if name in self._global_index:
            return self._global_index[name]
        self._global_index[name] = len(self._global_index)
        return self._global_index[name]

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
                :warningout: DeprecationWarning

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
                                        outputs=[('Y', FloatTensorType([1]))],
                                        target_opset=12)
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
            self.global_index(obj.name)

        # outputs
        for obj in self.obj.graph.output:
            if hasattr(obj, 'type') and str(obj.type) != '':
                outputs[obj.name] = _var_as_dict(obj)
            else:
                outputs[obj.name] = {'name': obj.name}
            self.global_index(obj.name)

        # initializer
        for obj in self.obj.graph.initializer:
            init_obj = _var_as_dict(obj)
            if init_obj is None:
                raise RuntimeError(  # pragma: no cover
                    "Unable to convert an initializer\n{}".format(obj))
            inits[obj.name] = init_obj
            self.global_index(obj.name)
            if 'value' not in inits[obj.name]:
                raise RuntimeError(  # pragma: no cover
                    "One initializer has no value: '{}'\n{}\n{}".format(
                        obj.name, inits[obj.name], obj))

        # nodes
        for node in self.obj.graph.node:
            dobj = _var_as_dict(node)
            if dobj is None:
                raise RuntimeError(  # pragma: no cover
                    "Unable to convert a node\n{}".format(node))
            if 'atts' in dobj:
                atts = dobj['atts']
                for k, v in atts.items():
                    if not isinstance(v, dict) or 'value' not in v:
                        raise RuntimeError(  # pragma: no cover
                            "A parameter has no (sparse) value '{}' "
                            "for node '{}'\nv={}\ndobj=[{}]".format(
                                k, node.name, v, node))
            if node.name in nodes:  # pragma: no cover
                i = 2
                while True:
                    new_name = "%s_n%i" % (node.name, i)
                    if new_name not in nodes:
                        break
                    i += 1
            else:
                new_name = node.name
            nodes[new_name] = OnnxInferenceNode(node, dobj, self.global_index)

        # names
        names = {}
        for k, v in inits.items():
            if (k, 0) in names:
                raise RuntimeError(  # pragma: no cover
                    "Initializer '{}' already exists (tag='{}').".format(
                        k, names[k, 0][0]))
            names[k, 0] = ('C', v)
        for k, v in variables.items():
            if (k, 0) in names:
                if k in inits:
                    # Kind of default value for an input
                    continue
                raise RuntimeError(  # pragma: no cover
                    "Variable '{}' already exists (tag='{}').".format(
                        k, names[k, 0][0]))
            names[k, 0] = ('I', v)
        for k, v in outputs.items():
            if (k, 0) in names and self.runtime != 'empty':
                raise RuntimeError(  # pragma: no cover
                    "Output '{}' already exists (tag='{}').".format(
                        k, names[k, 0][0]))
            names[k, 0] = ('O', v)
        for k, v in nodes.items():
            if (k, 1) in names:
                raise RuntimeError(  # pragma: no cover
                    "Node '{}' already exists (tag='{}').".format(
                        k, names[k, 0][0]))
            names[k, 1] = ('N', v)

        # ordering
        order = {}
        modif = 1
        intermediate = {}
        while modif > 0:
            modif = 0
            for (k, _), v in names.items():
                if (k, 1) in order:
                    # The operator node is already processed.
                    continue
                if v[0] in {'I', 'C'}:
                    if (k, 0) not in order:
                        order[k, 0] = len(order)  # A data node.
                        modif += 1
                    continue
                if v[0] == 'O':
                    continue
                if all((inp, 0) in order for inp in v[1].inputs):
                    # If all inputs are available,
                    # We tell the operator node is processed.
                    order[k, 1] = len(order)
                    modif += 1
                    for o in v[1].outputs:
                        if (o, 0) in order:
                            raise RuntimeError(  # pragma: no cover
                                "Two nodes share the same output '{}' or an operator and an output "
                                "share the same name. "
                                "(node: {}).".format(o, v[1]))
                        # We add a data node.
                        order[o, 0] = len(order)
                        intermediate[o] = None
                        modif += 1

        # compute
        rev = [(v, k[0], k[1]) for k, v in order.items()]
        rev.sort()
        sequence = []
        for _, name, node_kind in rev:
            if name not in nodes:
                continue
            if node_kind == 0:
                # It is an output which shares the same name
                # as a node.
                continue
            node = nodes[name]
            node.set_order(len(sequence))
            sequence.append(node)

        if len(sequence) == 0:
            raise RuntimeError(  # pragma: no cover
                "No runnable nodes was found in the ONNX graph"
                "\n--rev--\n{}"
                "\n--order--\n{}"
                "\n--nodes--\n{}"
                "\n---".format(
                    "\n".join([str(_) for _ in names.items()]),
                    "\n".join([str(_) for _ in order.items()]),
                    "\n".join([str(_) for _ in nodes.items()])))

        # defines where an intermediare output is not needed
        last_used = {}
        for node in sequence:
            for inp in node.inputs:
                last_used[inp] = node.order
        for k, ord in last_used.items():
            sequence[ord].add_variable_to_clean(k)

        return dict(inits=inits, inputs=variables, outputs=outputs,
                    nodes=nodes, sequence=sequence, intermediate=intermediate,
                    targets=targets, ir_version=self.obj.ir_version)

    def run(self, inputs, clean_right_away=False,
            intermediate=False, verbose=0, node_time=False,
            overwrite_types=None, fLOG=None):
        """
        Computes the predictions for this :epkg:`onnx` graph.

        :param inputs: inputs as dictionary or a dataframe
        :param clean_right_away: clean the intermediate outputs
                as soon as they are not needed
        :param intermediate: returns a dictionary of intermediate
            variables instead of the results only
        :param verbose: display information while predicting
        :param node_time: measure time of each node
        :param overwrite_types: shape inference does not work all the time,
            this allows to force types when building intermediate
            results, see @see fn select_model_inputs_outputs
        :param fLOG: logging function if *verbose > 0*
        :return: outputs as dictionary
            and a second dictionary of the time spent
            in each node if *node_time* is True

        .. exref::
            :title: Computes predictions with any runtime

            The following example compares predictions
            between :epkg:`scikit-learn` and this runtime
            for the python runtime.

            .. runpython::
                :showcode:
                :warningout: DeprecationWarning

                import numpy
                from sklearn.linear_model import LinearRegression
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from mlprodict.onnxrt import OnnxInference
                from mlprodict.onnx_conv import to_onnx

                iris = load_iris()
                X, y = iris.data, iris.target
                X_train, X_test, y_train, _ = train_test_split(X, y)
                clr = LinearRegression()
                clr.fit(X_train, y_train)

                exp = clr.predict(X_test[:5])
                print(exp)

                model_def = to_onnx(clr, X_train.astype(numpy.float32),
                                    target_opset=12)
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
        def retype(col_array):
            if (hasattr(col_array, 'categories') and
                    hasattr(col_array, 'from_codes')):
                # isinstance(col_array, pandas.Categorical):
                return col_array.astype(numpy.int64)
            return col_array

        if hasattr(inputs, 'columns') and hasattr(inputs, 'iloc'):
            # == isinstance(inputs, pandas.DataFrame)
            inputs = OrderedDict((
                name, retype(numpy.expand_dims(inputs[name].values, axis=1)))
                for name in inputs.columns)
        if intermediate:
            return self._run(inputs, clean_right_away=False,
                             intermediate=intermediate,
                             verbose=verbose, node_time=node_time,
                             overwrite_types=overwrite_types,
                             fLOG=fLOG)
        if overwrite_types is not None:
            raise RuntimeError(
                "overwrite_types is not used if intermediate is False.")
        return self._run(inputs, clean_right_away=False,
                         intermediate=intermediate,
                         verbose=verbose, node_time=node_time,
                         fLOG=fLOG)

    def display_sequence(self, verbose=1):
        """
        Shows the sequence of nodes to run if ``runtime=='python'``.
        """
        rows = []
        rows.append("#node: {}".format(len(self.sequence_)))
        for i, node in enumerate(self.sequence_):
            if verbose >= 1:
                rows.append("{}: {}".format(i, str(node)))
        return "\n".join(rows)

    def _run_sequence_runtime(self, inputs, clean_right_away=False,
                              intermediate=False, verbose=0, node_time=False,
                              overwrite_types=None, fLOG=None):
        if overwrite_types is not None:
            raise NotImplementedError(  # pragma: no cover
                "overwrite_types != None not implemented.")
        if clean_right_away:
            raise NotImplementedError(  # pragma: no cover
                "clean_right_away=true not implemented.")

        if node_time:
            mtime = []
        if verbose >= 1 and fLOG is not None:
            printed = set()

        if hasattr(self, "_values_init"):
            values = self._values_init.copy()  # pylint: disable=E0203
        else:
            values = [None] * len(self._global_index)
            if verbose >= 1 and fLOG is not None:
                for k, v in self.inits_.items():
                    values[self._global_index[k]] = v['value']
                    fLOG("+ki='{}': {} (dtype={} min={} max={})".format(
                        k, v['value'].shape, v['value'].dtype,
                        numpy_min(v['value']), numpy_max(v['value'])))
                    printed.add(k)
            else:
                for k, v in self.inits_.items():
                    values[self._global_index[k]] = v['value']
            # stores the array to skip initialing a second time
            if verbose == 0 or fLOG is None:
                self._values_init = values.copy()

        for name, value in inputs.items():
            values[self._global_index[name]] = value

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
                    if len(arr.shape) <= 1:
                        threshold = 8
                    else:
                        threshold = min(
                            50, min(50 // arr.shape[1], 8) * arr.shape[1])
                    if hasattr(arr, 'todense'):
                        fLOG(  # pragma: no cover
                            numpy.array2string(arr.todense(), max_line_width=120,
                                               suppress_small=True, threshold=threshold))
                    else:
                        fLOG(numpy.array2string(arr, max_line_width=120,
                                                suppress_small=True,
                                                threshold=threshold))
                else:  # pragma: no cover
                    s = str(arr)
                    if len(s) > 50:
                        s = s[:50] + "..."
                    fLOG(s)

            if verbose >= 2:
                for k in sorted(self._global_index):
                    if values[self._global_index[k]] is None:
                        continue
                    obj = values[self._global_index[k]]
                    if k not in printed:
                        printed.add(k)
                        if hasattr(obj, 'shape'):
                            fLOG("-kv='{}' shape={} dtype={} min={} max={}{}".format(
                                k, obj.shape, obj.dtype, numpy_min(obj),
                                numpy_max(obj),
                                ' (sparse)' if isinstance(obj, coo_matrix) else ''))
                        elif (isinstance(obj, list) and len(obj) > 0 and
                                not isinstance(obj[0], dict)):  # pragma: no cover
                            fLOG("-kv='{}' list len={} min={} max={}".format(
                                k, len(obj), min(obj), max(obj)))
                        else:  # pragma: no cover
                            fLOG("-kv='{}' type={}".format(k, type(obj)))

            keys = set(k for k in range(len(values)) if values[k] is not None)
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
                for k in range(len(values)):  # pylint: disable=C0200
                    if values[k] is None:
                        continue
                    if k not in keys and k not in printed:
                        printed.add(k)
                        name = list(
                            name for name in self._global_index  # pylint: disable=C0206
                            if self._global_index[name] == k)
                        if isinstance(values[k], (numpy.ndarray, coo_matrix)):
                            name = name[0]
                            mini = numpy_min(values[k])
                            maxi = numpy_max(values[k])
                            fLOG("+kr='{}': {} (dtype={} min={} max={}{})".format(
                                name, values[k].shape, values[k].dtype,
                                mini, maxi,
                                ' sparse' if isinstance(values[k], coo_matrix) else ''))
                            if verbose >= 3:
                                dispsimple(values[k])
                        else:
                            fLOG("+kr='{}': {}".format(
                                name, type(values[k])))
                            if verbose >= 3:  # pragma: no cover
                                dispsimple(values[k])

        if intermediate:
            values = [(v, k, values[v]) for k, v in self._global_index.items()]
            values.sort()
            values = OrderedDict((k, v) for _, k, v in values)
            return (values, mtime) if node_time else values

        try:
            res = {k: values[self._global_index[k]] for k in self.outputs_}
        except KeyError as e:  # pragma: no cover
            raise RuntimeError("Unable to find one output [{}]\n in [{}]"
                               ".".format(", ".join(sorted(self.outputs_)),
                                          ", ".join(sorted(values)))) from e
        return (res, mtime) if node_time else res

    def build_intermediate(self, outputs=None, verbose=0, overwrite_types=None,
                           fLOG=None):
        """
        Builds every possible :epkg:`ONNX` file
        which computes one specific intermediate output
        from the inputs.

        :param outputs: subsets of outputs to get,
            None to get all outputs,
        :param overwrite_types: shape inference does not work all the time,
            this allows to force types when building intermediate
            results, see @see fn select_model_inputs_outputs
        :param verbose: displays intermediate information
        :param fLOG: logging function
        :return: :epkg:`*py:collections:OrderedDict`

        .. versionchanged: 0.6
        """
        if verbose > 0:
            fLOG('[build_intermediate] BEGIN.')
        if outputs is not None:
            if isinstance(outputs, str):
                outputs = [outputs]
            if not isinstance(outputs, set):
                outputs = set(outputs)
        ord = OrderedDict()
        for output in enumerate_model_node_outputs(self.obj, order=True):
            if outputs is not None and output not in outputs:
                continue
            subonx = select_model_inputs_outputs(
                self.obj, outputs=output, infer_shapes=True,
                overwrite=overwrite_types)
            subonx = onnx_remove_node_unused(subonx)
            if verbose > 0:
                fLOG('[build_intermediate] + {}'.format(output))
            ord[output] = OnnxInference(subonx, runtime=self.runtime,
                                        skip_run=self.skip_run,
                                        runtime_options=self.runtime_options)
        if verbose > 0:
            fLOG('[build_intermediate] END.')
        return ord

    def _run_whole_runtime(self, inputs, clean_right_away=False,
                           intermediate=False, verbose=0, node_time=False,
                           overwrite_types=None, fLOG=None):
        # node_time is unused
        if clean_right_away:
            raise RuntimeError(  # pragma: no cover
                "clean_right_away=true does not work with this runtime.")
        if intermediate:
            if hasattr(self, "intermediate_onnx_inference_"):
                inter_run = self.intermediate_onnx_inference_  # pylint: disable=E0203
            else:
                if verbose > 0:
                    fLOG("-- OnnxInference: build intermediate")
                inter_run = self.build_intermediate(
                    verbose=verbose, fLOG=fLOG, overwrite_types=overwrite_types)
                self.intermediate_onnx_inference_ = inter_run
                graph = self.to_sequence()
                self.inits_ = graph['inits']

            if verbose >= 1:
                fLOG("-- OnnxInference: run {} nodes".format(
                    len(self.intermediate_onnx_inference_)))
            values = OrderedDict(inputs)
            for k, v in self.inits_.items():
                values[k] = v['value']
            if verbose >= 2:  # pragma: no cover
                for k in sorted(values):
                    fLOG("-k='{}' shape={} dtype={}".format(
                        k, values[k].shape, values[k].dtype))
            for node, oinf in self.intermediate_onnx_inference_.items():
                if verbose >= 4:
                    fLOG('[intermediate] %r' % node)
                    if verbose >= 5:  # pragma: no cover
                        fLOG(oinf.obj)
                output = oinf.run(inputs)[node]
                values[node] = output
                if verbose >= 1:
                    if verbose >= 4:
                        for k, v in inputs.items():
                            if isinstance(output, numpy.ndarray):
                                fLOG("-i='{}': {} (dtype={}) {}".format(
                                    k, v.shape, v.dtype, v.ravel().tolist()))
                            else:
                                fLOG("-i='{}': {} (dtype={}) - ?".format(
                                    k, v.shape, v.dtype))
                    if isinstance(output, numpy.ndarray):
                        fLOG("+k='{}': {} (dtype={})".format(
                            node, output.shape, output.dtype))
                        if verbose >= 2:
                            fLOG(output)
                    else:
                        fLOG("+k='{}': {}".format(  # pragma: no cover
                            node, type(output)))
                        if verbose >= 2:
                            fLOG(output)
            return values

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
            raise IndexError(  # pragma: no cover
                "Unable to get node name '{}'.\n{}".format(
                    node_name, "\n".join(node.name for node in self.obj.graph.node)))

        if att_name is None:
            return node_

        for att in node_.attribute:
            if att.name == att_name:
                return att

        raise IndexError(  # pragma: no cover
            "Unable to find attribute '{}' from node "
            "'{}'.".format(att_name, node_name))

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
        from ..onnx_tools.optim.sklearn_helper import enumerate_fitted_arrays, pairwise_array_distances

        if self.runtime != 'python':  # pragma: no cover
            raise RuntimeError("Initializers can be casted only if the "
                               "runtime is 'python' not '{}'.".format(self.runtime))

        if hasattr(self, '_values_init'):
            del self._values_init

        # first pass: simple cast
        done = []
        initializer = self.inits_
        for k, v in initializer.items():
            if isinstance(v['value'], numpy.ndarray):
                if v['value'].dtype == dtype_in:
                    v['value'] = v['value'].astype(dtype_out)
                    done.append(("pass1", "+", "init", k, v['value']))
                else:
                    done.append(("pass1", "-", "init", k,
                                 v['value']))  # pragma: no cover
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

    def _set_shape_inference_runtime(self):
        """
        Set shapes based on shape inference
        relying on the runtime.
        The values are stored in every node.
        """
        if not hasattr(self, 'sequence_') or not hasattr(self, 'inputs_'):
            raise RuntimeError(  # pragma: no cover
                "This method only works if the runtime is 'python' not "
                "'{}'.".format(self.runtime))
        values = OrderedDict()
        for k, v in self.inputs_.items():
            # The function assumes the first dimension is unknown
            # and is the batch size.
            values[k] = ShapeObject(v, use_n1=True, name=k)
        for k, v in self.inits_.items():
            values[k] = ShapeObject(v['value'], name=k)
        last = None
        for i, node in enumerate(self.sequence_):
            try:
                s = node._set_shape_inference_runtime(values)
                last = s
            except IndexError as e:  # pragma: no cover
                rows = []
                if last is not None:
                    for k, v in last.items():
                        rows.append("{}: {}".format(k, v))
                for k in range(i + 1):
                    rows.append("{} --> {}".format(k, self.sequence_[k]))
                raise RuntimeError("Unable to infer shape of node {}\n{}".format(
                    i, '\n'.join(rows))) from e
        return values

    def infer_shapes(self):
        """
        Computes expected shapes.

        :return: dictionary of shapes
        """
        return self._set_shape_inference_runtime()

    def _set_type_inference_runtime(self):
        """
        Set types based on type inference
        relying on the runtime.
        The values are stored in every node.
        """
        if not hasattr(self, 'sequence_') or not hasattr(self, 'inputs_'):
            raise RuntimeError(  # pragma: no cover
                "This method only works if the runtime is 'python' not "
                "'{}'.".format(self.runtime))
        values = OrderedDict()
        for k, v in self.inputs_.items():
            # The function assumes the first dimension is unknown
            # and is the batch size.
            values[k] = guess_numpy_type_from_string(v['type']['elem'])
        for k, v in self.inits_.items():
            values[k] = v['value'].dtype
        last = None
        for i, node in enumerate(self.sequence_):
            try:
                s = node._set_type_inference_runtime(values)
                last = s
            except IndexError as e:  # pragma: no cover
                rows = []
                if last is not None:
                    for k, v in last.items():
                        rows.append("{}: {}".format(k, v))
                for k in range(i + 1):
                    rows.append("{} --> {}".format(k, self.sequence_[k]))
                raise RuntimeError("Unable to infer type of node {}\n{}".format(
                    i, '\n'.join(rows))) from e
        return values

    def infer_types(self):
        """
        Computes expected shapes.

        :return: dictionary of types
        """
        return self._set_type_inference_runtime()

    def _set_size_inference_runtime(self, inputs):
        """
        Set sizes allocated during inference
        relying on the runtime.
        The values are stored in every node.
        """
        if not hasattr(self, 'sequence_') or not hasattr(self, 'inputs_'):
            raise RuntimeError(  # pragma: no cover
                "This method only works if the runtime is 'python' not "
                "'{}'.".format(self.runtime))
        values = OrderedDict()
        for k, v in self.inits_.items():
            values[k] = v['value']
        for k, v in self.inputs_.items():
            if k in inputs:
                values[k] = inputs[k]
        last = None
        for i, node in enumerate(self.sequence_):
            try:
                s = node._set_size_inference_runtime(values)
                last = s
            except IndexError as e:  # pragma: no cover
                rows = []
                if last is not None:
                    for k, v in last.items():
                        rows.append("{}: {}".format(k, v))
                for k in range(i + 1):
                    rows.append("{} --> {}".format(k, self.sequence_[k]))
                raise RuntimeError("Unable to infer size of node {}\n{}".format(
                    i, '\n'.join(rows))) from e
        return values

    def infer_sizes(self, inputs):
        """
        Computes expected sizes.

        :param inputs: inputs as a dictionary
        :return: dictionary of dictionary of sizes
        """
        res = self._set_size_inference_runtime(inputs)
        return {k: v for k, v in res.items() if k.startswith('#')}

    def _guess_inplace(self, input_inplace=False):
        """
        Looks into every node of the graph to see
        if there is a way to do the computation
        inplace. By default (*input_inplace=False*),
        the function assumes inputs cannot be modified
        so the first node cannot do inplace computation.
        This function only works with the python runtime.

        @param      input_inplace       the computation is allowed
                                        to overwrite the input

        This function checks that one node is used only
        once and then can be modified by the next node.
        Nodes `A`, `C` can be overwritten by the computation.
        Node `B` cannot as it is used by two nodes.

        .. blockdiag::

            diagram {
                A -> B -> C -> E;
                     B -> D;
           }

        It does not handle specific case such node `B` being
        overwritten by node `C` but without changing its shape
        and node `D` only needs the shape of `B`. Then `B` could
        be overwritten as well.
        """
        forbid = {}
        values = OrderedDict()
        for k in self.inputs_:
            values[k] = dict(inplace=input_inplace, to=[], fr=[])
        for k in self.inits_:
            values[k] = dict(inplace=False, to=[], fr=[])
        for node in self.sequence_:
            for n in node.inputs:
                values[n]['to'].append(node)
            for n in node.outputs:
                if node.op_type == 'Constant':
                    # We cannot modify constant.
                    forbid[n] = node
                if n not in values:
                    values[n] = dict(inplace=None, to=[], fr=[])
                values[n]['fr'].append(node)

        # checks the number of outputs
        outputs = set(self.output_names)
        modif = 1
        while modif > 0:
            modif = 0
            for n, v in values.items():
                if v['inplace'] is not None:
                    continue
                if n in forbid:
                    continue
                if len(v['to']) == 1:
                    v['inplace'] = True
                    modif += 1

        # convey the information to every node
        inplaces = {}
        for n, v in values.items():
            if v['inplace']:
                inplaces[n] = v
                for node in v['to']:
                    if n in outputs:
                        continue
                    node.enable_inplace_compute(n)

        return inplaces

    def _build_compile_run(self, debug=False):
        """
        Rewrite the run function in python,
        compiles it, and adds it as a method.

        @param      debug       insert debugging code
        @return                 method name, callable object

        .. exref::
            :title: Run a model with runtime 'python_compiled'

            The following code trains a model and compute
            the predictions with runtime ``'python_compiled'``.
            It converts the onnx graph into a python function
            which calls every operator. Its code is printed
            below.

            .. runpython::
                :showcode:
                :warningout: DeprecationWarning

                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import AdaBoostClassifier
                from sklearn.tree import DecisionTreeClassifier
                from skl2onnx import to_onnx
                from mlprodict.onnxrt import OnnxInference

                iris = load_iris()
                X, y = iris.data, iris.target
                X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
                y_train = y_train.astype(numpy.float32)
                clr = AdaBoostClassifier(
                    base_estimator=DecisionTreeClassifier(max_depth=3),
                    n_estimators=3)
                clr.fit(X_train, y_train)

                model_def = to_onnx(clr, X_train.astype(numpy.float32),
                                    target_opset=12)

                oinf2 = OnnxInference(model_def, runtime='python_compiled')
                print(oinf2.run({'X': X_test[:5]}))

                # prints out the python function equivalent
                # to the onnx graph
                print(oinf2)
        """
        def clean_name(name):
            return name.replace(":", "_").replace('.', '_').replace('/', '_')

        # inits
        inputs = self.input_names
        code = ['def compiled_run(dict_inputs):']
        if debug:
            code.append("    printed = {}")
        context = {}
        for k, v in self.inits_.items():
            if k.startswith("_OPT_"):
                raise RuntimeError(  # pragma: no cover
                    "The runtime cannot handle any constant name "
                    "starting with '_OPT_': '{}'.".format(k))
            if k in inputs:
                context["_OPT_" + k] = v['value']
                code.append("    # init: _OPT_{0}".format(k))
                if debug:
                    code.append(
                        "    debug_print('c.[_OPT_{0}]', _OPT_{0}, printed)".format(k))
            else:
                context[k] = v['value']
                code.append("    # init: {0}".format(k))
                if debug:
                    code.append(
                        "    debug_print('c.[{0}]', {0}, printed)".format(k))

        # method signature
        code.append("    # inputs")
        for inp in inputs:
            if '_OPT_' + inp in context:
                # optional inputs
                code.append(
                    "    {0} = dict_inputs.get('{1}', _OPT_{0})".format(
                        clean_name(inp), inp))
            else:
                code.append("    {0} = dict_inputs['{1}']".format(
                    clean_name(inp), inp))
            if debug:
                code.append(
                    "    debug_print('i.{0}', {1}, printed)".format(
                        clean_name(inp), inp))

        # code
        for i, node in enumerate(self.sequence_):
            name = "n{}_{}".format(i, node.ops_.__class__.__name__.lower())
            context[name] = node.ops_._run
            code.append('    ({1}, ) = {2}({0})'.format(
                ', '.join(map(clean_name, node.inputs)),
                ', '.join(map(clean_name, node.outputs)),
                name))
            if debug:
                code.append("    print('''# {}''')".format(code[-1][4:]))
                for o in node.outputs:
                    code.append(
                        "    debug_print('o.{0}', {1}, printed)".format(
                            clean_name(o), o))

        # return
        code.append('    return {')
        for out in self.output_names:
            code.append("        '{1}': {0},".format(
                clean_name(out), out))
        code.append('    }')
        final_code = '\n'.join(code)

        # compile the outcome
        context['self'] = self
        try:
            obj = compile(final_code, "<string>", 'exec')
        except SyntaxError as e:  # pragma: no cover
            raise SyntaxError(
                "Unable to compile\n#####\n{}".format(final_code)) from e
        fcts_obj = [_ for _ in obj.co_consts
                    if _ is not None and not isinstance(_, (bool, str, int))]
        fct = make_callable(
            "compiled_run", fcts_obj[0], final_code, context, debug)

        # end
        return "compiled_run", fct, final_code

    def reduce_size(self, pickable=False):
        """
        Reduces the memory footprint as much as possible.

        @param  pickable        keeps a pickle object?
        """
        import gc
        del self.graph_
        if not pickable:
            del self.obj
        if self.runtime in ('python_compiled', 'python_compiled_debug'):
            del self.sequence_
        gc.collect()

    def get_profiling(self, as_df=False):
        """
        Returns the profiling after a couple of execution.

        :param as_df: return the results as a dataframe (True)
        :return: dataframe or list of dictionaries

        .. versionadded:: 0.6
        """
        if (self.runtime_options is None or
                not self.runtime_options.get('enable_profiling', False)):
            raise RuntimeError(
                "Profiling is available if options 'enable_profiling' "
                "is set to true in 'runtime_options' but is %r." % self.runtime_options)
        prof = None
        if hasattr(self, '_whole'):
            prof = self._whole.get_profiling()
        if prof is None:
            raise NotImplementedError(  # pragma: no cover
                "profiling is only implemented for runtime 'onnxruntime1'.")
        if as_df:
            import pandas
            return pandas.DataFrame(prof)
        return prof
