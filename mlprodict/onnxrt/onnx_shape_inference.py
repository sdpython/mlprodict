"""
@file
@brief Runtime to infer shapes.

.. versionadded:: 0.9
"""
import numpy
from onnx import FunctionProto, ModelProto
from onnx.numpy_helper import to_array
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from .ops_shape.shape_result import ShapeResult
from .ops_shape.shape_container import ShapeContainer
from .ops_shape import shape_dispatch


class OnnxShapeInference:
    """
    Implements a micro runtime for ONNX graphs.
    It does not implements all the operator types.

    :param model_onnx: ONNX model

    Other attributes:

    * `known_shapes_`: shapes which can be inferred without any input
    * `cache_`: keeps track of the function used to infer
      the shapes
    * `is_isfunction`: tells if the graph is a function or a model

    .. runpython::
        :showcode:

        import pprint
        import numpy
        from mlprodict.onnxrt.onnx_shape_inference import OnnxShapeInference
        from mlprodict.npy.xop_variable import Variable
        from mlprodict.npy.xop import loadop

        opset = 15
        OnnxAdd = loadop('Add')
        dtype = numpy.float32

        cop = OnnxAdd('X', numpy.array(
            [[1]], dtype=dtype), op_version=opset)
        cop4 = OnnxAdd(cop, numpy.array([[2]], dtype=dtype),
                       output_names=['Y'])
        vari = Variable('X', numpy.float32, [None, 3])
        model_def = cop4.to_onnx([vari], run_shape=False)
        rt = OnnxShapeInference(model_def)
        out = rt.run()
        pprint.pprint(out.get())
    """

    def __init__(self, model_onnx):
        if not isinstance(model_onnx, (FunctionProto, ModelProto)):
            raise TypeError(  # pragma: no cover
                "model_onnx is not from FunctionProto or ModelProto "
                "%r." % type(model_onnx))
        self.is_function = isinstance(model_onnx, FunctionProto)
        self.model_onnx = model_onnx
        self.cache_ = {}
        self.known_shapes_ = self._run_empty()

    @property
    def input_names(self):
        "Returns input names."
        if self.is_function:
            return list(self.model_onnx.input)
        return [i.name for i in self.model_onnx.graph.input]

    @property
    def output_names(self):
        "Returns output names."
        if self.is_function:
            return list(self.model_onnx.output)
        return [i.name for i in self.model_onnx.graph.output]

    def __repr__(self):
        "Usual"
        return f"{self.__class__.__name__}(...)"

    @staticmethod
    def _get_shape(obj, known_shapes=None, result_name=None):
        if obj is None:
            return [], None, False
        dtype = TENSOR_TYPE_TO_NP_TYPE.get(
            obj.type.tensor_type.elem_type, None)
        shape = []
        for dimi, d in enumerate(obj.type.tensor_type.shape.dim):
            v = d.dim_value if d.dim_value > 0 else d.dim_param
            if v in ('', None):
                if known_shapes is None or result_name is None:
                    raise RuntimeError(  # pragma: no cover
                        "known_shapes must be specified if "
                        "a dimension is not.")
                v = known_shapes.get_new_name(v, result_name, dimi)
            shape.append(v)
        return shape, dtype, False

    def _run_empty(self):
        """
        Computes shape and types of all results.

        :return: all intermediates results and output as a dictionary
        """
        def get_obj(name, inputs):
            if self.is_function:
                return None
            if inputs:
                for o in self.model_onnx.graph.input:
                    if o.name == name:
                        return o
            else:
                for o in self.model_onnx.graph.output:
                    if o.name == name:
                        return o
            return None

        known_shapes = ShapeContainer()
        if not self.is_function:
            for init in self.model_onnx.graph.initializer:
                mat = to_array(init)
                known_shapes.update(init.name, ShapeResult(
                    init.name, mat.shape, mat.dtype, sparse=False))

        for name in self.input_names:
            if name in known_shapes:
                raise NotImplementedError(
                    f"Optional inputs are not implemented yet. (name={name!r})")
            shape, dtype, sparse = self._get_shape(
                get_obj(name, True), known_shapes, result_name=name)
            known_shapes.update(name, ShapeResult(
                name, shape, dtype, sparse=sparse))

        for name in self.output_names:
            if name in known_shapes:
                raise NameError(  # pragma: no cover
                    f"Output {name!r} is already present. Use Identity node.")
            shape, dtype, sparse = self._get_shape(
                get_obj(name, False), known_shapes, result_name=name)
            if dtype is None:
                # The onnx graph was created with named outputs
                # but with no type or shape.
                continue
            known_shapes.update(name, ShapeResult(
                name, shape, dtype, sparse=sparse))

        nodes = (
            self.model_onnx.node if self.is_function
            else self.model_onnx.graph.node)
        cont = True
        while cont:
            cont = False
            for node in nodes:
                cont = cont or shape_dispatch(
                    self.cache_, known_shapes, node, rt_class=self.__class__)
        return known_shapes

    def run(self, inputs=None):
        """
        Runs shape inference and type given known inputs.

        :param inputs: inputs
        :return: all results
        """
        known_shapes = self.known_shapes_.copy(deep=True)
        if inputs is None:
            known_shapes.resolve()
            return known_shapes

        cont = False
        for name, obj in inputs.items():
            shape, dtype, sparse = (
                obj.shape, obj.dtype, not isinstance(obj, numpy.ndarray))
            cont = cont or known_shapes.update(
                name, ShapeResult(name, shape, dtype, sparse=sparse))

        nodes = (
            self.model_onnx.node if self.is_function
            else self.model_onnx.graph.node)
        while cont:
            cont = False
            for node in nodes:
                updated = shape_dispatch(
                    self.cache_, known_shapes, node, rt_class=self.__class__)
                cont = cont or updated
        known_shapes.resolve()
        return known_shapes
