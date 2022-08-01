# pylint: disable=W0201
"""
@brief      test log(time=14s)
"""
import os
import unittest
import collections
import inspect
import traceback
from typing import Any
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import numpy
from onnx import (
    helper, numpy_helper, load as onnx_load, TensorProto,
    ModelProto)
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info, make_opsetid, make_function)
from onnxruntime import SessionOptions, GraphOptimizationLevel
from sklearn.cluster import KMeans
import autopep8
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxGather, OnnxIdentity, OnnxReshape, OnnxFlatten,
    OnnxSlice, OnnxSqueeze)
from skl2onnx.common._topology import Variable as SklVariable
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_tools.onnx_export import (
    export2onnx, export2tf2onnx, export2numpy, export2xop,
    export2cpp, select_attribute, export2python)
from mlprodict.testing.verify_code import verify_code
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.exports.tf2onnx_helper import (
    make_sure, make_name, map_onnx_to_numpy_type, get_max_value,
    GraphBuilder)
from mlprodict.tools.code_helper import print_code
from mlprodict.onnx_tools.exports.numpy_helper import (
    argmin_use_numpy_select_last_index,
    argmax_use_numpy_select_last_index,
    make_slice)
from mlprodict.onnx_conv import to_onnx
from mlprodict.testing.einsum import decompose_einsum_equation
import mlprodict.npy.numpy_onnx_impl as npnx
from mlprodict.npy import onnxnumpy_np, onnxnumpy
from mlprodict.npy.onnx_numpy_annotation import NDArrayType
from mlprodict.npy.xop_variable import Variable as XopVariable
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.npy import NDArray
from mlprodict.onnx_tools.optim import onnx_remove_node_unused
from mlprodict.plotting.text_plot import onnx_simple_text_plot


class ConvertFFT2DOp:

    supported_dtypes = [
        numpy.float32,
    ]

    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):  # pylint: disable=R0915
        '''
        Converter for ``FFT2D``.

        * producer: skl2onnx
        * version: 0
        * description:
        '''
        oldnode = node
        input_name = node.input[0]
        onnx_dtype = ctx.get_dtype(input_name)
        make_sure(onnx_dtype in ConvertFFT2DOp.supported_dtypes,
                  "Unsupported input type.")
        vars = {x: x for x in node.input}  # pylint: disable=W0622

        # initializers
        if getattr(ctx, 'verbose', False):
            print(f'[initializers] {cls!r}')

        list_value = [1.0, 0.0]
        value = numpy.array(list_value, dtype=numpy.float32).reshape((2, 1, 1))

        r_Un_Unsqueezecst = ctx.make_const(
            name=make_name('init_Un_Unsqueezecst'), np_val=value)
        vars['Un_Unsqueezecst'] = r_Un_Unsqueezecst.name

        list_value = [0]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Un_Unsqueezecst1 = ctx.make_const(
            name=make_name('init_Un_Unsqueezecst1'), np_val=value)
        vars['Un_Unsqueezecst1'] = r_Un_Unsqueezecst1.name

        list_value = [1.0, 1.0, 1.0, 1.0, 1.0, 6.123234262925839e-17,
                      -1.0, -1.8369701465288538e-16, 1.0, -1.0, 1.0, -1.0, 1.0,
                      -1.8369701465288538e-16, -1.0, 5.510910704284357e-16, 0.0,
                      0.0, 0.0, 0.0, 0.0, -1.0, -1.2246468525851679e-16, 1.0, 0.0,
                      -1.2246468525851679e-16, 2.4492937051703357e-16,
                      -3.6739402930577075e-16, 0.0, 1.0, -3.6739402930577075e-16, -1.0]
        value = numpy.array(list_value, dtype=numpy.float32).reshape((2, 4, 4))

        r_Un_Unsqueezecst2 = ctx.make_const(
            name=make_name('init_Un_Unsqueezecst2'), np_val=value)
        vars['Un_Unsqueezecst2'] = r_Un_Unsqueezecst2.name

        list_value = [-1]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Co_Concatcst = ctx.make_const(
            name=make_name('init_Co_Concatcst'), np_val=value)
        vars['Co_Concatcst'] = r_Co_Concatcst.name

        list_value = [-2]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst = ctx.make_const(
            name=make_name('init_Sl_Slicecst'), np_val=value)
        vars['Sl_Slicecst'] = r_Sl_Slicecst.name

        value = numpy.array(0, dtype=numpy.int64)

        r_Ga_Gathercst = ctx.make_const(
            name=make_name('init_Ga_Gathercst'), np_val=value)
        vars['Ga_Gathercst'] = r_Ga_Gathercst.name

        list_value = [0, 0]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst2 = ctx.make_const(
            name=make_name('init_Sl_Slicecst2'), np_val=value)
        vars['Sl_Slicecst2'] = r_Sl_Slicecst2.name

        list_value = [1, 4]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst3 = ctx.make_const(
            name=make_name('init_Sl_Slicecst3'), np_val=value)
        vars['Sl_Slicecst3'] = r_Sl_Slicecst3.name

        list_value = [1, 2]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst4 = ctx.make_const(
            name=make_name('init_Sl_Slicecst4'), np_val=value)
        vars['Sl_Slicecst4'] = r_Sl_Slicecst4.name

        list_value = [4]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst6 = ctx.make_const(
            name=make_name('init_Sl_Slicecst6'), np_val=value)
        vars['Sl_Slicecst6'] = r_Sl_Slicecst6.name

        list_value = [1]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst7 = ctx.make_const(
            name=make_name('init_Sl_Slicecst7'), np_val=value)
        vars['Sl_Slicecst7'] = r_Sl_Slicecst7.name

        list_value = [3]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst9 = ctx.make_const(
            name=make_name('init_Sl_Slicecst9'), np_val=value)
        vars['Sl_Slicecst9'] = r_Sl_Slicecst9.name

        value = numpy.array(1, dtype=numpy.int64)

        r_Ga_Gathercst2 = ctx.make_const(
            name=make_name('init_Ga_Gathercst2'), np_val=value)
        vars['Ga_Gathercst2'] = r_Ga_Gathercst2.name

        list_value = [2]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst18 = ctx.make_const(
            name=make_name('init_Sl_Slicecst18'), np_val=value)
        vars['Sl_Slicecst18'] = r_Sl_Slicecst18.name

        list_value = [1, 3]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst24 = ctx.make_const(
            name=make_name('init_Sl_Slicecst24'), np_val=value)
        vars['Sl_Slicecst24'] = r_Sl_Slicecst24.name

        list_value = [2, 3]
        value = numpy.array(list_value, dtype=numpy.int64)

        r_Sl_Slicecst25 = ctx.make_const(
            name=make_name('init_Sl_Slicecst25'), np_val=value)
        vars['Sl_Slicecst25'] = r_Sl_Slicecst25.name

        # nodes
        if getattr(ctx, 'verbose', False):
            print(f'[nodes] {cls!r}')

        attr = dict()
        inputs = [vars['Un_Unsqueezecst'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze'))
        vars['Un_expanded0'] = node.output[0]

        attr = dict()
        inputs = [vars['Un_Unsqueezecst2'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze1'))
        vars['Un_expanded03'] = node.output[0]

        attr = dict()
        inputs = [vars['x'], ]
        node = ctx.make_node(
            'Shape', inputs=inputs, attr=attr,
            name=make_name('Sh_Shape'))
        vars['Sh_shape0'] = node.output[0]

        attr = dict()
        inputs = [vars['Sh_shape0'], ]
        node = ctx.make_node(
            'Shape', inputs=inputs, attr=attr,
            name=make_name('Sh_Shape1'))
        vars['Sh_shape01'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Sh_shape01'], vars['Ga_Gathercst'], ]
        node = ctx.make_node(
            'Gather', inputs=inputs, attr=attr,
            name=make_name('Ga_Gather'))
        vars['Ga_output01'] = node.output[0]

        attr = dict()
        inputs = [vars['Ga_output01'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze2'))
        vars['Un_expanded05'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Un_expanded05'], ]
        node = ctx.make_node(
            'Concat', inputs=inputs, attr=attr,
            name=make_name('Co_Concat'))
        vars['Co_concat_result01'] = node.output[0]

        attr = dict()
        inputs = [vars['Sh_shape0'], vars['Sl_Slicecst'],
                  vars['Co_concat_result01'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice'))
        vars['Sl_output05'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Co_Concatcst'], vars['Sl_output05'], ]
        node = ctx.make_node(
            'Concat', inputs=inputs, attr=attr,
            name=make_name('Co_Concat1'))
        vars['Co_concat_result0'] = node.output[0]

        attr = dict()
        inputs = [vars['x'], vars['Co_concat_result0'], ]
        node = ctx.make_node(
            'Reshape', inputs=inputs, attr=attr,
            name=make_name('Re_Reshape'))
        vars['Re_reshaped0'] = node.output[0]

        attr = dict()
        inputs = [vars['Re_reshaped0'], vars['Sl_Slicecst2'],
                  vars['Sl_Slicecst3'], vars['Sl_Slicecst4'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice1'))
        vars['Sl_output04'] = node.output[0]

        attr = dict(perm=[0, 2, 1],)
        inputs = [vars['Sl_output04'], ]
        node = ctx.make_node(
            'Transpose', inputs=inputs, attr=attr,
            name=make_name('Tr_Transpose'))
        vars['Tr_transposed02'] = node.output[0]

        attr = dict()
        inputs = [vars['Tr_transposed02'], vars['Un_Unsqueezecst1'],
                  vars['Sl_Slicecst6'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice2'))
        vars['Sl_output03'] = node.output[0]

        attr = dict()
        inputs = [vars['Sl_output03'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze3'))
        vars['Un_expanded04'] = node.output[0]

        attr = dict()
        inputs = [vars['Un_expanded03'], vars['Un_expanded04'], ]
        node = ctx.make_node(
            'MatMul', inputs=inputs, attr=attr,
            name=make_name('Ma_MatMul'))
        vars['Ma_Y01'] = node.output[0]

        attr = dict()
        inputs = [vars['Ma_Y01'], vars['Un_Unsqueezecst1'],
                  vars['Sl_Slicecst9'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice3'))
        vars['Sl_output02'] = node.output[0]

        attr = dict(perm=[1, 0, 3, 2],)
        inputs = [vars['Sl_output02'], ]
        node = ctx.make_node(
            'Transpose', inputs=inputs, attr=attr,
            name=make_name('Tr_Transpose1'))
        vars['Tr_transposed01'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Tr_transposed01'], vars['Ga_Gathercst'], ]
        node = ctx.make_node(
            'Gather', inputs=inputs, attr=attr,
            name=make_name('Ga_Gather1'))
        vars['Ga_output0'] = node.output[0]

        attr = dict()
        inputs = [vars['Ga_output0'], vars['Un_Unsqueezecst1'],
                  vars['Sl_Slicecst7'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice4'))
        vars['Sl_output01'] = node.output[0]

        attr = dict()
        inputs = [vars['Sl_output01'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze4'))
        vars['Un_expanded02'] = node.output[0]

        attr = dict()
        inputs = [vars['Un_expanded0'], vars['Un_expanded02'], ]
        node = ctx.make_node(
            'MatMul', inputs=inputs, attr=attr,
            name=make_name('Ma_MatMul1'))
        vars['Ma_Y0'] = node.output[0]

        attr = dict(perm=[1, 0, 2, 3],)
        inputs = [vars['Ma_Y0'], ]
        node = ctx.make_node(
            'Transpose', inputs=inputs, attr=attr,
            name=make_name('Tr_Transpose2'))
        vars['Tr_transposed0'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Tr_transposed01'], vars['Ga_Gathercst2'], ]
        node = ctx.make_node(
            'Gather', inputs=inputs, attr=attr,
            name=make_name('Ga_Gather2'))
        vars['Ga_output03'] = node.output[0]

        attr = dict()
        inputs = [vars['Ga_output03'], vars['Un_Unsqueezecst1'],
                  vars['Sl_Slicecst7'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice5'))
        vars['Sl_output07'] = node.output[0]

        attr = dict()
        inputs = [vars['Sl_output07'], vars['Sl_Slicecst7'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze6'))
        vars['Un_expanded07'] = node.output[0]

        attr = dict()
        inputs = [vars['Un_expanded0'], vars['Un_expanded07'], ]
        node = ctx.make_node(
            'MatMul', inputs=inputs, attr=attr,
            name=make_name('Ma_MatMul2'))
        vars['Ma_Y03'] = node.output[0]

        attr = dict(perm=[1, 0, 2, 3],)
        inputs = [vars['Ma_Y03'], ]
        node = ctx.make_node(
            'Transpose', inputs=inputs, attr=attr,
            name=make_name('Tr_Transpose3'))
        vars['Tr_transposed04'] = node.output[0]

        attr = dict()
        inputs = [vars['Tr_transposed04'], vars['Sl_Slicecst7'],
                  vars['Sl_Slicecst18'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice6'))
        vars['Sl_output06'] = node.output[0]

        attr = dict()
        inputs = [vars['Sl_output06'], ]
        node = ctx.make_node(
            'Neg', inputs=inputs, attr=attr,
            name=make_name('Ne_Neg'))
        vars['Ne_Y0'] = node.output[0]

        attr = dict()
        inputs = [vars['Tr_transposed04'], vars['Un_Unsqueezecst1'],
                  vars['Sl_Slicecst7'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice7'))
        vars['Sl_output08'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Ne_Y0'], vars['Sl_output08'], ]
        node = ctx.make_node(
            'Concat', inputs=inputs, attr=attr,
            name=make_name('Co_Concat2'))
        vars['Co_concat_result03'] = node.output[0]

        attr = dict()
        inputs = [vars['Tr_transposed0'], vars['Co_concat_result03'], ]
        node = ctx.make_node(
            'Add', inputs=inputs, attr=attr,
            name=make_name('Ad_Add'))
        vars['Ad_C0'] = node.output[0]

        attr = dict()
        inputs = [vars['Ad_C0'], vars['Sl_Slicecst2'],
                  vars['Sl_Slicecst24'], vars['Sl_Slicecst25'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice8'))
        vars['Sl_output0'] = node.output[0]

        attr = dict()
        inputs = [vars['Sh_shape0'], vars['Un_Unsqueezecst1'],
                  vars['Sl_Slicecst'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice9'))
        vars['Sl_output010'] = node.output[0]

        attr = dict()
        inputs = [vars['Sl_output0'], ]
        node = ctx.make_node(
            'Shape', inputs=inputs, attr=attr,
            name=make_name('Sh_Shape3'))
        vars['Sh_shape03'] = node.output[0]

        attr = dict()
        inputs = [vars['Sh_shape03'], ]
        node = ctx.make_node(
            'Shape', inputs=inputs, attr=attr,
            name=make_name('Sh_Shape4'))
        vars['Sh_shape04'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Sh_shape04'], vars['Ga_Gathercst'], ]
        node = ctx.make_node(
            'Gather', inputs=inputs, attr=attr,
            name=make_name('Ga_Gather3'))
        vars['Ga_output04'] = node.output[0]

        attr = dict()
        inputs = [vars['Ga_output04'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Unsqueeze', inputs=inputs, attr=attr,
            name=make_name('Un_Unsqueeze7'))
        vars['Un_expanded08'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Un_expanded08'], ]
        node = ctx.make_node(
            'Concat', inputs=inputs, attr=attr,
            name=make_name('Co_Concat3'))
        vars['Co_concat_result05'] = node.output[0]

        attr = dict()
        inputs = [vars['Sh_shape03'], vars['Sl_Slicecst'],
                  vars['Co_concat_result05'], vars['Un_Unsqueezecst1'], ]
        node = ctx.make_node(
            'Slice', inputs=inputs, attr=attr,
            name=make_name('Sl_Slice10'))
        vars['Sl_output012'] = node.output[0]

        attr = dict(axis=0,)
        inputs = [vars['Sl_Slicecst18'],
                  vars['Sl_output010'], vars['Sl_output012'], ]
        node = ctx.make_node(
            'Concat', inputs=inputs, attr=attr,
            name=make_name('Co_Concat4'))
        vars['Co_concat_result04'] = node.output[0]

        attr = dict()
        inputs = [vars['Sl_output0'], vars['Co_concat_result04'], ]
        node = ctx.make_node(
            'Reshape', inputs=inputs, attr=attr,
            name=make_name('Re_Reshape1'))
        vars['y'] = node.output[0]

        # finalize
        if getattr(ctx, 'verbose', False):
            print(f'[replace_all_inputs] {cls!r}')
        ctx.replace_all_inputs(oldnode.output[0], node.output[0])
        ctx.remove_node(oldnode.name)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        return cls.any_version(13, ctx, node, **kwargs)


class ConvertSlice2Op:
    supported_dtypes = [
        numpy.float32,
    ]

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Slice(T input, Index begin, Index size)
        # T output = Slice(T input, Tind starts, Tind ends, Tind axes, Tind steps)
        # "ends" are exclusive, "axes" and "steps" are optional,
        # their default val are [0, ...] and 1
        input_tensor = node.input[0]
        starts = node.input[1]
        size = node.input[2]
        # in tf, size can be -1 which means all elem are taken,
        # so size can't be added starts directly.
        # the way to make sure size are not less than 0:
        # set "sizes"'s elem to be int_max if elem val is -1
        size_dtype = ctx.get_dtype(size)
        size_np_dtype = map_onnx_to_numpy_type(size_dtype)
        if (ctx.get_node_by_output(size).is_const() and
                ctx.get_node_by_output(starts).is_const()):
            starts = ctx.get_node_by_output(starts).get_tensor_value()
            sizes = ctx.get_node_by_output(size).get_tensor_value()
            ends = []
            for start, size in zip(starts, sizes):
                # get all elements
                if size == -1:
                    dtype = ctx.get_dtype(node.input[1])
                    make_sure(
                        dtype, f"dtype of {node.input[1]} is None")
                    make_sure(
                        dtype, f"dtype of {node.input[1]} is None")
                    ends.append(numpy.iinfo(dtype).max)
                else:
                    ends.append(start + size)

        else:
            neg_one_val = numpy.array([-1]).astype(size_np_dtype)
            neg_one = ctx.make_const(
                make_name("const"), neg_one_val).output[0]

            int_max_val = numpy.array(
                [get_max_value(size_np_dtype)]).astype(size_np_dtype)
            int_max = ctx.make_const(
                make_name("largest_int_val"), int_max_val).output[0]

            size_are_neg_one_flag = ctx.make_node(
                "Equal", [neg_one, size]).output[0]
            size_are_neg_one_flag = ctx.make_node(
                "Cast", [size_are_neg_one_flag],
                attr={"to": size_dtype}).output[0]
            value_to_add = ctx.make_node(
                "Mul", [int_max, size_are_neg_one_flag]).output[0]
            size_processed = ctx.make_node(
                "Add", [size, value_to_add]).output[0]
            ends = ctx.make_node(
                "Add", [starts, size_processed]).output[0]

        ctx.remove_node(node.name)
        inputs_map = {"data": input_tensor, "starts": starts, "ends": ends}
        kwargs = {**inputs_map, "outputs": node.output}
        _ = GraphBuilder(ctx).make_slice(kwargs, name=node.name)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)


class ConvertSqueeze2Op:

    supported_dtypes = [
        numpy.float32,
    ]

    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        '''
        Converter for ``Squeeze2``.

        * producer: skl2onnx
        * version: 0
        * description:
        '''
        oldnode = node
        input_name = node.input[0]
        onnx_dtype = ctx.get_dtype(input_name)
        np_dtype = map_onnx_to_numpy_type(onnx_dtype)
        make_sure(np_dtype in ConvertSqueeze2Op.supported_dtypes,
                  "Unsupported input type.")
        # shape = ctx.get_shape(input_name)
        varx = {x: x for x in node.input}

        # initializers
        if getattr(ctx, 'verbose', False):
            print(f'[initializers] {cls!r}')

        value = numpy.array([1], dtype=numpy.int64)
        varx['Sq_Squeezecst'] = ctx.make_const(
            name=make_name('init_Sq_Squeezecst'), np_val=value).name

        # nodes
        if getattr(ctx, 'verbose', False):
            print(f'[nodes] {cls!r}')

        node = GraphBuilder(ctx).make_squeeze(
            {'data': varx['X'], 'axes': [1]}, return_node=True)
        varx['Y'] = node.output[0]

        # finalize
        if getattr(ctx, 'verbose', False):
            print(f'[replace_all_inputs] {cls!r}')
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

    value = make_tensor_value_info('X', 1, [None, 1])
    inputs.append(value)

    # outputs
    print('[outputs]')   # verbose

    value = make_tensor_value_info('Y', 1, None)
    outputs.append(value)

    inames = [i.name for i in inputs]
    onames = [i.name for i in outputs]
    node = make_node('Squeeze2', inames, onames, name='Squeeze2')

    # graph
    print('[graph]')   # verbose
    graph = make_graph([node], 'Squeeze2', inputs, outputs)
    onnx_model = make_model(graph)
    onnx_model.ir_version = 7
    onnx_model.producer_name = 'skl2onnx'
    onnx_model.producer_version = ''
    onnx_model.domain = 'ai.onnx'
    onnx_model.model_version = 0
    onnx_model.doc_string = ''
    set_model_props(onnx_model, {})

    # opsets
    print('[opset]')   # verbose
    opsets = {'': 13}
    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for dom, value in opsets.items():
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = dom
        op_set.version = value

    return onnx_model


class TestExportOnnx(ExtTestCase):

    def test_get_max_value(self):
        self.assertEqual(get_max_value(numpy.int8), 127)

    def test_model_data_slice(self):
        opv = 14

        var = SklVariable('x', 'x', type=FloatTensorType([None, None, 4]),
                          scope=None)

        op = OnnxSlice(var,
                       numpy.array([0], dtype=numpy.int64),
                       numpy.array([1], dtype=numpy.int64),
                       op_version=opv)

        sq = OnnxSqueeze(op, numpy.array([0], dtype=numpy.int64),
                         op_version=opv, output_names=['y'])

        onx = sq.to_onnx(inputs=[var], target_opset=opv)
        with open("temp_slice.onnx", "wb") as f:
            f.write(onx.SerializeToString())

    def test_simple_configuration(self):
        op_version = 13

        def case1():
            xi = OnnxGather('x', numpy.array([3], dtype=numpy.int64),
                            op_version=op_version)
            xis = OnnxReshape(xi, numpy.array([-1], dtype=numpy.int64),
                              op_version=op_version)
            node = OnnxIdentity(xis, output_names=['y'], op_version=op_version)
            onx = node.to_onnx(inputs=[('x', Int64TensorType())],
                               target_opset=op_version)

            xi = OnnxGather('x', numpy.array([3], dtype=numpy.int64),
                            op_version=op_version)
            node = OnnxIdentity(xi, output_names=['y'], op_version=op_version)
            onx2 = node.to_onnx(inputs=[('x', Int64TensorType())],
                                target_opset=op_version)

            x = numpy.arange(10).astype(numpy.int64)
            for rt in ['python', 'onnxruntime1']:
                oinf = OnnxInference(onx, runtime=rt)
                y = oinf.run({'x': x})['y']
                self.assertEqual(y[0], 3)
                self.assertEqual(y.shape, (1, ))
                oinf = OnnxInference(onx2, runtime=rt)
                y = oinf.run({'x': x})['y']
                self.assertEqual(y[0], 3)
                self.assertEqual(y.shape, (1, ))

        def case2():
            # This proves that Reshape([-1], works on a number as well.
            xi = OnnxGather('x', numpy.array(3, dtype=numpy.int64),
                            op_version=op_version)
            xis = OnnxReshape(xi, numpy.array([-1], dtype=numpy.int64),
                              op_version=op_version)
            node = OnnxIdentity(xis, output_names=['y'], op_version=op_version)
            onx = node.to_onnx(inputs=[('x', Int64TensorType())],
                               target_opset=op_version)

            xi = OnnxGather('x', numpy.array(3, dtype=numpy.int64),
                            op_version=op_version)
            node = OnnxIdentity(xi, output_names=['y'], op_version=op_version)
            onx2 = node.to_onnx(inputs=[('x', Int64TensorType())],
                                target_opset=op_version)

            x = numpy.arange(10).astype(numpy.int64)
            for rt in ['python', 'onnxruntime1']:
                oinf = OnnxInference(onx, runtime=rt)
                y = oinf.run({'x': x})['y']
                self.assertEqual(y[0], 3)
                self.assertEqual(y.shape, (1, ))
                oinf = OnnxInference(onx2, runtime=rt)
                y = oinf.run({'x': x})['y']
                self.assertEqual(y, 3)
                self.assertEqual(y.shape, tuple())

        def case3():
            # This proves that Reshape([-1], works on a number as well.
            xi = OnnxGather('x', numpy.array(3, dtype=numpy.int64),
                            op_version=op_version)
            xis = OnnxFlatten(xi, axis=0, op_version=op_version)
            node = OnnxIdentity(xis, output_names=['y'], op_version=op_version)
            onx = node.to_onnx(inputs=[('x', Int64TensorType())],
                               target_opset=op_version)

            xi = OnnxGather('x', numpy.array(3, dtype=numpy.int64),
                            op_version=op_version)
            node = OnnxIdentity(xi, output_names=['y'], op_version=op_version)
            onx2 = node.to_onnx(inputs=[('x', Int64TensorType())],
                                target_opset=op_version)

            x = numpy.arange(10).astype(numpy.int64)
            for rt in ['onnxruntime1', 'python']:
                oinf = OnnxInference(onx, runtime=rt)
                y = oinf.run({'x': x})['y']
                self.assertEqual(y[0], 3)
                self.assertEqual(y.shape, (1, 1))
                oinf = OnnxInference(onx2, runtime=rt)
                y = oinf.run({'x': x})['y']
                self.assertEqual(y, 3)
                self.assertEqual(y.shape, tuple())

        case1()
        case2()
        case3()

    def verify(self, content, more_context=None, limit_left=10):
        try:
            left, __ = verify_code(content, exc=False)
        except (SyntaxError, AttributeError) as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {'numpy_helper': numpy_helper,
               'make_model': make_model,
               'make_node': make_node,
               'set_model_props': set_model_props,
               'make_tensor': make_tensor,
               'make_graph': make_graph,
               'make_function': make_function,
               'make_tensor_value_info': make_tensor_value_info,
               'print': print, 'sorted': sorted,
               'make_opsetid': make_opsetid,
               'collections': collections, 'inspect': inspect}
        if more_context is not None:
            loc.update(more_context)
            glo.update(more_context)
        out, err = StringIO(), StringIO()
        if limit_left is not None and len(left) >= limit_left:
            raise AssertionError(
                f"Too many unknown symbols ({len(left)}): {left!r} in\n{content}")

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_export_onnx(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = ["fft2d_any.onnx", "slice.onnx"]
        for rt in ['python', 'onnxruntime1']:
            for name in names:
                with self.subTest(name=name, rt=rt):
                    oinf0 = OnnxInference(
                        os.path.join(folder, name), runtime=rt)

                    x = numpy.random.randn(3, 1, 4).astype(numpy.float32)

                    new_onnx = export2onnx(
                        os.path.join(folder, name), name="FFT2D")
                    _, loc = self.verify(new_onnx)
                    model = loc['onnx_model']

                    if name == 'fft2d_any.onnx':
                        oinf = OnnxInference(
                            model, runtime=rt, new_outputs=['Sh_shape0'],
                            new_opset=10)
                        rr = oinf.run({'x': x})
                        if rr['Sh_shape0'].shape != (3, ):
                            self.assertEqual(rr['Sh_shape0'].shape, (3, ))

                    oinf = OnnxInference(model, runtime=rt)
                    if rt == 'python':
                        y = oinf0.run({'x': x})
                        y1 = oinf.run({'x': x})
                    else:
                        y = oinf0.run({'x': x})
                        y1 = oinf.run({'x': x})

                    new_onnx = export2onnx(
                        os.path.join(folder, name), verbose=False)
                    _, loc = self.verify(new_onnx)
                    model = loc['onnx_model']
                    oinf = OnnxInference(model, runtime=rt)
                    y2 = oinf.run({'x': x})

                    if y1['y'].shape[0] > 0 and y['y'].shape[0] > 0:
                        self.assertEqualArray(y['y'], y1['y'])
                    if name == 'fft2d_any.onnx':
                        self.assertEqualArray(y['y'], y2['y'])

                    code2 = oinf.to_onnx_code()
                    self.assertEqual(new_onnx, code2)

    def verify_tf(self, content):
        try:
            left, __ = verify_code(content, exc=False)
        except SyntaxError as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {'numpy': numpy, 'dict': dict, 'list': list,
               'print': print, 'sorted': sorted,
               'collections': collections, 'inspect': inspect,
               'helper': helper, "make_sure": make_sure,
               'ConvertFFT2DOp': ConvertFFT2DOp,
               'ConvertSlice2Op': ConvertSlice2Op,
               "make_name": make_name,
               'map_onnx_to_numpy_type': map_onnx_to_numpy_type,
               'GraphBuilder': GraphBuilder}
        out, err = StringIO(), StringIO()
        if len(left) >= 14:
            raise AssertionError(
                f"Too many unknown symbols: {left!r}.")

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    tb = traceback.format_exc()
                    raise AssertionError(
                        "Unable to execute a script due to %r\n%s. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, tb, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_export2tf2onnx(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = [("gslice.onnx", 'Slice2', 'X', (3, 10, 5), 'Y'),
                 ("gsqueeze.onnx", 'Squeeze2', 'X', (3, 1), 'Y'),
                 ("fft2d_any.onnx", 'FFT2D', 'x', (3, 1, 4), 'y')]
        for rt in ['python', 'onnxruntime1']:
            for name, op_name, x_name, x_shape, y_name in names:
                with self.subTest(name=name, rt=rt):
                    with open(os.path.join(folder, name), "rb") as f:
                        onx = onnx_load(f)
                    onx = onnx_remove_node_unused(onx)
                    oinf0 = OnnxInference(
                        onx, runtime=rt, runtime_options=dict(
                            log_severity_level=3))

                    x = numpy.random.randn(*x_shape).astype(numpy.float32)
                    y = oinf0.run({x_name: x})

                    new_onnx = export2tf2onnx(
                        os.path.join(folder, name), name=op_name,
                        verbose=False)
                    _, loc = self.verify_tf(new_onnx)
                    model = loc['onnx_raw']
                    self.assertIn(f'op_type: "{op_name}"', str(model))
                    self.assertNotEqual(
                        loc['onnx_raw'].SerializeToString(),
                        loc['onnx_model'].SerializeToString())
                    model = loc['onnx_model']
                    self.assertNotIn(f'op_type: "{op_name}"', str(model))

                    if rt == 'onnxruntime1':
                        opts = SessionOptions()
                        opts.log_severity_level = 3
                        opts.graph_optimization_level = (
                            GraphOptimizationLevel.ORT_DISABLE_ALL)
                        oinf = OnnxInference(
                            model, runtime=rt, runtime_options=opts)
                    else:
                        oinf = OnnxInference(model, runtime=rt)
                    y1 = oinf.run({x_name: x})

                    new_onnx = export2tf2onnx(
                        os.path.join(folder, name), name=op_name)
                    _, loc = self.verify_tf(new_onnx)
                    model = loc['onnx_model']
                    self.assertNotIn(f'op_type: "{op_name}"', str(model))
                    oinf = OnnxInference(
                        model, runtime=rt, runtime_options=dict(
                            log_severity_level=3))
                    y2 = oinf.run({x_name: x})

                    if y1[y_name].shape[0] > 0 and y[y_name].shape[0] > 0:
                        self.assertEqualArray(y[y_name], y1[y_name])
                        self.assertEqualArray(y[y_name], y2[y_name])

    def verify_numpy(self, content):
        try:
            left, __ = verify_code(content, exc=False)
        except SyntaxError as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {
            'numpy': numpy, 'dict': dict, 'list': list,
            'print': print, 'sorted': sorted,
            'collections': collections, 'inspect': inspect,
            'helper': helper, "make_sure": make_sure,
            'ConvertFFT2DOp': ConvertFFT2DOp, "make_name": make_name,
            'argmin_use_numpy_select_last_index': argmin_use_numpy_select_last_index,
            'argmax_use_numpy_select_last_index': argmax_use_numpy_select_last_index,
            'make_slice': make_slice}
        out, err = StringIO(), StringIO()
        if len(left) > 14:
            raise AssertionError(
                f"Too many unknown symbols ({len(left)}): {left!r} in \n{content}")

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_export2numpy(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = ["fft2d_any.onnx", "slice.onnx"]
        for name in names:
            with self.subTest(name=name):
                oinf0 = OnnxInference(os.path.join(folder, name))

                x = numpy.arange(12).reshape((3, 1, 4)).astype(numpy.float32)
                y = oinf0.run({'x': x})

                code = export2numpy(
                    os.path.join(folder, name), name="FFT2D")
                code += ("\nx = numpy.arange(12).reshape((3, 1, 4))."
                         "astype(numpy.float32)\ny = numpy_FFT2D(x)")
                _, loc = self.verify_numpy(code)
                self.assertEqualArray(y['y'], loc['y'])

    @ignore_warnings(UserWarning)
    def test_export2numpy_kmeans(self):
        X = numpy.arange(20).reshape(10, 2).astype(numpy.float32)
        X[:5] = - X[:5]
        tr = KMeans(n_clusters=2)
        tr.fit(X)
        onx = to_onnx(tr, X, target_opset=14)
        code = export2numpy(onx, name="kmeans", rename=True)

        oinf0 = OnnxInference(onx)
        y = oinf0.run({'X': X})

        code += ("\nx = numpy.arange(20).reshape(10, 2).astype(numpy.float32)"
                 "\nx[:5] = - x[:5]"
                 "\nlabel, scores = numpy_kmeans(x)")
        _, loc = self.verify_numpy(code)
        self.assertEqualArray(y['scores'], loc['scores'])
        self.assertEqualArray(y['label'], loc['label'])

    def verify_numpy_einsum(self, content):
        try:
            left, __ = verify_code(content, exc=False)
        except SyntaxError as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {
            'numpy': numpy, 'dict': dict, 'list': list,
            'print': print, 'sorted': sorted,
            'collections': collections, 'inspect': inspect,
            'helper': helper, "make_sure": make_sure,
            'ConvertFFT2DOp': ConvertFFT2DOp, "make_name": make_name,
            'argmin_use_numpy_select_last_index': argmin_use_numpy_select_last_index,
            'argmax_use_numpy_select_last_index': argmax_use_numpy_select_last_index,
            'map_onnx_to_numpy_type': map_onnx_to_numpy_type, 'make_slice': make_slice}
        out, err = StringIO(), StringIO()
        if len(left) > 14:
            raise AssertionError(
                f"Too many unknown symbols: {left!r}.")

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_export_einsum(self):
        x1 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)
        x2 = numpy.arange(4).reshape(2, 2).astype(numpy.float32)
        x3 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)
        r = numpy.einsum("bac,cd,def->ebc", x1, x2, x3)
        seq_clean = decompose_einsum_equation(
            "bac,cd,def->ebc", strategy='numpy', clean=True)
        onx = seq_clean.to_onnx("Y", "X1", "X2", "X3", dtype=numpy.float32,
                                target_opset=15)

        with self.subTest(rt='onnxruntime1'):
            opts = SessionOptions()
            opts.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
            oinf = OnnxInference(
                onx, runtime='onnxruntime1', runtime_options=opts)
            rr = oinf.run({'X1': x1, 'X2': x2, 'X3': x3})
            self.assertEqualArray(r, rr['Y'])
        with self.subTest(rt='python'):
            oinf = OnnxInference(onx)
            rr = oinf.run({'X1': x1, 'X2': x2, 'X3': x3})
            self.assertEqualArray(r, rr['Y'])

        code = export2numpy(onx, name="einsum", rename=True)
        self.assertIn("BM =", code)
        code += "\n".join([
            "x1 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)",
            "x2 = numpy.arange(4).reshape(2, 2).astype(numpy.float32)",
            "x3 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)",
            "r = numpy_einsum(x1, x2, x3)"
        ])
        _, loc = self.verify_numpy_einsum(code)
        self.assertEqualArray(r, loc['r'])

    def test_export_einsum2(self):
        x1 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)
        x2 = numpy.arange(4).reshape(2, 2).astype(numpy.float32)
        r = numpy.einsum("bac,cd->ad", x1, x2)
        seq_clean = decompose_einsum_equation(
            "bac,cd->ad", strategy='numpy', clean=True)
        onx = seq_clean.to_onnx("Y", "X1", "X2", dtype=numpy.float32)

        with self.subTest(rt='python'):
            oinf = OnnxInference(onx)
            rr = oinf.run({'X1': x1, 'X2': x2})
            self.assertEqualArray(r, rr['Y'])
        with self.subTest(rt='onnxruntime1'):
            oinf = OnnxInference(onx, runtime='onnxruntime1')
            rr = oinf.run({'X1': x1, 'X2': x2})
            self.assertEqualArray(r, rr['Y'])

        code = export2numpy(onx, name="einsum")
        code += "\n".join([
            "x1 = numpy.arange(8).reshape(2, 2, 2).astype(numpy.float32)",
            "x2 = numpy.arange(4).reshape(2, 2).astype(numpy.float32)",
            "r = numpy_einsum(x1, x2)"
        ])
        _, loc = self.verify_numpy_einsum(code)
        self.assertEqualArray(r, loc['r'])
        self.assertIn(", axis=3)", code)

    def test_onnx_dft_real_cst(self):

        def dft_real_cst(N, fft_length):
            n = numpy.arange(N)
            k = n.reshape((N, 1)).astype(numpy.float64)
            M = numpy.exp(-2j * numpy.pi * k * n / fft_length)
            both = numpy.empty((2,) + M.shape)
            both[0, :, :] = numpy.real(M)
            both[1, :, :] = numpy.imag(M)
            return both.astype(numpy.float32)

        @onnxnumpy_np(signature=NDArrayType(("T:int64", "T"), dtypes_out=('T',)))
        def onnx_dft_real_cst(x_shape, fft_length):
            N = x_shape[-2]
            n = npnx.arange(0, N).astype(numpy.float32)
            new_shape = npnx.concat(npnx.expand_dims(N, axis=0),
                                    numpy.array([1], dtype=numpy.int64))
            k = n.reshape(new_shape).astype(numpy.float32)
            kn = (k * n /
                  fft_length.astype(numpy.float32) *
                  npnx.cst(-2 * numpy.pi, dtype=numpy.float32))
            mcos = npnx.unsqueeze(npnx.cos(kn), axes=0)
            msin = npnx.unsqueeze(npnx.sin(kn), axes=0)
            return npnx.vstack(mcos, msin)

        x_shape = numpy.array([3, 4], dtype=numpy.int64)
        fft_length = numpy.array([2, 3], dtype=numpy.int64)
        exp = dft_real_cst(x_shape[-2], fft_length[-1])
        cus = onnx_dft_real_cst(x_shape, fft_length[-1])
        self.assertEqualArray(exp, cus, decimal=5)

    def assert_almost_equal(self, a, b, error=1e-5):
        """
        The function compares two matrices, one may be complex. In that case,
        this matrix is changed into a new matrix with a new first dimension,
        [0,::] means real part, [1,::] means imaginary part.
        """
        if a.dtype in (numpy.complex64, numpy.complex128):
            dtype = numpy.float64 if a.dtype == numpy.complex128 else numpy.float32
            new_a = numpy.empty((2,) + a.shape).astype(dtype)
            new_a[0] = numpy.real(a)
            new_a[1] = numpy.imag(a)
            self.assert_almost_equal(new_a, b, error)
            return
        if b.dtype in (numpy.complex64, numpy.complex128):
            self.assert_almost_equal(b, a, error)  # pylint: disable=W1114
            return
        if a.shape != b.shape:
            raise AssertionError(f"Shape mismatch {a.shape!r} != {b.shape!r}.")
        diff = numpy.abs(a.ravel() - b.ravel()).max()
        if diff > error:
            raise AssertionError(f"Mismatch max diff={diff!r} > {error!r}.")

    def test_einsum_numpy_full(self):

        def onnx_dft_real_cst(N, fft_length):
            n = npnx.arange(0, N).astype(numpy.float32)
            new_shape = npnx.concat(npnx.expand_dims(N, axis=0),
                                    numpy.array([1], dtype=numpy.int64))
            k = n.reshape(new_shape).astype(numpy.float32)
            kn = (k * n /
                  fft_length.astype(numpy.float32) *
                  npnx.cst(-2 * numpy.pi, dtype=numpy.float32))
            mcos = npnx.unsqueeze(npnx.cos(kn), axes=0)
            msin = npnx.unsqueeze(npnx.sin(kn), axes=0)
            return npnx.vstack(mcos, msin)

        def onnx_rfft_3d_1d(x, fft_length, transpose=True):
            if fft_length is None:
                raise RuntimeError("fft_length must be specified.")

            size = fft_length // 2 + 1
            cst = onnx_dft_real_cst(fft_length, fft_length)
            if transpose:
                xt = npnx.transpose(x, (0, 2, 1))
                a = cst[:, :, :fft_length]
                b = xt[:, :fft_length, :]
                a = npnx.expand_dims(a, 0)
                b = npnx.expand_dims(b, 1)
                res = npnx.matmul(a, b)
                res2 = res[:, :size, :]
                return npnx.transpose(res2, (1, 0, 3, 2))
            else:
                a = cst[:, :, :fft_length]
                b = x[:, :fft_length, :]
                a = npnx.expand_dims(a, 0)
                b = npnx.expand_dims(b, 1)
                res = npnx.matmul(a, b)
                return npnx.transpose(res, (1, 0, 2, 3))

        def onnx_rfft_3d_2d(x, fft_length):
            mat = x[:, :fft_length[-2], :fft_length[-1]]

            # first FFT
            res = onnx_rfft_3d_1d(mat, fft_length[-1], transpose=True)

            # second FFT decomposed on FFT on real part and imaginary part
            res2_real = onnx_rfft_3d_1d(res[0], fft_length[0], transpose=False)
            res2_imag = onnx_rfft_3d_1d(res[1], fft_length[0], transpose=False)
            res2_imag2 = npnx.vstack(-res2_imag[1:2], res2_imag[:1])
            res = res2_real + res2_imag2
            size = fft_length[1] // 2 + 1
            return res[:, :, :fft_length[-2], :size]

        @onnxnumpy_np(signature=NDArrayType(("T:all", numpy.int64), dtypes_out=('T',)))
        def onnx_rfft_2d_any_test(x, fft_length):
            new_shape = npnx.concat(
                numpy.array([-1], dtype=numpy.int64), x.shape[-2:], axis=0)
            mat2 = x.reshape(new_shape)
            f2 = onnx_rfft_3d_2d(mat2, fft_length)
            new_shape = npnx.concat(
                numpy.array([2], dtype=numpy.int64), x.shape[:-2], f2.shape[-2:])
            return f2.reshape(new_shape)

        for shape, fft_length in [((3, 1, 4), (1, 4)),
                                  ((5, 7), (5, 7))]:
            with self.subTest(shape=shape, fft_length=fft_length):
                fft_length = numpy.array(fft_length, dtype=numpy.int64)
                rnd = numpy.random.randn(*list(shape)).astype(numpy.float32)
                fft2d_cus = numpy.fft.fft2(rnd, fft_length)
                try:
                    fft2d_onx = onnx_rfft_2d_any_test(rnd, fft_length)
                except RuntimeError:
                    key = list(onnx_rfft_2d_any_test.signed_compiled)[0]
                    onx = onnx_rfft_2d_any_test.signed_compiled[key].compiled.onnx_
                    with open("temp_fft2s_dynamic.onnx", "wb") as f:
                        f.write(onx.SerializeToString())
                    oinf = OnnxInference(onx)
                    print('--------------------- ERROR')
                    res = oinf.run({'x': rnd, 'fft_length': fft_length},
                                   verbose=1, fLOG=print)
                    print('--------------------- ERROR')
                    raise

                self.assert_almost_equal(
                    fft2d_cus[..., :fft2d_onx.shape[-1]], fft2d_onx, error=1e-4)

                key = list(onnx_rfft_2d_any_test.signed_compiled)[0]
                self.assertEqual(
                    len(list(onnx_rfft_2d_any_test.signed_compiled)), 1)
                onx = onnx_rfft_2d_any_test.signed_compiled[key].compiled.onnx_
                for rt in ['python', 'onnxruntime1']:
                    with self.subTest(rt=rt):
                        oinf = OnnxInference(onx, runtime=rt)
                        res = oinf.run({'x': rnd, 'fft_length': fft_length})
                        self.assertEqualArray(fft2d_onx, res['y'], decimal=5)

                with open("temp_fft2s_dynamic.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                code = export2tf2onnx(
                    onx, name="FFT2D", autopep_options={'max_line_length': 120})

                self.assertIn("make_sure", code)
                if __name__ == "__main__" and shape == (3, 1, 4):
                    code = code.replace("make_sure(", "make_sure(")
                    code = code.replace("make_name(", "make_name(")
                    code = code.replace("map_onnx_to_numpy_type(",
                                        "map_onnx_to_numpy_type(")
                    code = code.replace("numpy.", "np.")
                    code = code.replace("TensorProto.", "onnx_pb.TensorProto.")
                    code = code.replace("dtype=np.float32", "dtype=np_dtype")
                    code = code.replace("value=make_tensor",
                                        "value=helper.make_tensor")
                    code = autopep8.fix_code(
                        code, options={'max_line_length': 120})
                    self.assertNotIn("numpy.", code)

    def test_sub_graph(self):
        data = os.path.abspath(os.path.dirname(__file__))
        debug = os.path.join(data, "data", "debug.onnx")
        code = export2onnx(debug)
        self.assertIn("def _create_Scan_Sc_Scan1_body():", code)

    def test_scan_knn(self):
        x = numpy.random.randn(3, 4).astype(numpy.float32)
        data = os.path.abspath(os.path.dirname(__file__))
        knn = os.path.join(
            data, "data", "SklearnKNeighborsRegressor2.model.onnx")
        onx = OnnxInference(knn)
        y1 = onx.run({'input': x})['variable']
        new_onnx = export2onnx(knn)
        _, loc = self.verify(new_onnx)
        model = loc['onnx_model']
        oinf = OnnxInference(model)
        y2 = oinf.run({'input': x})['variable']
        self.assertEqual(y1, y2)

    def test_select_attribute(self):
        class A:
            def __init__(self, i):
                self.i = i

            def __repr__(self):
                return f'A({self.i!r})'
        ens = [A("a"), A("b"), A("c"), A("a")]
        self.assertEqual(['a', 'b', 'c', 'a'], select_attribute(ens, 'i'))
        self.assertEqual(['a', 'a', 'b', 'c'],
                         select_attribute(ens, 'i', sort=True))
        self.assertEqual(['a', 'b', 'c'],
                         select_attribute(ens, 'i', sort=True, unique=True))

    def test_select_attribute_dict(self):
        self.assertEqual([], select_attribute([], 'i'))
        ens = [{'i': "a"}, {'i': "b"}, {'i': "c"}, {'i': "a"}]
        self.assertEqual(['a', 'b', 'c', 'a'], select_attribute(ens, 'i'))
        self.assertEqual(['a', 'a', 'b', 'c'],
                         select_attribute(ens, 'i', sort=True))
        self.assertEqual(['a', 'b', 'c'],
                         select_attribute(ens, 'i', sort=True, unique=True))

    def verify_xop(self, content, onx_graph):
        try:
            left, __ = verify_code(content, exc=False)
        except SyntaxError as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {'loadop': loadop, 'Variable': XopVariable,
               'print': print, 'sorted': sorted, 'len': len,
               'TensorProto': TensorProto, 'make_tensor': make_tensor,
               'OnnxOperatorFunction': OnnxOperatorFunction}
        glo.update(loc)
        out, err = StringIO(), StringIO()
        if len(left) >= 5:
            raise AssertionError(
                "Too many unknown symbols: %r in\n%s\n-----\n%s" % (
                    left, onnx_simple_text_plot(onx_graph), content))

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_export_xop(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = ["slice.onnx", "fft2d_any.onnx"]
        for rt in ['onnxruntime1', 'python']:
            for name in names:
                with self.subTest(name=name, rt=rt):
                    with open(os.path.join(folder, name), 'rb') as f:
                        onx_graph = onnx_load(f)
                    oinf0 = OnnxInference(
                        os.path.join(folder, name), runtime=rt)

                    x = numpy.random.randn(3, 1, 4).astype(numpy.float32)

                    new_onnx = export2xop(
                        os.path.join(folder, name), name="FFT2D")
                    _, loc = self.verify_xop(new_onnx, onx_graph)
                    model = loc['onnx_model']

                    try:
                        oinf = OnnxInference(model, runtime=rt)
                    except RuntimeError as e:
                        raise AssertionError(
                            "Issue with\n-----\n%s\n--CODE--\n%s\n--GOT--\n%s" % (
                                onnx_simple_text_plot(onx_graph), new_onnx,
                                onnx_simple_text_plot(model))) from e
                    if rt == 'python':
                        y = oinf0.run({'x': x})
                        y1 = oinf.run({'x': x})
                    else:
                        y = oinf0.run({'x': x})
                        y1 = oinf.run({'x': x})

                    new_onnx = export2xop(
                        os.path.join(folder, name), verbose=False)
                    _, loc = self.verify_xop(new_onnx, onx_graph)
                    model = loc['onnx_model']
                    oinf = OnnxInference(model, runtime=rt)
                    y2 = oinf.run({'x': x})

                    if y1['y'].shape[0] > 0 and y['y'].shape[0] > 0:
                        self.assertEqualArray(y['y'], y1['y'])
                    if name == 'fft2d_any.onnx':
                        self.assertEqualArray(y['y'], y2['y'])

    def test_export_function_xop(self):
        # ONNX
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(  # pylint: disable=W0621
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd(ov, numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)

        for rt in ['onnxruntime1', 'python']:
            with self.subTest(rt=rt):
                oinf0 = OnnxInference(onx, runtime=rt)
                x = numpy.random.randn(3, 1, 4).astype(numpy.float32)
                new_onnx = export2xop(onx, name="TEST")
                _, loc = self.verify_xop(new_onnx, onx)
                model = loc['onnx_model']

                try:
                    oinf = OnnxInference(model, runtime=rt)
                except RuntimeError as e:
                    raise AssertionError(
                        "Issue with\n-----\n%s\n--CODE--\n%s\n--GOT--\n%s" % (
                            onnx_simple_text_plot(onx), new_onnx,
                            onnx_simple_text_plot(model))) from e
                y = oinf0.run({'X': x})
                y1 = oinf.run({'X': x})

                new_onnx = export2xop(onx, name="TEST")
                _, loc = self.verify_xop(new_onnx, onx)
                model = loc['onnx_model']
                oinf = OnnxInference(model, runtime=rt)
                y2 = oinf.run({'X': x})
                self.assertEqual(y['Y'], y1['Y'])
                self.assertEqual(y['Y'], y2['Y'])

    def test_export_function_onnx(self):
        # ONNX
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(  # pylint: disable=W0621
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd(ov, numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)

        for rt in ['onnxruntime1', 'python']:
            with self.subTest(rt=rt):
                oinf0 = OnnxInference(onx, runtime=rt)
                x = numpy.random.randn(3, 1, 4).astype(numpy.float32)
                new_onnx = export2onnx(onx, name="TEST")
                _, loc = self.verify(new_onnx)
                model = loc['onnx_model']

                try:
                    oinf = OnnxInference(model, runtime=rt)
                except RuntimeError as e:
                    raise AssertionError(
                        "Issue with\n-----\n%s\n--CODE--\n%s\n--GOT--\n%s" % (
                            onnx_simple_text_plot(onx), new_onnx,
                            onnx_simple_text_plot(model))) from e
                y = oinf0.run({'X': x})
                y1 = oinf.run({'X': x})

                new_onnx = export2onnx(onx, name="TEST")
                _, loc = self.verify_xop(new_onnx, onx)
                model = loc['onnx_model']
                oinf = OnnxInference(model, runtime=rt)
                y2 = oinf.run({'X': x})
                self.assertEqual(y['Y'], y1['Y'])
                self.assertEqual(y['Y'], y2['Y'])

    def test_export_function_cpp(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        onx_file = os.path.join(data, "switch_axes.inlined.onnx")
        with open(onx_file, "rb") as f:
            model = onnx_load(f)
        self.assertIsInstance(model, ModelProto)
        code = export2cpp(model)
        self.assertIn('model.graph.ParseFromString(R"(', code)

    def test_export_function_python(self):
        # ONNX
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(  # pylint: disable=W0621
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd(ov, numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)

        class LocalDomain:
            def __init__(self, domain, version):
                self.domain = domain
                self.version = version

        mlprodict1 = LocalDomain('mlprodict', 1)
        opset14 = LocalDomain('', 14)
        opset14.Abs = numpy.abs
        opset14.Constant = lambda value: numpy_helper.to_array(value)
        x = numpy.random.randn(3, 4).astype(numpy.float32)

        for rt in ['python']:
            with self.subTest(rt=rt):
                oinf0 = OnnxInference(onx, runtime=rt)
                expected_onx = oinf0.run({'X': x})['Y']
                new_onnx = export2python(onx, name="TEST")
                self.assertIn('def main', new_onnx)
                self.assertIn(' + ', new_onnx)
                self.assertIn(' / ', new_onnx)
                _, loc = self.verify(
                    new_onnx, more_context={
                        'mlprodict1': mlprodict1,
                        'opset14': opset14})
                mlprodict1.AddAbs = loc['AddAbs']
                fct = loc['main']
                y = fct(x)
                expected = (numpy.abs(x) + 1) / 2
                self.assertEqualArray(expected, y)
                self.assertEqualArray(expected_onx, y)

    @staticmethod
    def fct_onnx_if(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
        "onnx numpy abs"
        xif = npnx.onnx_if(
            npnx.sum(x) > numpy.float32(0),
            then_branch=npnx.if_then_else(
                numpy.array([-1], dtype=numpy.float32)),
            else_branch=numpy.array([1], dtype=numpy.float32))
        return xif + numpy.float32(-7)

    def test_export_if(self):
        fct_if = onnxnumpy()(TestExportOnnx.fct_onnx_if)
        onx = fct_if.compiled.onnx_
        new_onnx = export2python(onx, name="TEST")
        self.assertIn('def main', new_onnx)
        self.assertIn(' > ', new_onnx)

        class LocalDomain:
            def __init__(self, domain, version):
                self.domain = domain
                self.version = version

        mlprodict1 = LocalDomain('mlprodict', 1)
        opset15 = LocalDomain('', 15)
        opset15.ReduceSum = numpy.sum
        opset15.Identity = lambda i: i
        opset15.Constant = lambda value: numpy_helper.to_array(value)

        _, loc = self.verify(
            new_onnx, more_context={
                'mlprodict1': mlprodict1,
                'opset15': opset15})

        fct = loc['main']
        x = numpy.random.randn(3, 4).astype(numpy.float32)
        y = fct(x)
        expected = fct_if(x)
        self.assertEqualArray(expected, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
