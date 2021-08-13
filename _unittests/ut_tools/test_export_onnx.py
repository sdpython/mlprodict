"""
@brief      test log(time=14s)
"""
import os
import unittest
import collections
import inspect
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Any
import numpy
from onnx import numpy_helper, helper
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info)
from sklearn.cluster import KMeans
import autopep8
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxGather, OnnxIdentity, OnnxReshape, OnnxFlatten)
from mlprodict.onnx_tools.onnx_export import (
    export2onnx, export2tf2onnx, export2numpy)
from mlprodict.testing.verify_code import verify_code
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.exports.tf2onnx_helper import (
    make_sure, make_name, map_onnx_to_numpy_type, GraphBuilder)
from mlprodict.tools.code_helper import print_code
from mlprodict.onnx_tools.exports.numpy_helper import (
    argmin_use_numpy_select_last_index,
    make_slice)
from mlprodict.onnx_conv import to_onnx
from mlprodict.testing.einsum import decompose_einsum_equation
import mlprodict.npy.numpy_onnx_impl as npnx
from mlprodict.npy import onnxnumpy_np
from mlprodict.npy.onnx_numpy_annotation import NDArrayType


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
            print('[initializers] %r' % cls)

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
            print('[nodes] %r' % cls)

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
            print('[replace_all_inputs] %r' % cls)
        ctx.replace_all_inputs(oldnode.output[0], node.output[0])
        ctx.remove_node(oldnode.name)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        return cls.any_version(13, ctx, node, **kwargs)


class TestExportOnnx(ExtTestCase):

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

    def verify(self, content):
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
        loc = {'numpy_helper': numpy_helper,
               'make_model': make_model,
               'make_node': make_node,
               'set_model_props': set_model_props,
               'make_tensor': make_tensor,
               'make_graph': make_graph,
               'make_tensor_value_info': make_tensor_value_info,
               'print': print, 'sorted': sorted,
               'collections': collections, 'inspect': inspect}
        out = StringIO()
        err = StringIO()
        if len(left) >= 5:
            raise AssertionError(
                "Too many unknown symbols: %r." % left)

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
        names = ["fft2d_any.onnx"]
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

                    self.assertEqualArray(y['y'], y1['y'])
                    self.assertEqualArray(y['y'], y2['y'])

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
               'ConvertFFT2DOp': ConvertFFT2DOp, "make_name": make_name,
               'map_onnx_to_numpy_type': map_onnx_to_numpy_type,
               'GraphBuilder': GraphBuilder}
        out = StringIO()
        err = StringIO()
        if len(left) >= 14:
            raise AssertionError(
                "Too many unknown symbols: %r." % left)

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

    def test_export2tf2onnx(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = ["fft2d_any.onnx"]
        for rt in ['python', 'onnxruntime1']:
            for name in names:
                with self.subTest(name=name, rt=rt):
                    oinf0 = OnnxInference(
                        os.path.join(folder, name), runtime=rt)

                    x = numpy.random.randn(3, 1, 4).astype(numpy.float32)
                    y = oinf0.run({'x': x})

                    new_onnx = export2tf2onnx(
                        os.path.join(folder, name), name="FFT2D")
                    _, loc = self.verify_tf(new_onnx)
                    model = loc['onnx_raw']
                    self.assertIn('op_type: "FFT2D"', str(model))
                    model = loc['onnx_model']
                    self.assertNotIn('op_type: "FFT2D"', str(model))

                    oinf = OnnxInference(model, runtime=rt)
                    y1 = oinf.run({'x': x})

                    new_onnx = export2tf2onnx(
                        os.path.join(folder, name), name="FFT2D")
                    _, loc = self.verify_tf(new_onnx)
                    model = loc['onnx_model']
                    self.assertNotIn('op_type: "FFT2D"', str(model))
                    oinf = OnnxInference(model, runtime=rt)
                    y2 = oinf.run({'x': x})

                    self.assertEqualArray(y['y'], y1['y'])
                    self.assertEqualArray(y['y'], y2['y'])

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
            'make_slice': make_slice}
        out = StringIO()
        err = StringIO()
        if len(left) > 14:
            raise AssertionError(
                "Too many unknown symbols: %r." % left)

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
        names = ["fft2d_any.onnx"]
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
            'map_onnx_to_numpy_type': map_onnx_to_numpy_type, 'make_slice': make_slice}
        out = StringIO()
        err = StringIO()
        if len(left) > 14:
            raise AssertionError(
                "Too many unknown symbols: %r." % left)

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
        onx = seq_clean.to_onnx("Y", "X1", "X2", "X3", dtype=numpy.float32)

        with self.subTest(rt='python'):
            oinf = OnnxInference(onx)
            rr = oinf.run({'X1': x1, 'X2': x2, 'X3': x3})
            self.assertEqualArray(r, rr['Y'])
        with self.subTest(rt='onnxruntime1'):
            oinf = OnnxInference(onx, runtime='onnxruntime1')
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
            raise AssertionError("Shape mismatch %r != %r." %
                                 (a.shape, b.shape))
        diff = numpy.abs(a.ravel() - b.ravel()).max()
        if diff > error:
            raise AssertionError("Mismatch max diff=%r > %r." % (diff, error))

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
                # print(code)
                self.assertIn("make_sure", code)
                if __name__ == "__main__" and shape == (3, 1, 4):
                    code = code.replace("make_sure(", "utils.make_sure(")
                    code = code.replace("make_name(", "utils.make_name(")
                    code = code.replace("map_onnx_to_numpy_type(",
                                        "utils.map_onnx_to_numpy_type(")
                    code = code.replace("numpy.", "np.")
                    code = code.replace("TensorProto.", "onnx_pb.TensorProto.")
                    code = code.replace("dtype=np.float32", "dtype=np_dtype")
                    code = code.replace("value=make_tensor",
                                        "value=helper.make_tensor")
                    code = autopep8.fix_code(
                        code, options={'max_line_length': 120})
                    self.assertNotIn("numpy.", code)
                    # print(code)


if __name__ == "__main__":
    # TestExportOnnx().test_simple_configuration()
    unittest.main()
