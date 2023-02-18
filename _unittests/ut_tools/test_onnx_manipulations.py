# pylint: disable=R0915,W0703,W0632
"""
@brief      test log(time=11s)
"""
import unittest
import os
import pprint
import time
import warnings
from collections import Counter
import numpy
from onnx import (
    helper, TensorProto, load, FunctionProto, ModelProto,
    GraphProto, AttributeProto)
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, ignore_warnings
from pyquickhelper.texthelper.edit_text_diff import (
    diff2html, edit_distance_text)
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.npy.xop_variable import Variable
from mlprodict.onnx_tools.optim.onnx_helper import onnx_statistics
from mlprodict.onnx_tools.onnx_tools import (
    enumerate_onnx_names, enumerate_onnx_nodes)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import onnx_remove_node_unused
from mlprodict.onnx_tools.onnx2py_helper import get_tensor_elem_type
from mlprodict.onnx_tools.onnx_manipulations import (
    select_model_inputs_outputs, enumerate_model_node_outputs,
    onnx_rename_names, insert_results_into_onnx, onnx_model_to_function,
    onnx_inline_function, onnx_function_to_model, change_input_type,
    change_subgraph_io_type_shape, onnx_rename_inputs_outputs,
    onnx_replace_functions, get_opsets,
    replace_initializer_by_constant_of_shape)
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnxrt.excs import MissingOperatorError
from mlprodict.onnx_tools.model_checker import check_onnx
from mlprodict.onnx_tools.onnx_export import export2cpp


class TestOptimOnnxManipulations(ExtTestCase):

    def test_onnx_remove_unused_outputs(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, "inter", infer_shapes=True, remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 1)
        self.assertEqual(stats2['nnodes'], 1)
        self.assertEqual(stats3['nnodes'], 1)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertNotIn('final', y1)
        self.assertNotIn('final', y2)
        self.assertIn('inter', y1)
        self.assertIn('inter', y2)
        self.assertEqualArray(y1['inter'], y2['inter'])

    def test_onnx_remove_unused_outputs_new(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def0 = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def0, "inter", infer_shapes=True, remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = select_model_inputs_outputs(
            model_def0, "inter", infer_shapes=True)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 1)
        self.assertEqual(stats2['nnodes'], 1)
        self.assertEqual(stats3['nnodes'], 1)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertNotIn('final', y1)
        self.assertNotIn('final', y2)
        self.assertIn('inter', y1)
        self.assertIn('inter', y2)
        self.assertEqualArray(y1['inter'], y2['inter'])

    def test_onnx_remove_unused_inputs(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', cop2,
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=TARGET_OPSET),
            cop3, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        check_onnx(model_def)

        rows = []

        def myprint(*args):
            rows.append(" ".join(map(str, args)))

        model_def0 = model_def
        model_def = select_model_inputs_outputs(
            model_def, inputs=["inter"], infer_shapes=True, remove_unused=False,
            verbose=2, fLOG=myprint)
        try:
            check_onnx(model_def)
        except Exception as e:
            raise AssertionError(  # pylint: disable=W0707
                "Model verification failed due to %s\n---LOG--\n%s"
                "\n--ONNX0--\n%s\n--ONNX1--\n%s" % (
                    str(e).split("\n", maxsplit=1)[0], "\n".join(rows),
                    onnx_simple_text_plot(model_def0),
                    onnx_simple_text_plot(model_def)))
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 1)
        self.assertEqual(stats2['ninits'], 1)
        self.assertEqual(stats3['ninits'], 0)
        self.assertEqual(stats2['nnodes'], 2)
        self.assertEqual(stats3['nnodes'], 2)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'inter': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'inter': x})
        self.assertIn('final', y1)
        self.assertIn('final', y2)
        self.assertNotIn('inter', y1)
        self.assertNotIn('inter', y2)
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_remove_unused_inputs_overwrite(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', cop2,
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=TARGET_OPSET),
            cop3, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, inputs=["inter"], infer_shapes=False,
            overwrite=dict(inter=(numpy.float32, [None, None]),
                           final=(numpy.float32, [None, None])),
            remove_unused=False)
        check_onnx(model_def)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 1)
        self.assertEqual(stats2['ninits'], 1)
        self.assertEqual(stats3['ninits'], 0)
        self.assertEqual(stats2['nnodes'], 2)
        self.assertEqual(stats3['nnodes'], 2)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'inter': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'inter': x})
        self.assertIn('final', y1)
        self.assertIn('final', y2)
        self.assertNotIn('inter', y1)
        self.assertNotIn('inter', y2)
        self.assertEqualArray(y1['final'], y2['final'])

    def test_enumerate_model_node_outputs(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        nodes1 = list(enumerate_model_node_outputs(model_def))
        nodes2 = list(enumerate_model_node_outputs(model_def, order=True))
        self.assertEqual(list(sorted(nodes1)), list(sorted(nodes2)))
        expected = ['inter', 'out_add_0', 'out_mul_0', 'final']
        self.assertEqual(nodes2, expected)

    def test_onnx_rename_names_exc(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        self.assertRaise(
            lambda: onnx_rename_names(model_def, strategy="none"),
            ValueError)

    def test_onnx_rename_names_simple(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(model_def, verbose=1, fLOG=flog)
        total = "\n".join(rows)
        self.assertIn("[onnx_rename_names] init: 'init_1' -> 'i1'", total)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'X': x})
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_rename_names_type(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(
            model_def, verbose=1, fLOG=flog, strategy='type')
        total = "\n".join(rows)
        self.assertIn("'init' -> 'i_DB'", total)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'X': x})
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_rename_node_scan(self):
        from mlprodict.npy.xop_opset import OnnxReduceSumSquareApi18
        (OnnxSub, OnnxIdentity, OnnxScan) = loadop(
            'Sub', 'Identity', 'Scan')

        def onnx_squareform_pdist(X, dtype=None, op_version=None, **kwargs):
            diff = OnnxSub('next_in', 'next',
                           op_version=op_version)
            id_next = OnnxIdentity('next_in', output_names=['next_out'],
                                   op_version=op_version)
            flat = OnnxReduceSumSquareApi18(
                diff, axes=[1], op_version=op_version,
                output_names=['scan_out'], keepdims=0)
            scan_body = id_next.to_onnx(
                [Variable('next_in', numpy.float32, (None, None)),  # tensor_type([None, None])),
                 Variable('next', numpy.float32, (None, ))],  # tensor_type([None]))]),
                outputs=[Variable('next_out', numpy.float32, (None, None)),  # ([None, None])),
                         Variable('scan_out', numpy.float32, (None, ))],  # tensor_type([None]))],
                other_outputs=[flat],
                target_opset=op_version)
            node = OnnxScan(X, X, output_names=['S1', 'S2'],
                            num_scan_inputs=1,
                            body=(scan_body.graph, [id_next, flat]),
                            op_version=op_version, **kwargs)
            return node[1]

        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        opv = TARGET_OPSET
        onnx_fct = OnnxIdentity(onnx_squareform_pdist(
            'x', op_version=opv), output_names='Y', op_version=opv)
        model_def = onnx_fct.to_onnx(inputs={'x': numpy.float32})

        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(
            model_def, verbose=1, fLOG=flog, strategy='type')
        total = "\n".join(rows)
        self.assertNotIn('name: "Re_ReduceSumSquare"', str(new_model))
        self.assertIn("'node__reducesumsquare_", total)
        oinf2 = OnnxInference(new_model)
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        y1 = oinf1.run({'x': x})
        y2 = oinf2.run({'x': x})
        self.assertEqualArray(y1['Y'], y2['Y'])

    def test_insert_results_into_onnx(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.INT64, None)  # pylint: disable=E1101
        node_def = helper.make_node('Shape', ['X'], ['Z0'], name='Zt')
        node_def1 = helper.make_node('Identity', ['Z0'], ['Z'], name='Zti')
        graph_def = helper.make_graph(
            [node_def, node_def1], 'test-model', [X], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        new_graph = insert_results_into_onnx(
            model_def, {'Z0': numpy.array([[29, 39]], dtype=numpy.int64)})
        s_graph = str(new_graph)
        self.assertIn('domain: "DEBUG"', s_graph)
        self.assertNotIn('pname', s_graph)
        self.assertIn('op_type: "DEBUG"', s_graph)
        self.assertRaise(lambda: insert_results_into_onnx(
            model_def, {'Zt': numpy.array([29, 39], dtype=numpy.int64)}),
            RuntimeError)
        # with open('debug.onnx', 'wb') as f:
        #     f.write(new_graph.SerializeToString())

        oinf1 = OnnxInference(model_def, inplace=False)
        oinf2 = OnnxInference(new_graph, inplace=False)
        cst = numpy.array([[5.6, 7.8]])
        self.assertEqualArray(oinf1.run({'X': cst})['Z'],
                              oinf2.run({'X': cst})['Z'])

        onx = oinf1.run2onnx({'X': cst})[1]
        s_graph = str(onx)
        self.assertIn('domain: "DEBUG"', s_graph)
        self.assertIn('op_type: "DEBUG"', s_graph)
        self.assertNotIn('pname', s_graph)
        oinf3 = OnnxInference(onx)
        self.assertEqualArray(oinf1.run({'X': cst})['Z'],
                              oinf3.run({'X': cst})['Z'])

    def test_insert_results_into_onnx_init(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.INT64, None)  # pylint: disable=E1101
        node_def = helper.make_node('Shape', ['X'], ['Z0'], name='Zt')
        node_def1 = helper.make_node('Identity', ['Z0'], ['Z'], name='Zti')
        graph_def = helper.make_graph(
            [node_def, node_def1], 'test-model', [X], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        new_graph = insert_results_into_onnx(
            model_def, {'Z0': numpy.array([[29, 39]], dtype=numpy.int64)},
            as_parameter=False, param_name=lambda k: k)
        s_graph = str(new_graph)
        self.assertIn('domain: "DEBUG"', s_graph)
        self.assertIn('op_type: "DEBUG"', s_graph)
        self.assertRaise(lambda: insert_results_into_onnx(
            model_def, {'Zt': numpy.array([29, 39], dtype=numpy.int64)}),
            RuntimeError)
        self.assertRaise(lambda: insert_results_into_onnx(
            model_def, {'X': numpy.array([29, 39], dtype=numpy.int64)}),
            NotImplementedError)
        # with open('debug.onnx', 'wb') as f:
        #     f.write(new_graph.SerializeToString())

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_graph)
        cst = numpy.array([[5.6, 7.8]])
        self.assertEqualArray(oinf1.run({'X': cst})['Z'],
                              oinf2.run({'X': cst})['Z'])

    def test_onnx_enumerate_onnx_names(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        names = list(enumerate_onnx_names(model_def))
        self.assertEqual(len(names), 16)
        self.assertIn('X', names)
        self.assertIn('inter', names)

    def test_onnx_to_function(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        fft2d = os.path.join(data, "fft2d.onnx")
        onx = load(fft2d)

        # original graph
        oinf = OnnxInference(onx)
        x = numpy.random.randn(7, 7).astype(numpy.float32)
        y = oinf.run({'x': x})['y']

        opsets1 = get_opsets(onx)
        fct, _ = onnx_model_to_function(onx, name="fft2d")
        opsets2 = get_opsets(fct)
        self.assertEqual(opsets1, opsets2)
        self.assertIsInstance(fct, FunctionProto)

        op = OnnxOperatorFunction(fct, 'X', output_names=['Y'])
        onx2 = op.to_onnx(numpy.float32, numpy.float32)
        s2 = str(onx2)
        self.assertIn("functions {", s2)
        self.assertIn('name: "fft2d"', s2)
        oinf2 = OnnxInference(onx2)
        y2 = oinf2.run({'X': x})['Y']
        self.assertEqualArray(y, y2)

    def test_onnx_inline_function(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        fft2d = os.path.join(data, "fft2d.onnx")
        onx = load(fft2d)
        fct, _ = onnx_model_to_function(onx, name="fft2d")
        op = OnnxOperatorFunction(fct, 'X', output_names=['Y'])
        onx2 = op.to_onnx(numpy.float32, numpy.float32)
        inlined, m = onnx_inline_function(onx2)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].op_type, "fft2d")
        s3 = str(inlined)
        self.assertNotIn("functions {", s3)

        x = numpy.random.randn(7, 7).astype(numpy.float32)
        oinf2 = OnnxInference(onx2)
        y2 = oinf2.run({'X': x})['Y']
        oinf3 = OnnxInference(inlined)
        y3 = oinf3.run({'X': x})['Y']
        self.assertEqualArray(y2, y3)

    def test_onnx_inline_function_function(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        fft2d = os.path.join(data, "fft2d.onnx")
        onx = load(fft2d)
        fct, _ = onnx_model_to_function(onx, name="fft2d")
        op = OnnxOperatorFunction(fct, 'X', output_names=['Y'])
        onx2 = op.to_onnx(numpy.float32, numpy.float32)

        fct, _ = onnx_model_to_function(onx2, name="fft2d")
        inlined, m = onnx_inline_function(fct, list(onx2.functions))
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].op_type, "fft2d")
        self.assertEqual(len(inlined.node), 35)

    def test_onnx_inline_subgraph(self, log=False):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        one = helper.make_tensor_value_info(
            'one', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101

        graph1 = helper.make_graph([], 'then', [], [X])
        graph2 = helper.make_graph([], 'else', [], [one])

        graph_def = helper.make_graph(
            [helper.make_node('Constant', [], ['one'], value_floats=[1.]),
             helper.make_node('Greater', ['X', 'one'], ['cond']),
             helper.make_node('If', ['cond'], ['Z'],
                              then_branch=graph1, else_branch=graph2)],
            'test', [X], [Z])

        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 15)])
        feeds = {'X': numpy.array([-5], dtype=numpy.float32)}

        for rt in ['python', 'python']:  # , 'onnxruntime1']:
            if log:
                print(rt)
            oinf = OnnxInference(model_def, runtime=rt)
            oinf.check_onnx()
            got = oinf.run(feeds)

            inlined, m = onnx_inline_function(
                model_def, {}, verbose=1 if log else 0, fLOG=print)
            self.assertEqual(len(m), 0)
            oinf = OnnxInference(inlined)
            oinf.check_onnx()
            goti = oinf.run(feeds)
            self.assertEqualArray(got['Z'], goti['Z'])

    def test_onnx_inline_subgraph_function(self, log=False):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        one = helper.make_tensor_value_info(
            'one', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101

        graph1 = helper.make_graph([], 'then', [], [X])
        graph2 = helper.make_graph([], 'else', [], [one])

        func_def = helper.make_function(
            'this', 'fct', ['X'], ['Z'], [
                helper.make_node('Constant', [], ['one'], value_floats=[1.]),
                helper.make_node('Greater', ['X', 'one'], ['cond']),
                helper.make_node('If', ['cond'], ['Z'],
                                 then_branch=graph1, else_branch=graph2)],
            opset_imports=[helper.make_operatorsetid('', 15)])

        graph_def = helper.make_graph(
            [helper.make_node('fct', ['X'], ['Z'], domain='this')],
            'test', [X], [Z])

        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 15),
                           helper.make_operatorsetid('this', 1)],
            functions=[func_def])
        feeds = {'X': numpy.array([-5], dtype=numpy.float32)}

        for rt in ['python']:  # , 'onnxruntime1']:
            if log:
                print(rt)
            oinf = OnnxInference(model_def, runtime=rt)
            oinf.check_onnx()
            got = oinf.run(feeds)

            inlined, m = onnx_inline_function(
                model_def, verbose=3 if log else 0, fLOG=print)
            self.assertNotIn('functions {', str(inlined))
            self.assertEqual(len(m), 1)
            oinf = OnnxInference(inlined)
            oinf.check_onnx()
            goti = oinf.run(feeds)
            self.assertEqualArray(got['Z'], goti['Z'])
            self.assertEqualArray(
                got['Z'], numpy.array([1], dtype=numpy.float32))

    def test_onnx_inline_subgraph_function_double(self, log=False):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        out = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101

        func_def_add = helper.make_function(
            'this', 'fctadd', ['input2'], ['output'], [
                helper.make_node('Constant', [], ['one'], value_floats=[1.]),
                helper.make_node('Add', ['input2', 'one'], ['output'])],
            opset_imports=[helper.make_operatorsetid('', 15)])

        graph1 = helper.make_graph(
            [helper.make_node('fctadd', ['input'], ['output'], domain='this')],
            'then', [], [out])
        graph2 = helper.make_graph(
            [helper.make_node('fctadd', ['input'], ['output'], domain='this')],
            'else', [], [out])

        func_def = helper.make_function(
            'this', 'fct', ['input'], ['output'], [
                helper.make_node('Constant', [], ['one'], value_floats=[1.]),
                helper.make_node('Greater', ['input', 'one'], ['cond']),
                helper.make_node('If', ['cond'], ['output'],
                                 then_branch=graph1, else_branch=graph2)],
            opset_imports=[helper.make_operatorsetid('', 15),
                           helper.make_operatorsetid('this', 1)])

        graph_def = helper.make_graph(
            [helper.make_node('fct', ['X'], ['ztmp'], domain='this'),
             helper.make_node('fct', ['ztmp'], ['output'], domain='this')],
            'test', [X], [Z])

        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 15),
                           helper.make_operatorsetid('this', 1)],
            functions=[func_def_add, func_def])
        feeds = {'X': numpy.array([-5], dtype=numpy.float32)}

        for rt in ['python']:  # , 'onnxruntime1']:
            if log:
                print(rt)
            oinf = OnnxInference(model_def, runtime=rt)
            oinf.check_onnx()
            got = oinf.run(feeds)

            inlined, m = onnx_inline_function(
                model_def, verbose=3 if log else 0, fLOG=print)
            self.assertNotIn('functions {', str(inlined))
            self.assertEqual(len(m), 10)
            oinf = OnnxInference(inlined)
            oinf.check_onnx()
            goti = oinf.run(feeds)
            self.assertEqualArray(got['output'], goti['output'])
            self.assertEqualArray(
                got['output'], numpy.array([-3], dtype=numpy.float32))

    def test_onnx_inline_subgraph_function2(self, log=False):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        one = helper.make_tensor_value_info(
            'one', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101

        graph1 = helper.make_graph([], 'then', [], [X])
        graph2 = helper.make_graph([], 'else', [], [one])
        g1 = helper.make_graph(
            [helper.make_node('Greater', ['X', 'one'], ['cond']),
             helper.make_node('If', ['cond'], ['Z'],
                              then_branch=graph1, else_branch=graph2)],
            'test', [], [Z])

        graph1 = helper.make_graph([], 'then', [], [X])
        graph2 = helper.make_graph([], 'else', [], [one])
        g2 = helper.make_graph(
            [helper.make_node('Greater', ['X', 'one'], ['cond']),
             helper.make_node('If', ['cond'], ['Z'],
                              then_branch=graph1, else_branch=graph2)],
            'test', [], [Z])

        func_def = helper.make_function(
            'this', 'fct', ['X'], ['Z'], [
                helper.make_node('Constant', [], ['one'], value_floats=[1.]),
                helper.make_node('Greater', ['X', 'one'], ['cond']),
                helper.make_node('If', ['cond'], ['Z'],
                                 then_branch=g1, else_branch=g2)],
            opset_imports=[helper.make_operatorsetid('', 15)])

        graph_def = helper.make_graph(
            [helper.make_node('fct', ['X'], ['Z'], domain='this')],
            'test', [X], [Z])

        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 15),
                           helper.make_operatorsetid('this', 1)],
            functions=[func_def])
        feeds = {'X': numpy.array([-5], dtype=numpy.float32)}

        for rt in ['python', 'python']:  # , 'onnxruntime1']:
            if log:
                print(rt)
            oinf = OnnxInference(model_def, runtime=rt)
            oinf.check_onnx()
            got = oinf.run(feeds)

            inlined, m = onnx_inline_function(
                model_def, verbose=1 if log else 0, fLOG=print)
            self.assertNotIn('functions {', str(inlined))
            self.assertEqual(len(m), 1)
            oinf = OnnxInference(inlined)
            oinf.check_onnx()
            goti = oinf.run(feeds)
            self.assertEqualArray(got['Z'], goti['Z'])
            self.assertEqualArray(
                got['Z'], numpy.array([1], dtype=numpy.float32))

    def test_onnx_inline_subgraph_function3_fct(self, log=False):
        # subfct
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        one = helper.make_tensor_value_info(
            'one', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101

        graph1 = helper.make_graph([], 'then', [], [X])
        graph2 = helper.make_graph([], 'else', [], [one])
        g1 = helper.make_graph(
            [helper.make_node('Greater', ['X', 'one'], ['cond']),
             helper.make_node('If', ['cond'], ['Z'],
                              then_branch=graph1, else_branch=graph2)],
            'test', [], [Z])

        graph1 = helper.make_graph([], 'then', [], [X])
        graph2 = helper.make_graph([], 'else', [], [one])
        g2 = helper.make_graph(
            [helper.make_node('Greater', ['X', 'one'], ['cond']),
             helper.make_node('If', ['cond'], ['Z'],
                              then_branch=graph1, else_branch=graph2)],
            'test', [], [Z])

        func_def1 = helper.make_function(
            'this', 'subfct', ['X'], ['Z'], [
                helper.make_node('Constant', [], ['one'], value_floats=[1.]),
                helper.make_node('Greater', ['X', 'one'], ['cond']),
                helper.make_node('If', ['cond'], ['Z'],
                                 then_branch=g1, else_branch=g2)],
            opset_imports=[helper.make_operatorsetid('', 15)])

        # mainfct
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101
        one = helper.make_tensor_value_info(
            'one', TensorProto.FLOAT, ['N'])  # pylint: disable=E1101

        gg1 = helper.make_graph(
            [helper.make_node('subfct', ['X'], ['Z'], domain='this')],
            'then', [], [Z])
        gg2 = helper.make_graph(
            [helper.make_node('subfct', ['X'], ['T'], domain='this'),
             helper.make_node('Neg', ['T'], ['Z'])],
            'else', [], [Z])

        func_def2 = helper.make_function(
            'this', 'mainfct', ['X'], ['Z'], [
                helper.make_node('Constant', [], ['one'], value_floats=[1.]),
                helper.make_node('Greater', ['X', 'one'], ['cond']),
                helper.make_node('If', ['cond'], ['Z'],
                                 then_branch=gg1, else_branch=gg2)],
            opset_imports=[helper.make_operatorsetid('', 15)])

        graph_def = helper.make_graph(
            [helper.make_node('mainfct', ['X'], ['Z'], domain='this')],
            'test', [X], [Z])

        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 15),
                           helper.make_operatorsetid('this', 1)],
            functions=[func_def1, func_def2])

        feeds = {'X': numpy.array([-5], dtype=numpy.float32)}

        for rt in ['python']:  # , 'onnxruntime1']:
            if log:
                print(rt)
            oinf = OnnxInference(model_def, runtime=rt)
            oinf.check_onnx()
            got = oinf.run(feeds)

            inlined, m = onnx_inline_function(
                model_def, verbose=1 if log else 0, fLOG=print)
            self.assertNotIn('functions {', str(inlined))
            self.assertEqual(len(m), 5)

            oinf2 = OnnxInference(model_def)
            oinf2.check_onnx()
            got2 = oinf2.run(feeds)
            self.assertEqualArray(got['Z'], got2['Z'])

            oinf3 = OnnxInference(inlined)
            oinf3.check_onnx()
            got3 = oinf3.run(feeds)
            self.assertEqualArray(got['Z'], got3['Z'])

    def common_test_onnx_inline_function_fft(self, subfolder, log=False,
                                             skip_inline=None,
                                             run_validation=True):
        from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException  # pylint: disable=E0611

        def _save_intermediate(name, oinf, save_intermediate):
            if save_intermediate is not None:
                text_base = onnx_simple_text_plot(
                    oinf.obj, recursive=True, indent=False)
                rows_base = text_base.split('\n')
                for k, v in oinf.intermediate_onnx_inference_.items():
                    fn = os.path.join(
                        save_intermediate,
                        f"debug_inter.f-{name}.rt-{oinf.runtime}.r-{k}.onnx")
                    with open(fn, 'wb') as f:
                        f.write(v.obj.SerializeToString())
                    text_new = onnx_simple_text_plot(
                        v.obj, recursive=True, indent=False)
                    rows_new = text_new.split('\n')

                    _, aligned, final = edit_distance_text(rows_base, rows_new)
                    ht = diff2html(rows_base, rows_new, aligned, final,
                                   two_columns=True)
                    with open(fn + ".html", 'w', encoding='utf-8') as f:
                        f.write(ht)

        def _check_run_(name, onx, inverse=False, check=False, runtime='python',
                        save_intermediate=None):
            inplace = True
            if isinstance(check, int):
                verbose = check
            else:
                verbose = 0 if not check else -10
            intermediate = verbose > 0 and runtime != 'python'
            if intermediate:
                inplace = False
            fLOG = print if verbose != 0 else None

            oinf = OnnxInference(onx, runtime=runtime, inplace=inplace)
            names = oinf.input_names

            if names[0] == 'window_length':
                # window function
                inputs = {'window_length': numpy.array([5], dtype=numpy.int64)}
                if 'alpha' in names:
                    inputs['alpha'] = numpy.array([0.56], dtype=numpy.float32)
                    inputs['beta'] = numpy.array([0.54], dtype=numpy.float32)
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG)
                res = got['output']
                self.assertEqual(res.shape, (5, ))
                self.assertEqual(res.dtype, numpy.float32)
                return got

            if names == ['x', 'axis1', 'axis2']:
                # switch axis
                inputs = {'x': numpy.random.randn(3, 4, 5).astype(numpy.float32),
                          'axis1': numpy.array([0], dtype=numpy.int64),
                          'axis2': numpy.array([2], dtype=numpy.int64)}
                try:
                    got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                                   intermediate=intermediate)
                    keepe = None
                except Exception as e:
                    keepe = e
                _save_intermediate(name, oinf, save_intermediate)
                if keepe:
                    raise keepe
                res = got['output']
                self.assertEqual(res.shape, (5, 4, 3))
                self.assertEqualArray(numpy.transpose(
                    inputs['x'], (2, 1, 0)), res)
                return got

            if names == ['x', 'fft_length', 'weights', 'onesided',
                         'inverse', 'normalize']:
                # dft_last_axis
                inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                          'fft_length': numpy.array([5], dtype=numpy.int64),
                          'weights': numpy.array([1, 1, 1, 1, 1], dtype=numpy.float32),
                          'onesided': numpy.array([0], dtype=numpy.int64),
                          'inverse': numpy.array([0], dtype=numpy.int64),
                          'normalize': numpy.array([0], dtype=numpy.int64)}
                ft = numpy.fft.fft(inputs['x'][:, :, :, 0], 5)
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape, (3, 4, 5, 2))
                self.assertEqualArray(
                    res[:, :, :, 0], numpy.real(ft), decimal=4)
                self.assertEqualArray(
                    res[:, :, :, 1], numpy.imag(ft), decimal=4)
                _save_intermediate(name, oinf, save_intermediate)
                return got

            if names == ['x', 'fft_length', 'onesided',
                         'inverse', 'normalize']:
                # dft_last_axis
                inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                          'fft_length': numpy.array([5], dtype=numpy.int64),
                          'onesided': numpy.array([0], dtype=numpy.int64),
                          'inverse': numpy.array([0], dtype=numpy.int64),
                          'normalize': numpy.array([0], dtype=numpy.int64)}
                ft = numpy.fft.fft(inputs['x'][:, :, :, 0], 5)
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape, (3, 4, 5, 2))
                self.assertEqualArray(
                    res[:, :, :, 0], numpy.real(ft), decimal=4)
                self.assertEqualArray(
                    res[:, :, :, 1], numpy.imag(ft), decimal=4)
                if intermediate:
                    inter = oinf.intermediate_onnx_inference_
                    for k, v in inter.items():
                        self.assertEqual(v.runtime, runtime)
                        with open(f"debug_{fct}.{runtime}.{k}.onnx", "wb") as f:
                            f.write(v.obj.SerializeToString())
                _save_intermediate(name, oinf, save_intermediate)
                return got

            if names == ['x', 'fft_length', 'axis', 'weights', 'onesided',
                         'inverse', 'normalize']:
                # dft_inv
                inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                          'fft_length': numpy.array([5], dtype=numpy.int64),
                          'weights': numpy.array([1, 1, 1, 1, 1], dtype=numpy.float32),
                          'axis': numpy.array([2], dtype=numpy.int64),
                          'onesided': numpy.array([0], dtype=numpy.int64),
                          'inverse': numpy.array([0], dtype=numpy.int64),
                          'normalize': numpy.array([0], dtype=numpy.int64)}
                ft = numpy.fft.fft(inputs['x'][:, :, :, 0], 5)
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape, (3, 4, 5, 2))
                self.assertEqualArray(
                    res[:, :, :, 0], numpy.real(ft), decimal=4)
                self.assertEqualArray(
                    res[:, :, :, 1], numpy.imag(ft), decimal=4)
                _save_intermediate(name, oinf, save_intermediate)
                return got

            if names == ['x', 'fft_length', 'axis', 'onesided',
                         'inverse', 'normalize']:
                # dft_inv
                inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                          'fft_length': numpy.array([5], dtype=numpy.int64),
                          'axis': numpy.array([2], dtype=numpy.int64),
                          'onesided': numpy.array([0], dtype=numpy.int64),
                          'inverse': numpy.array([0], dtype=numpy.int64),
                          'normalize': numpy.array([0], dtype=numpy.int64)}
                ft = numpy.fft.fft(inputs['x'][:, :, :, 0], 5)
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape, (3, 4, 5, 2))
                self.assertEqualArray(
                    res[:, :, :, 0], numpy.real(ft), decimal=4)
                self.assertEqualArray(
                    res[:, :, :, 1], numpy.imag(ft), decimal=4)
                _save_intermediate(name, oinf, save_intermediate)
                return got

            if names == ['x', 'fft_length', 'axis', 'inverse', 'onesided']:
                # dft or idft
                inputs = {'x': numpy.random.randn(3, 4, 5, 1).astype(numpy.float32),
                          'fft_length': numpy.array([5], dtype=numpy.int64),
                          'axis': numpy.array([2], dtype=numpy.int64),
                          'inverse': numpy.array([inverse], dtype=numpy.int64),
                          'onesided': numpy.array([0], dtype=numpy.int64)}
                if inverse == 0:  # dft
                    ft = numpy.fft.fft(inputs['x'][:, :, :, 0])
                else:  # idft
                    ft = numpy.fft.ifft(inputs['x'][:, :, :, 0])
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape, (3, 4, 5, 2))
                self.assertEqualArray(
                    res[:, :, :, 0], numpy.real(ft), decimal=4)
                self.assertEqualArray(
                    res[:, :, :, 1], numpy.imag(ft), decimal=4)
                _save_intermediate(name, oinf, save_intermediate)
                return got

            if names == ['x', 'fft_length', 'hop_length', 'n_frames',
                         'window', 'onesided']:
                # stft
                inputs = {'window': numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                                dtype=numpy.float32),
                          'fft_length': numpy.array([6], dtype=numpy.int64),
                          'hop_length': numpy.array([2], dtype=numpy.int64),
                          'n_frames': numpy.array([2], dtype=numpy.int64),
                          'onesided': numpy.array([0], dtype=numpy.int64)}
                inputs['x'] = numpy.random.randn(3, 8, 1).astype(numpy.float32)
                try:
                    import torch
                    p = torch.from_numpy(inputs['x'][:, :, 0])
                    win = torch.from_numpy(inputs['window'])
                    tft = torch.stft(p, n_fft=6, center=False,
                                     win_length=6, window=win,
                                     onesided=False, return_complex=True,
                                     hop_length=2)
                    ft = tft.numpy()
                except ImportError:
                    ft = None
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape, (3, 6, 2, 2))
                if ft is not None:
                    if inputs['hop_length'][0] == 1:
                        self.assertEqual(res.shape[:-1], ft.shape)
                        self.assertEqualArray(
                            res[:, :, :, 0], numpy.real(ft), decimal=4)
                        self.assertEqualArray(
                            res[:, :, :, 1], numpy.imag(ft), decimal=4)
                    else:
                        self.assertEqual(res.shape[:-1], ft.shape)
                        self.assertEqualArray(
                            res[:, :, :, 0], numpy.real(ft), decimal=4)
                        self.assertEqualArray(
                            res[:, :, :, 1], numpy.imag(ft), decimal=4)
                _save_intermediate(name, oinf, save_intermediate)
                return got

            if names == ['x', 'fft_length', 'hop_length', 'window', 'onesided']:
                # istft
                inputs = {'window': numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                                dtype=numpy.float32),
                          'fft_length': numpy.array([6], dtype=numpy.int64),
                          'hop_length': numpy.array([1], dtype=numpy.int64),
                          'onesided': numpy.array([0], dtype=numpy.int64)}
                c = (
                    numpy.random.randn(3, 6, 3).astype(numpy.float32) +
                    numpy.random.randn(3, 6, 3).astype(numpy.float32) * 1j)
                z = numpy.zeros(c.shape + (2, ), dtype=numpy.float32)
                z[:, :, :, 0] = numpy.real(c)
                z[:, :, :, 1] = numpy.imag(c)
                inputs['x'] = z
                try:
                    import torch
                    p = torch.from_numpy(c)
                    win = torch.from_numpy(inputs['window'])
                    tft = torch.istft(p, n_fft=6, center=False,
                                      win_length=6, window=win,
                                      onesided=False, return_complex=True,
                                      hop_length=1)
                    ft = tft.numpy()
                except ImportError:
                    ft = None
                got = oinf.run(inputs, verbose=verbose, fLOG=fLOG,
                               intermediate=intermediate)
                output_name = onx.graph.output[0].name
                res = got[output_name]
                self.assertEqual(res.shape[0], 3)
                self.assertEqual(res.shape, (3, 8, 2))
                if ft is not None:
                    # res = res[:, :, :]
                    self.assertEqual(res.shape[:-1], ft.shape)
                    # The test does not work when the input does not come from stft.
                    # self.assertEqualArray(
                    #     res[:, :, 0], numpy.real(ft), decimal=4)
                    # self.assertEqualArray(
                    #     res[:, :, 1], numpy.imag(ft), decimal=4)
                _save_intermediate(name, oinf, save_intermediate)
                return got

            raise NameError(f"Unable to process {names!r}.")

        def _check_run(name, onx, inverse=False, check=False, runtime='python'):
            t = time.perf_counter()
            res = _check_run_(name, onx, inverse=inverse, check=check,
                              runtime=runtime)
            d = time.perf_counter()
            if log:
                print("TIME  EXEC ", fct, d - t, "inverse=%d" % inverse)
            return res

        def _repare(fct, onx):
            onx.ir_version = 8
            onx = change_input_type(onx, {
                'window_length': TensorProto.INT64,
                'axis1': TensorProto.INT64,
                'axis2': TensorProto.INT64,
                'inverse': TensorProto.INT64,
                'onesided': TensorProto.INT64,
                'normalize': TensorProto.INT64})
            onx = change_subgraph_io_type_shape(onx, {
                'dims1': TensorProto.INT64,
                'dims1_0': TensorProto.INT64,
                'dims2': TensorProto.INT64,
                'dims2_3': TensorProto.INT64,
                'dims3': TensorProto.INT64,
                'dims3_7': TensorProto.INT64})
            onx = onnx_rename_inputs_outputs(onx, {
                'return_val': 'output',
                'norm_67': 'output',
                'final_2': 'output',
                'final_3': 'output'})
            if "_window" in fct:
                onx = change_subgraph_io_type_shape(onx, shape_changes={
                    'output': ['N'],
                    'alpha': [1],
                    'beta': [1],
                    'window_length': [1]})
            else:
                onx = change_subgraph_io_type_shape(onx, shape_changes={
                    'axis1': [1],
                    'axis2': [1],
                    'normalize': [1],
                    'inverse': [1],
                    'onesided': [1],
                    'fft_length': [1],
                    'x': [],
                    'output': []})

            # domain
            domains = set(op.domain for op in onx.opset_import)
            if 'this' not in domains:
                op_set = onx.opset_import.add()  # pylint: disable=E1101
                op_set.domain = 'this'
                op_set.version = 1
            return onx

        def _type_info(name):
            if name in {'x', 'weights', 'window'}:
                return numpy.float32
            if name in {'fft_length', 'axis', 'hop_length', 'n_frames',
                        'axis1', 'axis2'}:
                return numpy.int64
            if name in {'onesided', 'inverse', 'normalize'}:
                return numpy.int64
            if name in {'final_3', 'return_val', 'final', 'output', 'final_2'}:
                return numpy.float32
            raise AssertionError(f"Unexpected name {name!r}.")

        def _validate(fct, model, check_onnx_model=True, path_error=None, inverse=False):
            if check_onnx_model and isinstance(model, ModelProto):
                try:
                    check_onnx(model)
                except Exception as e:
                    rows = []

                    def look(op_type, nodes, seq):
                        for node in nodes:
                            if node.op_type == op_type:
                                rows.append(
                                    "%r - %s" % (
                                        seq,
                                        str(node).replace(" ", "").replace("\n", " ")))
                            for att in node.attribute:
                                if att.type == AttributeProto.GRAPH:
                                    look(op_type, att.g.node,
                                         seq + [node.op_type])

                    look('Constant', model.graph.node, [])
                    for f in model.functions:
                        look('Constant', f.node, ['F', f.name])
                    if path_error is not None:
                        with open(path_error, "wb") as f:
                            f.write(model.SerializeToString())
                    _check_run_(fct, model, inverse=inverse, check=True)
                    raise AssertionError(
                        "Invalid model for function %r due to %r\n---\n%s"
                        "\n---\n%s." % (
                            fct, str(e), "\n".join(rows),
                            str(model))) from e
            if isinstance(model, ModelProto):
                _validate(fct, model.graph, check_onnx_model=check_onnx_model)
                return model
            if isinstance(model, GraphProto):
                self.assertEqual(len(model.output), 1)
                for i in model.input:
                    elem = get_tensor_elem_type(i)
                    if i.name in {'x', 'data', 'alpha', 'beta', 'window', 'weights'}:
                        if elem != TensorProto.FLOAT:
                            raise AssertionError(
                                "Unexpected element type %r for input %r "
                                "in function %r.\n%s" % (
                                    elem, i.name, fct,
                                    onnx_simple_text_plot(
                                        model, recursive=True, raise_exc=False)))
                    else:
                        if elem != TensorProto.INT64:
                            raise AssertionError(
                                "Unexpected element type %r for input %r "
                                "in function %r.\n%s" % (
                                    elem, i.name, fct,
                                    onnx_simple_text_plot(
                                        model, recursive=True, raise_exc=False)))
                for i in model.output:
                    elem = get_tensor_elem_type(i)
                    if i.name in {'output', 'final'}:
                        if elem != TensorProto.FLOAT:
                            raise AssertionError(
                                "Unexpected element type %r for output %r "
                                "in function %r.\n%s" % (
                                    elem, i.name, fct,
                                    onnx_simple_text_plot(
                                        model, recursive=True, raise_exc=False)))
                    else:
                        if elem != TensorProto.INT64:
                            raise AssertionError(
                                "Unexpected element type %r for output %r "
                                "in function %r.\n%s" % (
                                    elem, i.name, fct,
                                    onnx_simple_text_plot(
                                        model, recursive=True, raise_exc=False)))
                return model
            if isinstance(model, FunctionProto):
                self.assertEqual(len(model.output), 1)
                return model
            raise AssertionError(f'Unexpected type {type(model)!r}.')

        def _m2f_shape_fct(name, dtype):
            if dtype == TensorProto.FLOAT:
                return []
            if dtype == TensorProto.INT64:
                return [1]
            raise NotImplementedError(
                f"Unable to process {name!r}, {dtype!r}.")

        temp = get_temp_folder(
            __file__, 'temp_onnx_inline_function_' + subfolder)
        fcts = ["blackman_window", "hamming_window", "hann_window",
                "switch_axes", "dft_last_axis", "dft_inv", "dft",
                "stft", "istft"]

        # first loop, conversion to function
        data = os.path.join(os.path.dirname(__file__), "data", subfolder)
        models = {}
        protos = {}
        for fct in fcts:
            inv_set = [False] if fct != 'dft' else [0, 1]
            for inv in inv_set:
                if log:
                    t = time.perf_counter()
                    print("STEP1 begin", fct)
                onx = load(os.path.join(data, fct + ".onnx"))
                onx = _repare(fct, onx)
                self.assertFalse(isinstance(onx, tuple))
                if run_validation and fct not in {'stft', 'istft'}:
                    _validate(fct, onx, path_error=os.path.join(
                        temp, fct + '.error.check.onnx'))
                try:
                    OnnxInference(onx)
                    use_fct = False
                except (MissingOperatorError, RuntimeError):
                    # The model misses a function.
                    use_fct = True
                if use_fct:
                    fpr, _ = onnx_model_to_function(onx)
                    if run_validation:
                        _validate(fct, fpr)
                    onx = onnx_function_to_model(
                        fpr, protos, type_info=_type_info,
                        shape_fct=_m2f_shape_fct)
                    if run_validation:
                        _validate(fct, onx)

                try:
                    _check_run(fct, onx, inverse=inv)
                except (RuntimeError, AttributeError, NameError) as e:
                    raise AssertionError(
                        "Unable to run fct %r\n---\n%s" % (
                            fct, onnx_simple_text_plot(
                                onx, recursive=True))) from e
                proto, _ = onnx_model_to_function(onx)
                _validate(fct, proto)
                proto.domain = 'this'
                protos[proto.domain, proto.name] = proto
                models[fct] = onx
                if log:
                    print("STEP1 end  ", fct, time.perf_counter() - t)

        rows = []

        def myprint(*args):
            rows.append(' '.join(map(str, args)))

        if log:
            print()

        # second loop, inlining functions
        inlined_models = {}
        atts_def = {'inverse': 0, 'onesided': 0}
        for fct, onx in models.items():
            if run_validation:
                _validate(fct, onx)
            if log:
                t = time.perf_counter()
                print("STEP2 begin", fct)
            del rows[:]
            if skip_inline is None or fct not in skip_inline:
                inline_protos = protos
            else:
                inline_protos = {k: v for k, v in protos.items()
                                 if k not in skip_inline[fct]}

            with open(os.path.join(temp, fct + '.onnx'), 'wb') as f:
                f.write(onx.SerializeToString())
            with open(os.path.join(temp, fct + '.txt'), 'w') as f:
                f.write(helper.printable_graph(onx.graph))
            with open(os.path.join(temp, fct + ".fct.onnx"), "wb") as f:
                f.write(_validate(fct, onnx_model_to_function(
                    onx)[0]).SerializeToString())
            with open(os.path.join(temp, fct + ".fct.att.onnx"), "wb") as f:
                f.write(_validate(
                    fct, onnx_model_to_function(
                        onx, inputs2par=atts_def)[0]).SerializeToString())
            verbose = 4
            if log:
                ti = time.perf_counter()
            try:
                inlined, _ = onnx_inline_function(
                    onx, inline_protos, verbose=verbose, fLOG=myprint)
            except RuntimeError as e:
                raise AssertionError(
                    "Unable to inline function %r\n%s\n#####\n%s" % (
                        fct, "\n".join(rows),
                        onnx_simple_text_plot(onx, recursive=True))) from e
            if run_validation:
                _validate(fct, inlined)
            if skip_inline is not None and fct in skip_inline:
                sx = str(inlined)
                for n in skip_inline[fct]:
                    if f'"{n[1]}"' not in sx:
                        raise AssertionError(
                            "Unable to find %r (fct=%r, inline_protos=%r) "
                            "in\n%s" % (n, fct, list(inline_protos), sx))
            if log:
                print("TIME  INLIN", fct, time.perf_counter() - ti)
            distri = Counter((n.domain, n.op_type)
                             for n in enumerate_onnx_nodes(inlined))
            if ('this', 'dft_last_axis') in distri:
                raise AssertionError(
                    "Inlining went wrong for fct=%r\n----\n%s\n----\n%s" % (
                        fct, pprint.pformat(distri), "\n".join(rows)))
            if len(inlined.functions) > 0:
                if skip_inline is not None and fct in skip_inline:
                    fs_ = set((f.domain, f.name) for f in inlined.functions)
                    inter = fs_ - (skip_inline[fct] & fs_)
                else:
                    inter = inlined.functions
                if len(inter) > 0:
                    raise AssertionError(
                        "Inlining* went wrong for fct=%r\n----\n%s\n----\n%s" % (
                            fct, pprint.pformat(distri), "\n".join(rows)))

            # replaced the skip_inline functions by their inlined versions
            if skip_inline is not None and fct in skip_inline:
                inlined = onnx_replace_functions(
                    inlined,
                    {n: onnx_model_to_function(inlined_models[n[1]],
                                               domain='this')[0]
                     for n in skip_inline[fct]})
                _validate(fct, inlined)

            with self.subTest(fct=fct, inline=True):
                try:
                    _check_run(fct, inlined)
                except (RuntimeError, AttributeError, NameError, IndexError) as e:
                    raise AssertionError(
                        "Unable to run inlined function %r"
                        "\n--#I#--\n--#I#--inlined\n%s"
                        "\n--#N#--\n--#N#--not inlined\n%s"
                        "\n--#L#--\n--#L#--log\n%s" % (
                            fct, onnx_simple_text_plot(
                                inlined, recursive=True, raise_exc=False),
                            onnx_simple_text_plot(
                                onx, recursive=True),
                            "\n".join(map(str, rows)))) from e
            with open(os.path.join(temp, fct + '.inlined.onnx'), 'wb') as f:
                f.write(inlined.SerializeToString())
            inlined_models[fct] = inlined
            with open(os.path.join(temp, fct + '.inlined.txt'), 'w') as f:
                f.write(helper.printable_graph(inlined.graph))
            with open(os.path.join(temp, fct + '.inlined.cpp'), 'w') as f:
                f.write(export2cpp(inlined))
            type_info = {i.name: i.type.tensor_type.elem_type
                         for i in inlined.graph.input}
            type_info.update({i.name: i.type.tensor_type.elem_type
                             for i in inlined.graph.output})
            fct_whole = _validate(fct, onnx_model_to_function(inlined)[0])
            simple_graph = onnx_function_to_model(
                fct_whole, type_info=type_info, as_function=True,
                shape_fct=_m2f_shape_fct)
            if run_validation:
                _validate(fct, simple_graph)
            with open(os.path.join(temp, fct + '.inlined.graph.onnx'), 'wb') as f:
                f.write(simple_graph.SerializeToString())
            if log:
                print("STEP2 end  ", fct, time.perf_counter() - t)

        if log:
            print()

        # third loop, checking inlined functions with onnxruntime
        if not run_validation:
            return
        from onnxruntime import InferenceSession
        from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
            Fail, InvalidArgument, InvalidGraph)
        for fct, onx in inlined_models.items():
            if run_validation:
                _validate(fct, onx)
            if log:
                t = time.perf_counter()
                print("STEP3 begin", fct)
            good = True
            try:
                InferenceSession(onx.SerializeToString())
            except (Fail, InvalidArgument, InvalidGraph) as e:
                good = False
                if log:
                    print("ERROR3", fct, e)
                # print(onnx_simple_text_plot(onx, recursive=True, raise_exc=False))
                with open(os.path.join(temp, fct + '.error.ort.onnx'), 'wb') as f:
                    f.write(onx.SerializeToString())
                with open(os.path.join(temp, fct + '.error.ort.onnx.txt'), 'w') as f:
                    f.write(str(onx))
                warnings.warn(
                    "Unable to load inlined function %r "
                    "with onnxruntime due to %r." % (fct, e))
            if log:
                print("STEP3 end  ", fct, time.perf_counter() - t)

            if not good:
                continue
            try:
                _check_run(fct, onx, runtime="onnxruntime1")
                with open(os.path.join(temp, fct + '.valid.ort.exec.onnx'), 'wb') as f:
                    f.write(onx.SerializeToString())
            except (RuntimeError, AttributeError, NameError, IndexError,
                    RuntimeException) as e:
                with open(os.path.join(temp, fct + '.error.ort.exec.onnx'), 'wb') as f:
                    f.write(onx.SerializeToString())
                if log:
                    print("--------------")
                    print("--------------")
                    _check_run_(fct, onx, runtime="python", check=1)
                    print("--------------")
                    print("--------------")
                    _check_run_(fct, onx, runtime="onnxruntime1", check=1,
                                save_intermediate=temp)
                    print("--------------")
                    print("--------------")
                    raise AssertionError(
                        "Unable to run inlined function with onnxruntime %r"
                        "\n%s" % (
                            fct, onnx_simple_text_plot(
                                onx, recursive=True, raise_exc=False))) from e
                else:
                    warnings.warn(
                        "Unable to run inlined function %r "
                        "with onnxruntime due to %r." % (fct, e))

    def test_onnx_inline_function_fft(self, log=False):
        self.common_test_onnx_inline_function_fft(
            'fft', log=log, run_validation=False)

    @ignore_warnings(UserWarning)
    def test_onnx_inline_function_fft2(self, log=False):
        self.common_test_onnx_inline_function_fft(
            'fft2', log=log, skip_inline={
                'stft': {('this', 'dft')},
                'istft': {('this', 'dft')}})

    def test_replace_initializer(self):
        OnnxMatMul, OnnxSub = loadop('MatMul', 'Sub')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxMatMul('X', numpy.random.randn(2, 100).astype(dtype),
                         op_version=TARGET_OPSET)
        cop2 = OnnxSub(cop, numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['y'])
        model_def = cop2.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})['y']
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = set(n.op_type for n in repl.graph.node)
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = OnnxInference(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run({'X': x})['y']
        self.assertEqualArray(y1, y2)


if __name__ == "__main__":
    # TestOptimOnnxManipulations().test_replace_initializer()
    unittest.main(verbosity=2)
