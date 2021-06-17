"""
@file
@brief Function to measure the performance of einsum decomposition.
"""
from itertools import permutations
import numpy
from onnx import helper, TensorProto
from ...tools.ort_wrapper import InferenceSession
from ...onnxrt import OnnxInference
from ..bench_helper import measure_time
from .einsum_impl import decompose_einsum_equation, apply_einsum_sequence


def _make_einsum_model(equation, opset=14):  # opset=13, 14, ...
    from skl2onnx.common._topology import OPSET_TO_IR_VERSION  # pylint: disable=E0611,E0001
    inputs = equation.split('->')[0].split(',')

    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid('', opset)],
        ir_version=OPSET_TO_IR_VERSION.get(opset, 7),
        producer_name='mlprodict',
        producer_version='0.1',
        graph=helper.make_graph(
            name='einsum_test',
            inputs=[
                helper.make_tensor_value_info(
                    "X%d" % i, TensorProto.FLOAT, None)  # pylint: disable=E1101
                for i in range(len(inputs))],
            outputs=[
                helper.make_tensor_value_info(
                    "Y", TensorProto.FLOAT, None)],  # pylint: disable=E1101
            nodes=[
                helper.make_node(
                    "Einsum", ["X%d" % i for i in range(len(inputs))], ["Y"],
                    equation=equation)
            ]
        )
    )
    return model


def _make_inputs(equation, shapes):
    inputs = equation.split('->')[0].split(',')
    dims = [len(i) for i in inputs]

    if isinstance(shapes, int):
        N = shapes
        shapes = [(N, ) * le for le in dims]
    else:
        if len(shapes) != len(inputs):
            raise ValueError(
                "Unexpected number of shapes %r with equation %r."
                "" % (shapes, equation))
    inputs = [numpy.random.randn(*sh) for sh in shapes]
    return [i.astype(numpy.float32) for i in inputs]


def einsum_benchmark(equation="abc,cd->abd", shape=30, perm=False,
                     runtime='python', use_tqdm=False,
                     number=5, repeat=5, opset=14):  # opset=13, 14, ...
    """
    Investigates whether or not the decomposing einsum is faster.

    :param equation: einsum equation to test
    :param shape: an integer (all dimension gets the same size) or
        a list of shapes in a string separated with `;`)
    :param perm: check on permutation or all letter permutations
    :param runtime: numpy, python, onnxruntime
    :param use_tqdm: show progress
    :param output: output file (usually a csv file or an excel file),
        it requires pandas
    :param number: usual parameter to measure a function
    :param repeat: usual parameter to measure a function
    :param opset: target opset
    :return: list of dictionaries as an iterator
    """
    scenarios = []
    if (isinstance(shape, list) and
            all(map(lambda t: isinstance(t, int), shape))):
        shape_list = shape
    else:
        shape_list = [shape]

    if perm:
        if equation.lower() != equation:
            raise ValueError(
                "Only equations with lower letters are allowed but equation %r "
                "is not." % equation)
        letters = list(sorted(set(
            c for c in equation if "a" <= c < "z" or "A" <= c < "Z")))
        for p in permutations(letters):
            replace = {d: c for c, d in zip(letters, p)}
            eq = equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            for dec in ['einsum', 'dec']:
                for sh in shape_list:
                    scenarios.append((eq, runtime, dec, sh))
    else:
        for dec in ['einsum', 'dec']:
            for sh in shape_list:
                scenarios.append((equation, runtime, dec, sh))

    if use_tqdm:
        from tqdm import tqdm  # pragma: no cover
        loop = tqdm(scenarios)  # pragma: no cover
    else:
        loop = scenarios

    for eq, rt, dec, sh in loop:
        inputs = _make_inputs(equation, sh)

        if dec == 'dec':
            seq = decompose_einsum_equation(eq, strategy='numpy', clean=True)
        else:
            seq = None

        if rt == 'numpy':
            if dec == 'einsum':
                fct = lambda *x, eq=eq: numpy.einsum(eq, *x, optimize=True)
            else:
                fct = lambda *x, seq=seq: apply_einsum_sequence(seq, *x)
        elif rt == 'onnxruntime':
            if dec == 'einsum':
                onx = _make_einsum_model(equation, opset=opset)
            else:
                onx = seq.to_onnx('Y', *["X%d" % i for i in range(len(inputs))],
                                  opset=opset)
            sess = InferenceSession(
                onx.SerializeToString())  # pylint: disable=W0612
            fct = lambda *x, se=sess: se.run(
                None, {"X%d" % i: v for i, v in enumerate(x)})
        elif rt == 'python':
            if dec == 'einsum':
                onx = _make_einsum_model(equation, opset=opset)
            else:
                onx = seq.to_onnx('Y', *["X%d" % i for i in range(len(inputs))],
                                  opset=opset)
            oinf = OnnxInference(onx)  # pylint: disable=W0612
            fct = lambda *x, oi=oinf: oi.run(
                {"X%d" % i: v for i, v in enumerate(x)})
        else:
            raise ValueError("Unexpected runtime %r." % rt)

        res = measure_time(fct, *inputs, repeat=repeat, number=number)
        res['rt'] = rt
        res['dec'] = dec
        res['eq'] = eq
        res['shapes'] = ";".join(
            map(str, [m.shape for m in inputs])).replace(' ', '')
        yield res
