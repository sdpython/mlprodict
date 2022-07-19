"""
@file
@brief Investigate issues happening with float32.
"""
from io import BytesIO
import numpy
from numpy.random import randint
from onnx import ModelProto, FunctionProto, GraphProto, load
from onnx.checker import check_model


class MissingInputError(RuntimeError):
    "Raised when an input is missing."
    pass


def astype_range(arr, dtype=numpy.float32, force=1):
    """
    Computes ranges for every number in an array
    once converted into *float32*. The function returns
    two matrices which produces two numbers
    *a* et *b*, the number rounded to float32
    is in interval :math:`[a, b]`.

    @param      arr     array
    @param      dtype   type to convert to
    @param      force   does something like *[i] +/- force |i - [i]|*
    @return             minimum, maximum
    """
    conv = arr.astype(dtype)
    delta = numpy.abs(arr - conv)
    delta = numpy.maximum(numpy.abs(arr) * 1e-7, delta)
    maxa = (conv + delta * force).astype(dtype)
    mina = (conv - delta * force).astype(dtype)
    return mina, maxa


def enumerate_random_inputs(inputs, n=100, dtype=numpy.float32, force=1):
    """
    Enumerates random matrices.

    @param      inputs      inputs (dictionary)
    @param      n           number of iterations
    @param      dtype       type to convert to
    @param      force       does something like *[i] +/- force |i - [i]|*
    """
    keys = list(inputs)
    ranges = {k: astype_range(v, dtype=dtype, force=force)
              for k, v in inputs.items()}
    for _ in range(n):
        new_inputs = {}
        for k in keys:
            rnd = randint(0, 2, inputs[k].size).reshape(  # pylint: disable=E1101
                inputs[k].shape)  # pylint: disable=E1101
            if rnd.min() == rnd.max() or rnd.max() != 1:
                raise RuntimeError(  # pragma: no cover
                    "Minimum and maximum are equal or maximum is not 1. "
                    "Randomness failed.")
            rnd = rnd.astype(dtype)
            ma1 = ranges[k][0] * rnd
            ma2 = ranges[k][1] * (-(rnd - 1))
            inp = (ma1 + ma2)
            new_inputs[k] = inp
        yield new_inputs


def onnx_shaker(oinf, inputs, output_fct, n=100, dtype=numpy.float32, force=1):
    """
    Shakes a model :epkg:`ONNX`.
    Explores the ranges for every prediction.
    Uses @see fn astype_range

    @param      oinf        object of type @see cl OnnxInference
    @param      inputs      inputs
    @param      output_fct  output function which extracts
                            a single array from the output
    @param      dtype       type to convert to
    @param      force       does something like *[i] +/- force |i - [i]|*
    @return                 ranges for each predictions

    See notebook :ref:`onnxshakerrst` for an example of use.
    """
    results = None
    for i, new_inputs in enumerate(enumerate_random_inputs(
            inputs, n=n, dtype=dtype, force=force)):
        res_ = oinf.run(new_inputs)
        res = output_fct(res_)
        sq = numpy.squeeze(res)
        if len(sq.shape) != 1:
            raise ValueError(  # pragma: no cover
                f"The function only works with shape={sq.shape}")
        if results is None:
            results = numpy.empty((sq.shape[0], n), dtype=sq.dtype)
        results[:, i] = sq

    results.sort(axis=1)
    return results


def check_onnx(model, use_onnx=False, known_results=None,
               path=None):
    """
    Checks consistency of the model.

    :param model: onnx graph
    :param use_onnx: calls `onnx.checker.check_model`
    :param known_results: known results
    :param path: path to a node (through subgraphs)
    """
    if isinstance(model, bytes):
        model = load(BytesIO(model))

    def raise_missing(name, node, p, kn):
        raise MissingInputError(
            "Missing input %r in node type=%r and name=%r "
            "path=%r, known=\n%s\n--ONNX--\n%s" % (
                name, node.op_type, node.name,
                [n.name for n in p], "\n".join(sorted(kn)),
                str(model)))

    if isinstance(model, ModelProto):
        try:
            check_onnx(model.graph, known_results=known_results)
        except MissingInputError as e:
            raise MissingInputError(
                f"Wrong ONNX model\n--ONNX\n{str(model)}") from e
        for f in model.functions:
            check_onnx(f)
        return
    if known_results is None:
        known_results = {}
    else:
        known_results = known_results.copy()
    if isinstance(model, FunctionProto):
        for i in model.input:
            known_results[i] = i
    elif isinstance(model, GraphProto):
        for i in model.input:
            known_results[i.name] = i
        for i in model.initializer:
            known_results[i.name] = i
    else:
        raise TypeError(  # pragma: no cover
            f"Unexpected type {type(model)!r}.")

    if path is None:
        path = []
    else:
        path = path.copy()

    for node in model.node:
        for i in node.input:
            if i == '':
                # optional input
                continue
            if i not in known_results:
                raise_missing(i, node, path + [node], known_results)
            for att in node.attribute:
                if hasattr(att, 'g') and att.g is not None:
                    check_onnx(att.g, use_onnx=use_onnx,
                               known_results=known_results,
                               path=path + [att, node])
        for o in node.output:
            known_results[o] = node

    if use_onnx:
        check_model(model)
