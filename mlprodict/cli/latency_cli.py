"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
from io import StringIO
from collections import OrderedDict
import json
import numpy
from onnx import TensorProto
from pandas import DataFrame
from cpyquickhelper.numbers import measure_time
from onnxruntime import InferenceSession, SessionOptions
from ..onnxrt import OnnxInference
from ..onnxrt.ops_whole.session import OnnxWholeSession


def _random_input(typ, shape, batch):
    if typ in ('tensor(double)', TensorProto.DOUBLE):  # pylint: disable=E1101
        dtype = numpy.float64
    elif typ in ('tensor(float)', TensorProto.FLOAT):  # pylint: disable=E1101
        dtype = numpy.float32
    else:
        raise NotImplementedError(
            "Unable to guess dtype from %r." % typ)

    if len(shape) <= 1:
        new_shape = shape
    elif shape[0] in (None, 0):
        new_shape = tuple([batch] + list(shape[1:]))
    else:
        new_shape = shape
    return numpy.random.randn(*new_shape).astype(dtype)


def random_feed(inputs, batch=10):
    """
    Creates a dictionary of random inputs.

    :param batch: dimension to use as batch dimension if unknown
    :return: dictionary
    """
    res = OrderedDict()
    for inp in inputs:
        name = inp.name
        if hasattr(inp.type, 'tensor_type'):
            typ = inp.type.tensor_type.elem_type
            shape = tuple(getattr(d, 'dim_value', 0)
                          for d in inp.type.tensor_type.shape.dim)
        else:
            typ = inp.type
            shape = inp.shape
        res[name] = _random_input(typ, shape, batch)
    return res


def latency(model, law='normal', size=1, number=10, repeat=10, max_time=0,
            runtime="onnxruntime", device='cpu', fmt=None,
            profiling=None, profile_output='profiling.csv'):
    """
    Measures the latency of a model (python API).

    :param model: ONNX graph
    :param law: random law used to generate fake inputs
    :param size: batch size, it replaces the first dimension
        of every input if it is left unknown
    :param number: number of calls to measure
    :param repeat: number of times to repeat the experiment
    :param max_time: if it is > 0, it runs as many time during
        that period of time
    :param runtime: available runtime
    :param device: device, `cpu`, `cuda:0`
    :param fmt: None or `csv`, it then
        returns a string formatted like a csv file
    :param profiling: if True, profile the execution of every
        node, if can be by name or type.
    :param profile_output: output name for the profiling
        if profiling is specified

    .. cmdref::
        :title: Measures model latency
        :cmd: -m mlprodict latency --help
        :lid: l-cmd-latency

        The command generates random inputs and call many times the
        model on these inputs. It returns the processing time for one
        iteration.

        Example::

            python -m mlprodict latency --model "model.onnx"
    """
    if not os.path.exists(model):
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find model %r." % model)
    if profiling not in (None, '', 'name', 'type'):
        raise ValueError(
            "Unexpected value for profiling: %r." % profiling)
    size = int(size)
    number = int(number)
    repeat = int(repeat)
    if max_time in (None, 0, ""):
        max_time = None
    else:
        max_time = float(max_time)
        if max_time <= 0:
            max_time = None

    if law != "normal":
        raise ValueError(
            "Only law='normal' is supported, not %r." % law)

    if device != 'cpu':
        raise NotImplementedError(  # pragma no cover
            "Only support cpu for now not %r." % device)

    if profiling in ('name', 'type') and profile_output in (None, ''):
        raise ValueError(  # pragma: no cover
            'profiling is enabled but profile_output is wrong (%r).'
            '' % profile_output)

    if runtime == "onnxruntime":
        if profiling in ('name', 'type'):
            so = SessionOptions()
            so.enable_profiling = True
            sess = InferenceSession(model, sess_options=so)
        else:
            sess = InferenceSession(model)
        fct = lambda feeds: sess.run(None, feeds)
        inputs = sess.get_inputs()
    else:
        if profiling in ('name', 'type'):
            runtime_options = {"enable_profiling": True}
            if runtime != 'onnxruntime1':
                raise NotImplementedError(  # pragma: no cover
                    "Profiling is not implemented for runtime=%r." % runtime)
        else:
            runtime_options = None
        oinf = OnnxInference(model, runtime=runtime,
                             runtime_options=runtime_options)
        fct = lambda feeds: oinf.run(feeds)
        inputs = oinf.obj.graph.input

    feeds = random_feed(inputs, size)
    res = measure_time(lambda: fct(feeds), number=number, repeat=repeat, context={},
                       max_time=max_time, div_by_number=True)
    if profiling in ('name', 'type'):
        if runtime == 'onnxruntime':
            profile_name = sess.end_profiling()
            with open(profile_name, 'r', encoding='utf-8') as f:
                js = json.load(f)
            js = OnnxWholeSession.process_profiling(js)
            df = DataFrame(js)
        else:
            df = oinf.get_profiling(as_df=True)
        if profiling == 'name':
            gr = df[['dur', "name"]].groupby(
                "name").sum().sort_values('dur')
        else:
            gr = df[['dur', "args_op_name"]].groupby(
                "args_op_name").sum().sort_values('dur')
        gr.reset_index(drop=False).to_csv(profile_output, index=False)

    if fmt == 'csv':
        st = StringIO()
        df = DataFrame([res])
        df.to_csv(st, index=False)
        return st.getvalue()
    if fmt in (None, ''):
        return res
    raise ValueError(  # pragma: no cover
        "Unexpected value for fmt: %r." % fmt)
