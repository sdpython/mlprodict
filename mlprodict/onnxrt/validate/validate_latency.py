"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
from collections import OrderedDict
import json
import numpy
from onnx import TensorProto
from pandas import DataFrame
from .. import OnnxInference
from ..ops_whole.session import OnnxWholeSession


def _random_input(typ, shape, batch):
    if typ in ('tensor(double)', TensorProto.DOUBLE):  # pylint: disable=E1101
        dtype = numpy.float64
    elif typ in ('tensor(float)', TensorProto.FLOAT):  # pylint: disable=E1101
        dtype = numpy.float32
    else:
        raise NotImplementedError(
            f"Unable to guess dtype from {typ!r}.")

    if len(shape) <= 1:
        new_shape = shape
    elif shape[0] in (None, 0):
        new_shape = tuple([batch] + list(shape[1:]))
    else:
        new_shape = shape
    return numpy.random.randn(*new_shape).astype(dtype)


def random_feed(inputs, batch=10, empty_dimension=1):
    """
    Creates a dictionary of random inputs.

    :param batch: dimension to use as batch dimension if unknown
    :param empty_dimension: if a dimension is null, replaces it by this value
    :return: dictionary
    """
    res = OrderedDict()
    for inp in inputs:
        name = inp.name
        if hasattr(inp.type, 'tensor_type'):
            typ = inp.type.tensor_type.elem_type
            shape = tuple(getattr(d, 'dim_value', batch)
                          for d in inp.type.tensor_type.shape.dim)
            shape = (shape[0], ) + tuple(
                b if b > 0 else empty_dimension for b in shape[1:])
        else:
            typ = inp.type
            shape = inp.shape
        res[name] = _random_input(typ, shape, batch)
    return res


def latency(model, law='normal', size=1, number=10, repeat=10, max_time=0,
            runtime="onnxruntime", device='cpu', profiling=None):
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
    :param profiling: if True, profile the execution of every
        node, if can be sorted by name or type,
        the value for this parameter should e in `(None, 'name', 'type')`,
    :return: dictionary or a tuple (dictionary, dataframe)
        if the profiling is enable

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
    from cpyquickhelper.numbers import measure_time  # delayed import

    if isinstance(model, str) and not os.path.exists(model):
        raise FileNotFoundError(  # pragma: no cover
            f"Unable to find model {model!r}.")
    if profiling not in (None, '', 'name', 'type'):
        raise ValueError(
            f"Unexpected value for profiling: {profiling!r}.")
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
            f"Only law='normal' is supported, not {law!r}.")

    if device in ('cpu', 'CPUExecutionProviders'):
        providers = ['CPUExecutionProviders']
    elif device in ('cuda:0', 'CUDAExecutionProviders'):
        if runtime != 'onnxruntime':
            raise NotImplementedError(  # pragma: no cover
                "Only runtime 'onnxruntime' supports this device or provider "
                "%r." % device)
        providers = ['CUDAExecutionProviders']
    elif ',' in device:
        from onnxruntime import get_all_providers  # delayed import
        if runtime != 'onnxruntime':
            raise NotImplementedError(  # pragma: no cover
                "Only runtime 'onnxruntime' supports this device or provider "
                "%r." % device)
        providers = device.split(',')
        allp = set(get_all_providers())
        for p in providers:
            if p not in allp:
                raise ValueError(
                    f"One device or provider {p!r} is not supported among {allp!r}.")
    else:
        raise ValueError(  # pragma no cover
            f"Device {device!r} not supported.")

    if runtime in ("onnxruntime", "onnxruntime-cuda"):
        from onnxruntime import InferenceSession, SessionOptions  # delayed import
        providers = ['CPUExecutionProvider']
        if runtime == "onnxruntime-cuda":
            providers = ['CUDAExecutionProvider'] + providers
        if profiling in ('name', 'type'):
            so = SessionOptions()
            so.enable_profiling = True
            sess = InferenceSession(
                model, sess_options=so, providers=providers)
        else:
            sess = InferenceSession(model, providers=providers)
        fct = lambda feeds: sess.run(None, feeds)
        inputs = sess.get_inputs()
    else:
        if profiling in ('name', 'type'):
            runtime_options = {"enable_profiling": True}
            if runtime != 'onnxruntime1':
                raise NotImplementedError(  # pragma: no cover
                    f"Profiling is not implemented for runtime={runtime!r}.")
        else:
            runtime_options = None
        oinf = OnnxInference(model, runtime=runtime,
                             runtime_options=runtime_options)
        fct = lambda feeds: oinf.run(feeds)
        inputs = oinf.obj.graph.input

    feeds = random_feed(inputs, size)
    res = measure_time(
        lambda: fct(feeds), number=number, repeat=repeat, context={},
        max_time=max_time, div_by_number=True)
    for k, v in feeds.items():
        res[f"shape({k})"] = "x".join(map(str, v.shape))
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
            gr = df[['dur', "args_op_name", "name"]].groupby(
                ["args_op_name", "name"]).sum().sort_values('dur')
        else:
            gr = df[['dur', "args_op_name"]].groupby(
                "args_op_name").sum().sort_values('dur')
        return res, gr

    return res
