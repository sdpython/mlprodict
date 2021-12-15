"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
from collections import OrderedDict
import numpy
from cpyquickhelper.numbers import measure_time
from onnxruntime import InferenceSession
from ..onnxrt import OnnxInference


def _random_input(typ, shape, batch):
    if typ == 'tensor(double)':
        dtype = numpy.float64
    elif typ == 'tensor(float)':
        dtype = numpy.float32
    else:
        raise NotImplementedError(
            "Unable to guess dtype from %r." % typ)

    if len(shape) <= 1:
        new_shape = shape
    elif shape[0] is None:
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
        typ = inp.type
        shape = inp.shape
        res[name] = _random_input(typ, shape, batch)
    return res


def latency(model, law='normal', size=1, number=10, repeat=10, max_time=0,
            runtime="onnxruntime", device='cpu'):
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

    if runtime == "onnxruntime":
        sess = InferenceSession(model)
        fct = lambda feeds: sess.run(None, feeds)
        inputs = sess.get_inputs()
    else:
        oinf = OnnxInference(model, runtime=runtime)
        fct = lambda feeds: oinf.run(feeds)
        inputs = oinf.obj.graph.input

    feeds = random_feed(inputs, size)
    res = measure_time(lambda: fct(feeds), number=number, repeat=repeat, context={},
                       max_time=max_time, div_by_number=True)
    return res
