# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun
from ..shape_object import ShapeObject


class CommonRNN(OpRun):

    def __init__(self, onnx_node, expected_attributes=None, desc=None,
                 **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)

        if self.direction in ("forward", "reverse"):
            self.num_directions = 1
        elif self.direction == "bidirectional":
            self.num_directions = 2
        else:
            raise RuntimeError(  # pragma: no cover
                "Unknown direction '{}'.".format(self.direction))

        if len(self.activation_alpha) != self.num_directions:
            raise RuntimeError(  # pragma: no cover
                "activation_alpha must have the same size as num_directions={}".format(
                    self.num_directions))
        if len(self.activation_beta) != self.num_directions:
            raise RuntimeError(  # pragma: no cover
                "activation_beta must have the same size as num_directions={}".format(
                    self.num_directions))

        self.f1 = self.choose_act(self.activations[0],
                                  self.activation_alpha[0],
                                  self.activation_beta[0])
        if len(self.activations) > 1:
            self.f2 = self.choose_act(self.activations[1],
                                      self.activation_alpha[1],
                                      self.activation_beta[1])
        self.nb_outputs = len(onnx_node.output)
        if getattr(self, 'layout', 0) != 0:
            raise NotImplementedError(
                "The runtime is not implemented when layout=%r != 0." % self.layout)

    def choose_act(self, name, alpha, beta):
        if name == b"Tanh":
            return self._f_tanh
        if name == b"Affine":
            return lambda x: x * alpha + beta
        raise RuntimeError(  # pragma: no cover
            "Unknown activation function '{}'.".format(name))

    def _f_tanh(self, x):
        return numpy.tanh(x)

    def _step(self, X, R, B, W, H_0):
        h_list = []
        H_t = H_0
        for x in numpy.split(X, X.shape[0], axis=0):
            H = self.f1(numpy.dot(x, numpy.transpose(W)) +
                        numpy.dot(H_t, numpy.transpose(R)) +
                        numpy.add(*numpy.split(B, 2)))
            h_list.append(H)
            H_t = H
        concatenated = numpy.concatenate(h_list)
        if self.num_directions == 1:
            output = numpy.expand_dims(concatenated, 1)
        return output, h_list[-1]

    def _run(self, X, W, R, B=None, sequence_lens=None, initial_h=None):  # pylint: disable=W0221
        self.num_directions = W.shape[0]

        if self.num_directions == 1:
            R = numpy.squeeze(R, axis=0)
            W = numpy.squeeze(W, axis=0)
            if B is not None:
                B = numpy.squeeze(B, axis=0)
            if sequence_lens is not None:
                sequence_lens = numpy.squeeze(sequence_lens, axis=0)
            if initial_h is not None:
                initial_h = numpy.squeeze(initial_h, axis=0)

            hidden_size = R.shape[-1]
            batch_size = X.shape[1]

            b = (B if B is not None else
                 numpy.zeros(2 * hidden_size, dtype=numpy.float32))
            h_0 = (initial_h if initial_h is not None else
                   numpy.zeros((batch_size, hidden_size), dtype=numpy.float32))

            B = b
            H_0 = h_0
        else:
            raise NotImplementedError()  # pragma: no cover

        Y, Y_h = self._step(X, R, B, W, H_0)
        return (Y, ) if self.nb_outputs == 1 else (Y, Y_h)

    def _infer_shapes(self, X, W, R, B=None, sequence_lens=None, initial_h=None):  # pylint: disable=W0221
        num_directions = W.shape[0]

        if num_directions == 1:
            hidden_size = R[-1]
            batch_size = X[1]
            y_shape = ShapeObject((X[0], num_directions, batch_size, hidden_size),
                                  dtype=X.dtype)
        else:
            raise NotImplementedError()  # pragma: no cover
        if self.nb_outputs == 1:
            return (y_shape, )
        y_h_shape = ShapeObject((num_directions, batch_size, hidden_size),
                                dtype=X.dtype)
        return (y_shape, y_h_shape)


class RNN_7(CommonRNN):

    atts = {
        'activation_alpha': [0.],
        'activation_beta': [0.],
        'activations': ['tanh', 'tanh'],
        'clip': [],
        'direction': 'forward',
        'hidden_size': None,
    }

    def __init__(self, onnx_node, desc=None, **options):
        CommonRNN.__init__(self, onnx_node, desc=desc,
                           expected_attributes=RNN_7.atts,
                           **options)


class RNN_14(CommonRNN):

    atts = {
        'activation_alpha': [0.],
        'activation_beta': [0.],
        'activations': ['tanh', 'tanh'],
        'clip': [],
        'direction': 'forward',
        'hidden_size': None,
        'layout': 0,
    }

    def __init__(self, onnx_node, desc=None, **options):
        CommonRNN.__init__(self, onnx_node, desc=desc,
                           expected_attributes=RNN_14.atts,
                           **options)


if onnx_opset_version() >= 14:
    RNN = RNN_14
else:  # pragma: no cover
    RNN = RNN_7
