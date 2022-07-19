# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun


class CommonRNN(OpRun):

    def __init__(self, onnx_node, expected_attributes=None, desc=None,
                 **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)

        if self.direction in (b"forward", b"reverse"):
            self.num_directions = 1
        elif self.direction == "bidirectional":
            self.num_directions = 2
        else:
            raise RuntimeError(  # pragma: no cover
                f"Unknown direction '{self.direction}'.")

        if len(self.activation_alpha) != self.num_directions:
            raise RuntimeError(  # pragma: no cover
                "activation_alpha must have the same size as num_directions={}".format(
                    self.num_directions))
        if len(self.activation_beta) != self.num_directions:
            raise RuntimeError(  # pragma: no cover
                "activation_beta must have the same size as num_directions={}".format(
                    self.num_directions))

        self.f1 = self.choose_act(
            self.activations[0],
            self.activation_alpha[0] if len(
                self.activation_alpha) > 0 else None,
            self.activation_beta[0] if len(self.activation_beta) > 0 else None)
        if len(self.activations) > 1:
            self.f2 = self.choose_act(
                self.activations[1],
                self.activation_alpha[1] if len(
                    self.activation_alpha) > 1 else None,
                self.activation_beta[1] if len(self.activation_beta) > 1 else None)
        self.nb_outputs = len(onnx_node.output)

    def choose_act(self, name, alpha, beta):
        if name in (b"Tanh", b'tanh', 'tanh', 'Tanh'):
            return self._f_tanh
        if name in (b"Affine", b"affine", 'Affine', 'affine'):
            return lambda x: x * alpha + beta
        raise RuntimeError(  # pragma: no cover
            f"Unknown activation function '{name}'.")

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

    def _run(self, X, W, R, B=None, sequence_lens=None, initial_h=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
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
                 numpy.zeros(2 * hidden_size, dtype=X.dtype))
            h_0 = (initial_h if initial_h is not None else
                   numpy.zeros((batch_size, hidden_size), dtype=X.dtype))

            B = b
            H_0 = h_0
        else:
            raise NotImplementedError(  # pragma: no cover
                "Unsupported value %r for num_directions and operator %r." % (
                    self.num_directions, self.__class__.__name__))

        Y, Y_h = self._step(X, R, B, W, H_0)
        # if self.layout == 1:
        #    #Y = numpy.transpose(Y, [2, 0, 1, 3])
        #    Y_h = Y[:, :, -1, :]

        return (Y, ) if self.nb_outputs == 1 else (Y, Y_h)


class RNN_7(CommonRNN):

    atts = {
        'activation_alpha': [0.],
        'activation_beta': [0.],
        'activations': [b'Tanh', b'Tanh'],
        'clip': [],
        'direction': b'forward',
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
        'activations': [b'Tanh', b'Tanh'],
        'clip': [],
        'direction': b'forward',
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
