# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class CommonLSTM(OpRun):

    def __init__(self, onnx_node, expected_attributes=None, desc=None,
                 **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)
        self.nb_outputs = len(onnx_node.output)
        self.number_of_gates = 3

    def f(self, x):
        return 1 / (1 + numpy.exp(-x))

    def g(self, x):
        return numpy.tanh(x)

    def h(self, x):
        return numpy.tanh(x)

    def _step(self, X, R, B, W, H_0, C_0, P):
        seq_length = X.shape[0]
        hidden_size = H_0.shape[-1]
        batch_size = X.shape[1]

        Y = numpy.empty(
            [seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []

        [p_i, p_o, p_f] = numpy.split(P, 3)  # pylint: disable=W0632
        H_t = H_0
        C_t = C_0
        for x in numpy.split(X, X.shape[0], axis=0):
            gates = numpy.dot(x, numpy.transpose(W)) + numpy.dot(H_t, numpy.transpose(R)) + numpy.add(
                *numpy.split(B, 2))
            i, o, f, c = numpy.split(gates, 4, -1)  # pylint: disable=W0632
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C

        concatenated = numpy.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.layout == 0:
            Y_h = Y[-1]
        else:
            Y = numpy.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h

    def _run(self, X, W, R, B=None, sequence_lens=None,  # pylint: disable=W0221
             initial_h=None, initial_c=None, P=None,
             attributes=None, verbose=0, fLOG=None):
        number_of_gates = 4
        number_of_peepholes = 3

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
            if initial_c is not None:
                initial_c = numpy.squeeze(initial_c, axis=0)
            if P is not None:
                P = numpy.squeeze(P, axis=0)

            hidden_size = R.shape[-1]
            batch_size = X.shape[1]

            if self.layout != 0:
                X = numpy.swapaxes(X, 0, 1)
            if B is None:
                B = numpy.zeros(2 * number_of_gates *
                                hidden_size, dtype=numpy.float32)
            if P is None:
                P = numpy.zeros(number_of_peepholes *
                                hidden_size, dtype=numpy.float32)
            if initial_h is None:
                initial_h = numpy.zeros(
                    (batch_size, hidden_size), dtype=numpy.float32)
            if initial_c is None:
                initial_c = numpy.zeros(
                    (batch_size, hidden_size), dtype=numpy.float32)
        else:
            raise NotImplementedError(  # pragma: no cover
                "Unsupported value %r for num_directions and operator %r." % (
                    self.num_directions, self.__class__.__name__))

        Y, Y_h = self._step(X, R, B, W, initial_h, initial_c, P)

        return (Y, ) if self.nb_outputs == 1 else (Y, Y_h)


class LSTM(CommonLSTM):

    atts = {
        'activation_alpha': [0.],
        'activation_beta': [0.],
        'activations': [b'Tanh', b'Tanh'],
        'clip': [],
        'direction': b'forward',
        'hidden_size': None,
        'layout': 0,
        'input_forget': 0,
    }

    def __init__(self, onnx_node, desc=None, **options):
        CommonLSTM.__init__(self, onnx_node, desc=desc,
                            expected_attributes=LSTM.atts,
                            **options)
