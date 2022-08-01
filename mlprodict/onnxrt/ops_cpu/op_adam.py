# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _apply_adam(r, t, x, g, v, h,
                norm_coefficient, norm_coefficient_post,
                alpha, beta, epsilon):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Update momentum.
    v_new = alpha * v + (1 - alpha) * g_regularized
    # Update second-order momentum.
    h_new = beta * h + (1 - beta) * (g_regularized * g_regularized)
    # Compute element-wise square root.
    h_sqrt = numpy.sqrt(h_new) + epsilon
    # Adjust learning rate.
    r_adjusted = None
    if t > 0:
        # Consider bias correction on momentums.
        r_adjusted = r * numpy.sqrt(1 - beta**t) / (1 - alpha**t)
    else:
        # No bias correction on momentums.
        r_adjusted = r
    # Apply Adam update rule.
    x_new = x - r_adjusted * (v_new / h_sqrt)
    # It's possible to apply regularization in the end.
    x_final = (1 - norm_coefficient_post) * x_new
    return x_final, v_new, h_new


class Adam(OpRun):

    atts = {'alpha': 0.8999999761581421,
            'beta': 0.9990000128746033,
            'epsilon': 9.999999974752427e-07,
            'norm_coefficient': 0.,
            'norm_coefficient_post': 0.}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Adam.atts,
                       **options)

    def _run(self, *data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(data) == 6:
            return self._run1(*data)
        n = (len(data) - 2) // 4
        xs = []
        vs = []
        hs = []
        for i in range(0, n):
            a, b, c = self._run1(*data[:2], data[2 + i],
                                 data[2 + n + i], data[2 + n * 2 + i],
                                 data[2 + n * 3 + i])
            xs.append(a)
            vs.append(b)
            hs.append(c)
        return tuple(xs + vs + hs)

    def _run1(self, r, t, x, g, v, h):  # pylint: disable=W0221
        x_new, v_new, h_new = _apply_adam(
            r, t, x, g, v, h, self.norm_coefficient,
            self.norm_coefficient_post, self.alpha, self.beta, self.epsilon)
        return x_new, v_new, h_new
