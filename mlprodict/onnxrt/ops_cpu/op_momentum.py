# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


def _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta):
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply SG with momentum update rule.
    x_new = x - r * v_new
    return x_new, v_new


class Momentum(OpRun):

    atts = {'alpha': 0,
            'beta': 0,
            'mode': b'standard',
            'norm_coefficient': 0.}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Momentum.atts,
                       **options)

    def _run(self, *data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(data) == 5:
            return self._run1(*data)
        n = (len(data) - 2) // 3
        xs = []
        vs = []
        for i in range(0, n):
            a, b = self._run1(*data[:2], data[2 + i],
                              data[2 + n + i], data[2 + n * 2 + i])
            xs.append(a)
            vs.append(b)
        return tuple(xs + vs)

    def _run1(self, r, t, x, g, v):  # pylint: disable=W0221
        x_new, v_new = _apply_momentum(
            r, t, x, g, v, self.norm_coefficient, self.alpha, self.beta)
        return x_new, v_new
