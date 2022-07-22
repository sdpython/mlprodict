"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def softmaxcrossentropy(x, target, weight=None, reduction='mean',
                        ignore_index=None, get_log_prob=None):
    """
    Modified version of `softmaxcrossentropy.py
    <https://github.com/onnx/onnx/blob/main/onnx/backend/
    test/case/node/softmaxcrossentropy.py>`_ to handle other type
    than float32.
    """
    input_shape = x.shape
    if len(input_shape) == 1:
        raise RuntimeError(f"Unsupported shape {input_shape!r}.")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # compute log_softmax
    max_x = numpy.max(x, axis=1, keepdims=True)
    exp_x = numpy.exp(x - max_x)
    p = exp_x / numpy.sum(exp_x, axis=1, keepdims=True)
    inp = numpy.log(p)
    log_prob = None
    if get_log_prob is True:
        log_prob = numpy.copy(inp)

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        gather_weight = numpy.take(
            weight, numpy.array(target, dtype=numpy.int32), mode='clip')
        if ignore_index is not None:
            gather_weight = numpy.where(
                target == ignore_index, 0, gather_weight).astype(dtype=x.dtype)
    elif ignore_index is not None:
        gather_weight = numpy.where(
            target == ignore_index, 0, 1).astype(dtype=x.dtype)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        inp = inp.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = inp.shape[2]
    neg_gather_element_input = numpy.zeros((N, D), dtype=x.dtype)
    for i in range(N):
        for d in range(D):
            if target[i, d] != ignore_index:
                neg_gather_element_input[i, d] = -inp[i, target[i, d], d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == b'mean':
            loss = loss.sum() / gather_weight.sum()
            if get_log_prob is True:
                return loss, log_prob
            return (loss, )

    if reduction == b'mean':
        loss = numpy.mean(loss)
    elif reduction == b'sum':
        loss = numpy.sum(loss)

    if get_log_prob is True:
        return loss, log_prob
    return (loss, )


class SoftmaxCrossEntropyLoss(OpRun):
    """
    Python runtime for function *SoftmaxCrossEntropyLoss*.
    """

    atts = {'reduction': b'mean', 'ignore_index': -1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=SoftmaxCrossEntropyLoss.atts,
                       **options)

    def _run(self, x, target, weight=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        n_outputs = len(self.onnx_node.output)
        return softmaxcrossentropy(
            x, target, weight=weight, reduction=self.reduction,  # pylint: disable=E1101
            ignore_index=self.ignore_index,  # pylint: disable=E1101
            get_log_prob=n_outputs == 2)
