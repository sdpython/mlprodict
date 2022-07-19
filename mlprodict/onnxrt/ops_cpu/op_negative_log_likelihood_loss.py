"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _compute_negative_log_likelihood_loss(x, target, weight=None,
                                          reduction=b'mean', ignore_index=None):
    """
    Modified version of `softmaxcrossentropy.py
    <https://github.com/onnx/onnx/blob/main/onnx/backend/
    test/case/node/negativeloglikelihoodloss.py>`_ to handle other type
    than float32.
    """
    input_shape = x.shape
    if len(input_shape) == 1:
        raise RuntimeError(f"Unsupported shape {input_shape!r}.")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        # setting mode='clip' to deal with ignore_index > C or < 0 cases.
        # when the target value is > C or < 0, it doesn't matter which value we are
        # taking in gather_weight, since it will be set to 0 in the following if-block
        # use numpy.int32 to make it compatible with x86 machines
        gather_weight = numpy.take(weight, numpy.array(
            target, dtype=numpy.int32), mode='clip')
        # set `ignore_index`'s loss weight to 0.
        # The loss tensor will be multiplied by this weight tensor,
        # so `ingore_index`'s loss value will be eliminated.
        if ignore_index is not None:
            gather_weight = numpy.where(
                target == ignore_index, 0, gather_weight).astype(dtype=x.dtype)
    elif ignore_index != -1:
        gather_weight = numpy.where(
            target == ignore_index, 0, 1).astype(dtype=x.dtype)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        x = x.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = x.shape[2]
    neg_gather_element_input = numpy.zeros((N, D), dtype=x.dtype)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -x[i][target[i][d]][d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == b'mean':
            loss = loss.sum() / gather_weight.sum()
            return (loss, )

    if reduction == b'mean':
        loss = numpy.mean(loss)
    elif reduction == b'sum':
        loss = numpy.sum(loss)
    return (loss, )


class NegativeLogLikelihoodLoss(OpRun):
    """
    Python runtime for function *NegativeLogLikelihoodLoss*.
    """

    atts = {'reduction': b'mean', 'ignore_index': -1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=NegativeLogLikelihoodLoss.atts,
                       **options)

    def _run(self, x, target, weight=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return _compute_negative_log_likelihood_loss(
            x, target, weight=weight, reduction=self.reduction,  # pylint: disable=E1101
            ignore_index=self.ignore_index)  # pylint: disable=E1101
