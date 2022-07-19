# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from scipy.spatial.distance import cdist
from ._op import OpRunBinaryNum
from ._new_ops import OperatorSchema


class CDist(OpRunBinaryNum):

    atts = {'metric': 'sqeuclidean', 'p': 2.}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc,
                                expected_attributes=CDist.atts,
                                **options)

    def _run(self, a, b, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        metric = self.metric.decode('ascii')
        if metric == 'minkowski':
            res = cdist(a, b, metric=metric, p=self.p)
        else:
            res = cdist(a, b, metric=metric)
        # scipy may change the output type
        res = res.astype(a.dtype)
        return (res, )

    def _find_custom_operator_schema(self, op_name):
        if op_name == "CDist":
            return CDistSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def to_python(self, inputs):
        metric = self.metric.decode('ascii')
        if metric == 'minkowski':
            return ('from scipy.spatial.distance import cdist',
                    "return cdist({}, {}, metric='{}', p={})".format(
                        inputs[0], inputs[1], metric, self.p))
        return ('from scipy.spatial.distance import cdist',
                f"return cdist({inputs[0]}, {inputs[1]}, metric='{metric}')")


class CDistSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'CDist')
        self.attributes = CDist.atts
