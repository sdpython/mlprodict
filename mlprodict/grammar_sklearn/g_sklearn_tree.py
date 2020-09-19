# -*- coding: utf-8 -*-
"""
@file
@brief List of converters from scikit-learn model.
"""
import numpy
from .g_sklearn_type_helpers import check_type
from .grammar.gactions import MLActionVar, MLActionCst, MLActionIfElse, MLActionReturn
from .grammar.gactions_tensor import MLActionTensorTake
from .grammar.gactions_num import MLActionTestInf, MLActionTestEqual
from .grammar.gmlactions import MLModel


def sklearn_decision_tree_regressor(model, input_names=None, output_names=None, **kwargs):
    """
    Converts a `DecisionTreeRegressor
    <http://scikit-learn.org/stable/modules/generated/
    sklearn.tree.DecisionTreeRegressor.html>`_
    model into a *grammar* model (semantic graph representation).

    @param      model           scikit-learn model
    @param      input_names     name of the input features
    @param      output_names    name of the output predictions
    @param      kwargs          addition parameter (*with_loop*)
    @return                     graph model

    If *input* is None or *output* is None, default values
    will be given to the outputs
    ``['Prediction', 'Score']`` for the outputs.
    If *input_names* is None, it wil be ``'Features'``.

    Additional parameters:
    - *with_loop*: False by default, *True* not implemented.

    .. note::

        The code to compute on output is
        `here <https://github.com/scikit-learn/scikit-learn/blob/
        ef5cb84a805efbe4bb06516670a9b8c690992bd7/sklearn/tree/_tree.pyx#L806>`_:

        ::

            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

    TODO: improve C code (all leaves are computed and this is unnecessary).
    TODO: create a function tree and an intermediate node and use it here.
    """
    if kwargs.get('with_loop', False):
        raise NotImplementedError(  # pragma: no cover
            "Loop version is not implemented.")
    if output_names is None:
        output_names = ['Prediction', 'Score']
    if input_names is None:
        input_names = 'Features'

    from sklearn.tree import DecisionTreeRegressor
    check_type(model, DecisionTreeRegressor)

    # We convert the tree into arrays.
    # run help(model.tree_).
    lthres = MLActionCst(model.tree_.threshold.ravel().astype(
        numpy.float32), comment="threshold")
    lleft = MLActionCst(model.tree_.children_left.ravel().astype(
        numpy.int32), comment="left")
    lright = MLActionCst(model.tree_.children_right.ravel().astype(
        numpy.int32), comment="right")
    lfeat = MLActionCst(model.tree_.feature.ravel().astype(
        numpy.int32), comment="indfeat")
    lvalue = MLActionCst(model.tree_.value.ravel().astype(
        numpy.float32), comment="value")

    ex = numpy.zeros(model.n_features_, numpy.float32)
    lvar = MLActionVar(ex, input_names)

    lind = MLActionCst(numpy.int32(0), comment="lind")
    th = MLActionTensorTake(lthres, lind)
    m1 = MLActionCst(numpy.int32(-1), comment="m1")

    max_depth = model.tree_.max_depth
    cont = None
    new_lind = None
    for i in range(0, max_depth):
        # Leave ?
        if new_lind is not None:
            lind = new_lind

        le = MLActionTensorTake(lleft, lind)
        lr = MLActionTensorTake(lright, lind)

        di = MLActionTensorTake(lfeat, lind)
        df = MLActionTensorTake(lfeat, di)
        xx = MLActionTensorTake(lvar, df)
        te = MLActionTestInf(xx, th)

        new_lind = MLActionIfElse(te, le, lr, comment="lind{0}".format(i))
        le = MLActionTensorTake(lleft, new_lind)
        th = MLActionTensorTake(lthres, new_lind)

        eq = MLActionTestEqual(m1, le)
        va = MLActionTensorTake(lvalue, new_lind)
        cont = MLActionIfElse(eq, va, th, comment="cont{0}".format(i))

    ret = MLActionReturn(cont)
    return MLModel(ret, output_names, name=DecisionTreeRegressor.__name__)
