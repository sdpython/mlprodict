"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import numpy
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType)


text_alpha_num = [
    ('zero', 0),
    ('one', 1),
    ('two', 2),
    ('three', 3),
    ('four', 4),
    ('five', 5),
    ('six', 6),
    ('seven', 7),
    ('eight', 8),
    ('nine', 9),
    ('dix', 10),
    ('eleven', 11),
    ('twelve', 12),
    ('thirteen', 13),
    ('fourteen', 14),
    ('fifteen', 15),
    ('sixteen', 16),
    ('seventeen', 17),
    ('eighteen', 18),
    ('nineteen', 19),
    ('twenty', 20),
    ('twenty one', 21),
    ('twenty two', 22),
    ('twenty three', 23),
    ('twenty four', 24),
    ('twenty five', 25),
    ('twenty six', 26),
    ('twenty seven', 27),
    ('twenty eight', 28),
    ('twenty nine', 29),
]


def _guess_noshape(obj, shape):
    if isinstance(obj, numpy.ndarray):
        if obj.dtype == numpy.float32:
            return FloatTensorType(shape)  # pragma: no cover
        if obj.dtype == numpy.float64:
            return DoubleTensorType(shape)
        raise NotImplementedError(  # pragma: no cover
            "Unable to process object(1) [{}].".format(obj))
    raise NotImplementedError(  # pragma: no cover
        "Unable to process object(2) [{}].".format(obj))


def _noshapevar(fct):

    def process_itt(itt, Xort):
        if isinstance(itt, tuple):
            return (process_itt(itt[0], Xort), itt[1])

        # name = "V%s_" % str(id(Xort))[:5]
        new_itt = []
        for a, b in itt:
            # shape = [name + str(i) for s in b.shape]
            shape = [None for s in b.shape]
            new_itt.append((a, _guess_noshape(b, shape)))
        return new_itt

    def new_fct(**kwargs):
        X, y, itt, meth, mo, Xort = fct(**kwargs)
        new_itt = process_itt(itt, Xort)
        return X, y, new_itt, meth, mo, Xort
    return new_fct


def _1d_problem(fct):

    def new_fct(**kwargs):
        n_features = kwargs.get('n_features', None)
        if n_features not in (None, 1):
            raise RuntimeError(  # pragma: no cover
                "Misconfiguration: the number of features must not be "
                "specified for a 1D problem.")
        X, y, itt, meth, mo, Xort = fct(**kwargs)
        new_itt = itt  # process_itt(itt, Xort)
        X = X[:, 0]
        return X, y, new_itt, meth, mo, Xort
    return new_fct
