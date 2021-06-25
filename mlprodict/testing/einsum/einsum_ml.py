"""
@file
@brief Functions used to predict the cost of a transposition.
"""
import numpy


_ml_transpose_coefs = {
    'CST_': 0.4720163707200312,
    'begin': 0.0,
    'dbegin': 0.0,
    'dend': 0.0,
    'dim': 0.0,
    'discont': 0.0180766756730043,
    'edit': 0.06940318842803926,
    'end': 0.0,
    'end16': 0.0,
    'end32': 0.0,
    'ibegin16': 0.0,
    'ibegin2': 0.0,
    'ibegin32': 0.0,
    'ibegin4': 0.0,
    'ibegin64': 0.0,
    'ibegin8': 0.04389296884016416,
    'iend16': 0.5316238365817172,
    'iend2': 0.16287259236456927,
    'iend32': 0.0,
    'iend4': 0.0,
    'iend64': 0.0,
    'iend8': 0.0,
    'middle': 1.3381940773605624e-06,
    'rbegin': 0.0,
    'rdiscont': 0.0,
    'redit': 0.18604684802855143,
    'rend': 0.0,
    'rend16': 0.0,
    'rend32': 0.0,
    'rev': 0.42909943168149206,
    'rmiddle': 0.0,
    'rot': 0.22272566615803094,
    'size': 2.8663794075460607e-06}


def _edit_distance(mot1, mot2):
    dist = {(-1, -1): 0}
    if len(mot1) == 0:
        for j, d in enumerate(mot2):
            dist[-1, j] = dist[-1, j - 1] + 1
            dist[j, -1] = dist[j - 1, -1] + 1
    for i, c in enumerate(mot1):
        dist[i, -1] = dist[i - 1, -1] + 1
        dist[-1, i] = dist[-1, i - 1] + 1
        for j, d in enumerate(mot2):
            opt = []
            if (i - 1, j) in dist:
                x = dist[i - 1, j] + 1
                opt.append((x, (i - 1, j)))
            if (i, j - 1) in dist:
                x = dist[i, j - 1] + 1
                opt.append((x, (i, j - 1)))
            if (i - 1, j - 1) in dist:
                x = dist[i - 1, j - 1] + (1 if c != d else 0)
                opt.append((x, (i - 1, j - 1)))
            mi = min(opt)
            dist[i, j] = mi[0]

    return dist[len(mot1) - 1, len(mot2) - 1]


def _is_rotation(perm):
    t = tuple(perm)
    c = list(range(len(perm)))
    for i in range(len(c)):
        for k in range(len(c)):  # pylint: disable=C0200
            c[k] = (k + i) % len(c)
        if t == tuple(c):
            return True
    return False


def _relu(x, origin=0):
    return origin if x < origin else x


def compute_transposition_features(shape, perm):
    """
    Given a shape and a permutation, computes many features
    used to predict the cost of the transposition.

    :param shape: shape
    :param perm: permutation
    :return: dictionary of features

    .. runpython::
        :showcode:

        import pprint
        from mlprodict.testing.einsum.einsum_ml import (
            compute_transposition_features)

        pprint.pprint(
            compute_transposition_features((3, 5, 7), (2, 1, 0)))
    """
    total = numpy.prod(numpy.array(shape, dtype=numpy.int64))

    begin = 1
    dbegin = 0
    for i, p in enumerate(perm):
        if p != i:
            break
        dbegin += 1
        begin *= shape[i]

    end = 1
    dend = 0
    for i in range(len(perm) - 1, -1, -1):
        if perm[i] != i:
            break
        dend += 1
        end *= shape[i]

    dis_cont = 0
    for i in range(1, len(shape)):
        if perm[i] != perm[i - 1] + 1:
            dis_cont += 1

    middle = max(1, int(total / (end * begin)))
    feat = dict(size=total, begin=begin, end=end, middle=middle,
                dim=len(shape), discont=dis_cont)

    for c in [16, 32]:
        feat["end%d" % c] = _relu(end, c)

    keys = list(feat)
    for k in keys:
        if k in {'dim', 'cpu', 'size'}:
            continue
        feat['r%s' % k] = float(feat[k] / total)

    for c in [2, 4, 8, 16, 32, 64]:
        feat["iend%d" % c] = float(end >= c)
        feat["ibegin%d" % c] = float(begin >= c)

    # feat['CST'] = 1
    feat['CST_'] = -1
    feat['dbegin'] = - dbegin
    feat['dend'] = - dend

    keys = list(feat)
    for k in keys:
        if k.startswith('end') or k.startswith('begin'):
            feat[k] = - feat[k]
        elif k.startswith('rend') or k.startswith('rbegin'):
            feat[k] = - feat[k]
        elif k.startswith('iend') or k.startswith('ibegin'):
            feat[k] = - feat[k]
        elif k == "rdiscont":
            feat[k] = - feat[k]

    idp = list(range(len(perm)))
    feat["rot"] = -1 if _is_rotation(perm) else 0
    feat["rev"] = 1 if perm == tuple(idp[::-1]) else 0
    feat["edit"] = _edit_distance(idp, perm)
    feat["redit"] = feat["edit"] / len(idp)
    return feat


def predict_transposition_cost(shape, perm, coefs=None):
    """
    Given a shape and a permutation, predicts the cost of the
    transposition.

    :param shape: shape
    :param perm: permutation
    :param coefs: trained coefficients or None to get
        the default ones
    :return: dictionary of features

    .. runpython::
        :showcode:

        import pprint
        from mlprodict.testing.einsum.einsum_ml import (
            compute_transposition_features)

        pprint.pprint(
            compute_transposition_features((3, 5, 7), (2, 1, 0)))
    """
    if coefs is None:
        coefs = _ml_transpose_coefs
    feat = compute_transposition_features(shape, perm)
    res = 0
    for k, v in feat.items():
        res += v * coefs[k]
    return max(0., res / 1000)
