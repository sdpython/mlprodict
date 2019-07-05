# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


class ZipMapDictionary(dict):
    """
    Custom dictionary class much faster for this runtime,
    it implements a subet of the same methods.
    """
    __slots__ = ['_rev_keys', '_values', '_mat']

    @staticmethod
    def build_rev_keys(keys):
        res = {}
        for i, k in enumerate(keys):
            res[k] = i
        return res

    def __init__(self, rev_keys, values, mat=None):
        """
        @param      rev_keys        returns by @see me build_rev_keys,
                                    *{keys: column index}*
        @param      values          values
        @param      mat             matrix if values is a row index,
                                    one or two dimensions
        """
        dict.__init__(self)
        self._rev_keys = rev_keys
        self._values = values
        self._mat = mat

    def __getitem__(self, key):
        """
        Returns the item mapped to keys.
        """
        if self._mat is None:
            return self._values[self._rev_keys[key]]
        else:
            return self._mat[self._values, self._rev_keys[key]]

    def __setitem__(self, pos, value):
        raise RuntimeError("Changing an element is not supported.")

    def __len__(self):
        """
        Returns the number of items.
        """
        return len(self._values) if self._mat is None else self._mat.shape[1]

    def __iter__(self):
        for k in self._rev_keys:
            yield k

    def __contains__(self, key):
        return key in self._rev_keys

    def items(self):
        if self._mat is None:
            for k, v in self._rev_keys.items():
                yield k, self._values[v]
        else:
            for k, v in self._rev_keys.items():
                yield k, self._mat[self._values, v]

    def keys(self):
        for k in self._rev_keys.keys():
            yield k

    def values(self):
        if self._mat is None:
            for v in self._values:
                yield v
        else:
            for v in self._mat[self._values]:
                yield v

    def asdict(self):
        res = {}
        for k, v in self.items():
            res[k] = v
        return res


class ArrayZipMapDictionary(list):
    """
    Mocks an array without changing the data it receives.
    Notebooks :ref:`onnxnodetimerst` illustrates the weaknesses
    and the strengths of this class compare to a list
    of dictionaries.

    .. index:: ZipMap
    """

    def __init__(self, rev_keys, mat):
        """
        @param      rev_keys        dictionary *{keys: column index}*
        @param      mat             matrix if values is a row index,
                                    one or two dimensions
        """
        list.__init__(self)
        self._rev_keys = rev_keys
        self._mat = mat

    def __len__(self):
        return self._mat.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        return ZipMapDictionary(self._rev_keys, i, self._mat)

    def __setitem__(self, pos, value):
        raise RuntimeError("Changing an element is not supported.")

    @property
    def values(self):
        """
        Equivalent to ``DataFrame(self).values``.
        """
        return self._mat

    @property
    def columns(self):
        """
        Equivalent to ``DataFrame(self).columns``.
        """
        res = [(v, k) for k, v in self._rev_keys.items()]
        res.sort()
        return [_[1] for _ in res]


class ZipMap(OpRun):

    atts = {'classlabels_int64s': [], 'classlabels_strings': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ZipMap.atts,
                       **options)
        if hasattr(self, 'classlabels_int64s'):
            self.rev_keys_ = ZipMapDictionary.build_rev_keys(
                self.classlabels_int64s)
        elif hasattr(self, 'classlabels_strings'):
            self.rev_keys_ = ZipMapDictionary.build_rev_keys(
                self.classlabels_strings)
        else:
            raise RuntimeError(
                "classlabels_int64s or classlabels_strings must be not empty.")

    def _run(self, x):  # pylint: disable=W0221
        res = ArrayZipMapDictionary(self.rev_keys_, x)
        return (res, )
