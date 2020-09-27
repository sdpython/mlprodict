# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class ZipMapDictionary(dict):
    """
    Custom dictionary class much faster for this runtime,
    it implements a subset of the same methods.
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
        if mat is not None:
            if not isinstance(mat, numpy.ndarray):
                raise TypeError(  # pragma: no cover
                    'matrix is expected, got {}.'.format(type(mat)))
            if len(mat.shape) not in (2, 3):
                raise ValueError(  # pragma: no cover
                    "matrix must have two or three dimensions but got {}"
                    ".".format(mat.shape))
        dict.__init__(self)
        self._rev_keys = rev_keys
        self._values = values
        self._mat = mat

    def __getstate__(self):
        """
        For pickle.
        """
        return dict(_rev_keys=self._rev_keys,
                    _values=self._values,
                    _mat=self._mat)

    def __setstate__(self, state):
        """
        For pickle.
        """
        if isinstance(state, tuple):
            state = state[1]
        self._rev_keys = state['_rev_keys']
        self._values = state['_values']
        self._mat = state['_mat']

    def __getitem__(self, key):
        """
        Returns the item mapped to keys.
        """
        if self._mat is None:
            return self._values[self._rev_keys[key]]
        return self._mat[self._values, self._rev_keys[key]]

    def __setitem__(self, pos, value):
        "unused but used by pickle"
        pass

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

    def __str__(self):
        return "ZipMap(%r)" % str(self.asdict())


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
        if mat is not None:
            if not isinstance(mat, numpy.ndarray):
                raise TypeError(  # pragma: no cover
                    'matrix is expected, got {}.'.format(type(mat)))
            if len(mat.shape) not in (2, 3):
                raise ValueError(  # pragma: no cover
                    "matrix must have two or three dimensions but got {}"
                    ".".format(mat.shape))
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
        raise RuntimeError(
            "Changing an element is not supported (pos=[{}]).".format(pos))

    @property
    def values(self):
        """
        Equivalent to ``DataFrame(self).values``.
        """
        if len(self._mat.shape) == 3:
            return self._mat.reshape((self._mat.shape[1], -1))
        return self._mat

    @property
    def columns(self):
        """
        Equivalent to ``DataFrame(self).columns``.
        """
        res = [(v, k) for k, v in self._rev_keys.items()]
        if len(res) == 0:
            if len(self._mat.shape) == 2:
                res = [(i, 'c%d' % i) for i in range(self._mat.shape[1])]
            elif len(self._mat.shape) == 3:
                # multiclass
                res = [(i, 'c%d' % i)
                       for i in range(self._mat.shape[0] * self._mat.shape[2])]
            else:
                raise RuntimeError(  # pragma: no cover
                    "Unable to guess the right number of columns for "
                    "shapes: {}".format(self._mat.shape))
        else:
            res.sort()
        return [_[1] for _ in res]

    @property
    def is_zip_map(self):
        return True

    def __str__(self):
        return 'ZipMaps[%s]' % ', '.join(map(str, self))


class ZipMap(OpRun):
    """
    The class does not output a dictionary as
    specified in :epkg:`ONNX` specifications
    but a @see cl ArrayZipMapDictionary which
    is wrapper on the input so that it does not
    get copied.
    """

    atts = {'classlabels_int64s': [], 'classlabels_strings': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ZipMap.atts,
                       **options)
        if hasattr(self, 'classlabels_int64s') and len(self.classlabels_int64s) > 0:
            self.rev_keys_ = ZipMapDictionary.build_rev_keys(
                self.classlabels_int64s)
        elif hasattr(self, 'classlabels_strings') and len(self.classlabels_strings) > 0:
            self.rev_keys_ = ZipMapDictionary.build_rev_keys(
                self.classlabels_strings)
        else:
            self.rev_keys_ = {}

    def _run(self, x):  # pylint: disable=W0221
        res = ArrayZipMapDictionary(self.rev_keys_, x)
        return (res, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        """
        return (ShapeObject((x[0], ), dtype='map'), )
