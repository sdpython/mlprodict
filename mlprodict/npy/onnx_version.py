"""
@file
@brief Identifies a version of a function.

.. versionadded:: 0.6
"""
from collections import namedtuple

_version_ = namedtuple("_version_", ['args', 'kwargs'])


class FctVersion(_version_):
    """
    Identifies a version of a function based on its
    arguments and its parameters.
    """

    def __init__(self, args, kwargs):
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = tuple()
        if not isinstance(args, tuple):
            raise TypeError("args must be a tuple.")
        if not isinstance(kwargs, tuple):
            raise TypeError("kwargs must be a tuple.")
        _version_.__init__(args, kwargs)

    def __repr__(self):
        "usual"
        def cl(s):
            return str(s).replace("<class '", "").replace("'>", "")
        if self.args is None:
            sa = "None"
        else:
            sa = ",".join(map(cl, self.args))
            sa = ("(%s)" % sa) if len(self.args) > 1 else ("(%s,)" % sa)

        return "%s(%s, %s)" % (
            self.__class__.__name__, sa, self.kwargs)

    def __len__(self):
        "Returns the sum of lengths."
        return ((0 if self.args is None else len(self.args)) +
                (0 if self.kwargs is None else len(self.kwargs)))

    def as_tuple(self):
        "Returns a single tuple for the version."
        return ((tuple() if self.args is None else self.args) +
                (tuple() if self.kwargs is None else self.kwargs))

    def as_tuple_with_sep(self, sep):
        "Returns a single tuple for the version."
        return ((tuple() if self.args is None else self.args) +
                (sep, ) +
                (tuple() if self.kwargs is None else self.kwargs))

    def as_string(self):
        "Returns a single stirng identifier."
        val = "_".join(map(str, self.as_tuple_with_sep("_")))
        val = val.replace("<class 'numpy.", "").replace(
            '.', "_").replace("'>", "").replace(" ", "")
        return val.lower()
