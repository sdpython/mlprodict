"""
@file
@brief Identifies a version of a function.

.. versionadded:: 0.6
"""
from collections import namedtuple


class FctVersion(namedtuple("_version_", ['args', 'kwargs'])):
    """
    Identifies a version of a function based on its
    arguments and its parameters.
    """
    __slots__ = ()

    def _check_(self):
        if self.args is not None and not isinstance(self.args, tuple):
            raise TypeError("args must be None or a tuple.")
        if self.kwargs is not None and not isinstance(self.kwargs, tuple):
            raise TypeError("kwargs must None or be a tuple.")

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
