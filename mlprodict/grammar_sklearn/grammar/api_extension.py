"""
@file
@brief Implements decorators to extend the API.
"""


class AutoAction:
    """
    Extends the API to automatically look for exporters.
    """

    def _reset_cache(self):
        """
        A same node may appear at different places in the graph.
        It means the output is used twice. However, we don't want to
        include the code to generate that same output twice. We cache it
        and keep some information about it.
        """
        self._cache = None
        for child in self.children:
            child._reset_cache()

    def export(self, lang="json", hook=None, result_name=None):
        """
        Exports into any format.
        The method is looking for one method call
        '_export_<lang>' and calls it if found.

        @param      lang            language
        @param      hook            tweaking parameters
        @param      result_name     the name of the result decided by the parent of this node
        @return                     depends on the language
        """
        self._reset_cache()
        name = "_export_{0}".format(lang)
        if hasattr(self, name):
            try:
                return getattr(self, name)(hook=hook, result_name=result_name)
            except TypeError as e:  # pragma: no cover
                raise TypeError(
                    "Signature of '{0}' is wrong for type '{1}'".format(name, type(self))) from e
        else:
            raise NotImplementedError(  # pragma: no cover
                "No conversion is implemented for lang='{0}' and type='{1}'".format(
                    lang, type(self)))

    @staticmethod
    def cache(func):
        """Caches the result of a function."""

        def func_wrapper(self, hook=None, result_name=None):
            """Wrapper to cache the result of a function."""
            if self._cache is not None:
                c = self._cache.copy()
                c['cache'] = True
                return c
            else:
                ret = func(self, hook=hook, result_name=result_name)
                if not isinstance(ret, dict):
                    raise TypeError(  # pragma: no cover
                        "A dictionary was expected not '{0}'.\nIssue with class '{1}'"
                        "".format(
                            type(ret), type(self)))
                self._cache = ret
                ret = ret.copy()
                ret['cache'] = False
                return ret
        return func_wrapper


class AutoType:
    """
    Extends the API to automatically look for exporters.
    """

    def format_value(self, value, lang="json", hook=None):
        """
        Exports into any format.
        The method is looking for one method call
        '_export_<lang>' and calls it if found.

        @param      value           value to format
        @param      lang            language
        @param      hook            tweaking parameters
        @return                     depends on the language
        """
        name = "_format_value_{0}".format(lang)
        if hasattr(self, name):
            try:
                return getattr(self, name)(value, hook=hook)
            except TypeError as e:  # pragma: no cover
                raise TypeError(
                    "Singature of '{0}' is wrong for type '{1}'".format(name, type(self))) from e
        else:
            raise NotImplementedError(  # pragma: no cover
                "No formatting is implemented for lang='{0}' and type='{1}'".format(
                    lang, type(self)))
