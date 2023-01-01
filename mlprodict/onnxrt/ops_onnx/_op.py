"""
@file
@brief Additional methods for the extension of
:epkg:`ReferenceEvaluator`.
"""
from io import BytesIO
import pickle
from typing import Any, Dict
from onnx import NodeProto
from onnx.reference.op_run import OpRun


class OpRunExtended(OpRun):
    """
    Base class to cache C++ implementation based on inputs.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRun.__init__(self, onnx_node, run_params)
        self._cache = {}

    def get_cache_key(self, **kwargs):
        """
        Returns a key mapped to the corresponding C++ implementation.
        """
        b = BytesIO()
        pickle.dump(kwargs, b)
        return b.getvalue()

    def has_cache_key(self, key):
        """
        Tells if a key belongs to the cache.
        """
        return key in self._cache

    def get_cache_impl(self, key):
        """
        Returns the cached implementation for key *key*.
        """
        return self._cache[key]

    def cache_impl(self, key, rt):
        """
        Caches an implementation.
        """
        if key in self._cache:
            raise RuntimeError(f"Key {key!r} is already cached.")
        self._cache[key] = rt
        return rt
