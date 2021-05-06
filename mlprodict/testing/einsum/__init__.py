"""
@file
@brief Shortcut to *testing.einsum*.
"""

from .einsum_bench import einsum_benchmark
from .einsum_fct import einsum
from .einsum_impl import decompose_einsum_equation, apply_einsum_sequence
from .einsum_impl_classes import EinsumSubOp, GraphEinsumSubOp
from .einsum_impl_ext import (
    numpy_extended_dot,
    numpy_extended_dot_python,
    numpy_extended_dot_matrix,
    numpy_diagonal)
