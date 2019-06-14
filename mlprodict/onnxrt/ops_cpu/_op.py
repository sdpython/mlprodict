# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import onnx.defs

_schemas = {
    schema.name: schema for schema in onnx.defs.get_all_schemas_with_history()}


class OpRun:
    """
    Ancestor to all operator in this subfolder.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        """
        @param      onnx_node               :epkg:`onnx` node
        @param      desc                    internal representation
        @param      expected_attributes     expected attributes for this node
        @param      options                 runtime options
        """
        self._provider = 'CPU'
        self.onnx_node = onnx_node
        self.desc = desc
        self._schema = _schemas[onnx_node.op_type]
        if desc is not None:
            if 'atts' in desc:
                for a, b in desc['atts'].items():
                    if not isinstance(b, dict) or 'value' not in b:
                        raise ValueError("Unexpected value {}.".format(b))
                    options[a] = b['value']
        if expected_attributes is not None:
            for a, b in expected_attributes.items():
                if a not in options:
                    if b is None:
                        raise RuntimeError("Parameter '{}' is missing from operator '{}', given {}.".format(
                            a, onnx_node.op_type, list(sorted(options))))
                    else:
                        setattr(self, a, b)
        for k, v in options.items():
            setattr(self, k, v)

        for k, v in self._schema.attributes.items():
            if not hasattr(self, k):
                raise RuntimeError("Attribute '{}' is expected based on ONNX specifications '{}'.".format(
                    k, v))

    def _run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        raise NotImplementedError("This method should be overwritten.")

    def run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        try:
            return self._run(*args, **kwargs)
        except TypeError as e:
            raise TypeError("Issues with types {}.".format(
                ", ".join(str(type(_)) for _ in args))) from e
