"""
@file
@brief Actions definition.
"""
from .api_extension import AutoAction
from .gactions import MLAction, MLActionFunction


class MLModel(MLActionFunction):
    """
    Base class for every machine learned model
    """

    def __init__(self, input, output_names=None, name=None):
        """
        @param  name             a name which identifies the action
        @param  input           an action which produces the output result
        @param  output_names    names for the outputs
        """
        MLActionFunction.__init__(self, input, name=name)
        self.output_names = output_names

    @property
    def InputNames(self):
        """
        Returns the input names
        """
        vars = self.enumerate_variables()
        res = list(sorted(set(v.name_var for v in vars)))
        if len(res) == 0:
            raise ValueError(  # pragma: no cover
                "At least one variable must be defined.")
        return res

    @property
    def OutputNames(self):
        """
        Returns the output names
        """
        return self.output_names

    @AutoAction.cache
    def _export_json(self, hook=None, result_name=None):
        js = MLAction._export_json(self, hook=hook)
        js.update({"input_names": self.InputNames,
                   "output_names": self.OutputNames})
        return js

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            result_name = "pred"
        return MLActionFunction._export_c(self, hook=hook, result_name=result_name)
