"""
@file
@brief Action definition.
"""
import numpy
from .api_extension import AutoAction
from .gtypes import (
    MLType, MLNumTypeFloat32, MLNumTypeFloat64,
    MLTensor, MLNumTypeInt32, MLNumTypeInt64, MLNumTypeBool)


class MLAction(AutoAction):
    """
    Base class for every action.
    """

    def __init__(self, inputs, output, name, children=None):
        """
        @param      inputs      type of inputs
        @param      output      output type
        @param      name        a name which identifies the action
        @param      children    actions used to compute this one
        """
        if not isinstance(inputs, list):
            raise TypeError(
                'inputs must be a list of MLType.')  # pragma: no cover
        for t in inputs:
            if not isinstance(t, MLType):
                raise TypeError(  # pragma: no cover
                    "Every input must be a MLType not '{0}'.".format(type(t)))
        if not isinstance(output, MLType):
            raise TypeError('output must be of MLType.')  # pragma: no cover
        self.inputs = inputs
        self.output = output
        self.name = name
        self.children = children if children is not None else []
        for child in self.children:
            if not isinstance(child, MLAction):  # pragma: no cover
                raise TypeError("All children must be of type MLAction")

    def execute(self, **kwargs):
        """
        Computes the action. Returns the output.
        """
        # It must be overwritten.
        self.children_results_ = [child.execute(
            **kwargs) for child in self.children]
        for v, tv in zip(self.children_results_, self.inputs):
            tv.validate(v)

    @property
    def ChildrenResults(self):
        """
        Return the last execution results.
        """
        return self.children_results_

    def enumerate_variables(self):
        """
        Enumerates all variables.
        """
        for child in self.children:
            for var in child.enumerate_variables():
                yield var

    def graph_execution(self):
        """
        Returns a formated string which retruns the outputs.
        """
        rows = []
        rows.append("-- BEGIN {0}  {3}                                             id={1} output={2}".format(
            self.name, id(self), self.output._cache, getattr(self, "comment", "")))
        for i, ch in enumerate(self.children):
            gr = ch.graph_execution()
            temp = ["    " + li for li in gr.split("\n")]
            temp[0] = "  {0}-".format(i) + temp[0][4:]
            rows.extend(temp)
        rows.append(
            "-- END {0} -- output={1}".format(self.name, self.output._cache))
        return "\n".join(rows)

    @AutoAction.cache
    def _export_json(self, hook=None, result_name=None):
        val = {"output": self.output._export_json()}
        if self.children:
            val["action"] = dict(name=self.name,
                                 variants=[c._export_json(hook=hook) for c in self.children])
        else:
            val["action"] = dict(name=self.name)
        if self.inputs:
            val["input"] = [i._export_json(hook=hook)
                            for i in self.inputs]
        return val

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(
                "result_name must not be None")  # pragma: no cover
        rows = []
        rows.append("// {0}-{1} - children".format(id(self), self.name))
        names = []
        if self.children:
            for i, c in enumerate(self.children):
                rname = "{0}{1}{2}".format(
                    result_name, getattr(self, "cname", ""), i)
                dc = c._export_c(hook=hook, result_name=rname)
                if not dc['cache']:
                    rows.append(dc['code'])
                names.append(dc['result_name'])
        rows.append("// {0}-{1} - itself".format(id(self), self.name))
        res = "\n".join(rows)
        return {'code': res, 'result_name': result_name, 'child_names': names}


class MLActionCst(MLAction):
    """
    Constant
    """

    def __init__(self, cst, inout_type=None, comment=None):
        """
        @param  cst             constant
        @param  inout_type      type
        @param  comment         comment
        """
        if inout_type is None:
            inout_type = MLActionCst.guess_type(cst)
        MLAction.__init__(self, [], inout_type, "cst")
        inout_type.validate(cst)
        self.cst = cst
        self.comment = comment

    @staticmethod
    def guess_type(value):
        """
        Guesses a type given a value.
        """
        if isinstance(value, numpy.float32):
            return MLNumTypeFloat32()
        if isinstance(value, numpy.float64):
            return MLNumTypeFloat64()
        if isinstance(value, (int, numpy.int32)):
            return MLNumTypeInt32()
        if isinstance(value, (int, numpy.int64)):
            return MLNumTypeInt64()
        if isinstance(value, numpy.ndarray):
            a = numpy.zeros(1, value.dtype)
            t = MLActionCst.guess_type(a[0])
            return MLTensor(t, value.shape)
        raise NotImplementedError(  # pragma: no cover
            "Not implemented for type '{0}'".format(type(value)))

    def execute(self, **kwargs):
        MLAction.execute(self, **kwargs)
        return self.output.validate(self.cst)

    def graph_execution(self):
        if self.comment:
            return "cst: {0} = {1}".format(self.comment, self.cst)
        return "cst: {0}".format(self.cst)

    @AutoAction.cache
    def _export_json(self, hook=None, result_name=None):
        res = {"name": "cst",
               "value": self.output._format_value_json(self.cst, hook=hook)}
        if hasattr(self, "comment"):
            res["comment"] = self.comment
        return res

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError("result_name cannot be None.")  # pragma: no cover
        dc = self.output._export_c(hook='declare', result_name=result_name)
        res = "{0} = {1};".format(
            dc['code'], self.output._format_value_c(self.cst))
        if self.comment:
            res += " // {0}".format(self.comment)
        return {'code': res, 'result_name': result_name}


class MLActionVar(MLActionCst):
    """
    Variable. The constant is only needed to guess the
    variable type.
    """

    def __init__(self, value, name, inout_type=None):
        """
        @param  value       value
        @param  name        variable name
        @param  inout_type  type
        """
        MLActionCst.__init__(self, value, inout_type)
        self.name = "var"
        self.name_var = name

    def execute(self, **kwargs):
        MLAction.execute(self, **kwargs)
        if self.name_var not in kwargs:
            raise KeyError(  # pragma: no cover
                "Unable to find variable name '{0}'".format(self.name_var))
        return self.output.validate(kwargs[self.name_var])

    def enumerate_variables(self):
        """
        Enumerates itself.
        """
        yield self

    def graph_execution(self):
        return "var: {0} = {1} ({2})".format(self.name_var, self.name, self.output._cache)

    @AutoAction.cache
    def _export_json(self, hook=None, result_name=None):
        return {"name": "var", "value": self.name_var}

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(  # pragma: no cover
                "result_name must not be None")
        dc = self.output._export_c(hook='typeref', result_name=result_name)
        res = "{0} = {1};".format(dc['code'], self.name_var)
        return {'code': res, 'result_name': result_name}


class MLActionFunctionCall(MLAction):
    """
    Any function call.
    """

    def __init__(self, name, output, *acts):
        """
        @param  name        function name
        @param  output      type
        @param  *acts       list of arguments
        """
        for act in acts:
            if not isinstance(act, MLAction):
                raise TypeError(  # pragma: no cover
                    "All element of acts must be MLAction not '{0}'.".format(type(act)))
        MLAction.__init__(self, [act.output for act in acts],
                          output, name, children=acts)
        self.cname = 'c'

    def _optional_parameters(self):
        """
        Returns additional parameters to add the function call.
        """
        return None

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(
                "result_name must not be None")  # pragma: no cover
        dcf = MLAction._export_c(self, hook=hook, result_name=result_name)
        rows = [dcf['code']]
        fcall = ", ".join(dcf['child_names'])
        add = self._optional_parameters()  # pylint: disable=E1128
        if add is not None:
            fcall = ", ".join([fcall, add])
        dc = self.output._export_c(hook='declare', result_name=result_name)
        rows.append(dc['code'] + ";")
        ep = self.output._byref_c()
        type_list = "_".join(c.output.CTypeSingle for c in self.children)
        rows.append("{0}_{4}({3}{1}, {2});".format(
            self.name, result_name, fcall, ep, type_list))
        rows.append("// {0}-{1} - done".format(id(self), self.name))
        # Addition printf to debug the C++ code.
        # rows.append('printf("C++ {1} %f\\n", {0});'.format(result_name, self.name))
        res = {'code': "\n".join(rows), 'result_name': dcf['result_name']}
        return res


class MLActionBinary(MLAction):
    """
    Any binary operation.
    """

    def __init__(self, act1, act2, name):
        """
        @param  act1        first element
        @param  act2        second element
        @param  name        operator name
        """
        if not isinstance(act1, MLAction):
            raise TypeError("act1 must be MLAction.")  # pragma: no cover
        if not isinstance(act2, MLAction):
            raise TypeError("act2 must be MLAction.")  # pragma: no cover
        MLAction.__init__(self, [act1.output, act2.output], act2.output, name,
                          children=[act1, act2])

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(
                "result_name must not be None")  # pragma: no cover
        dc = MLAction._export_c(self, hook=hook, result_name=result_name)
        rows = [dc['code']]
        dc2 = self.output._export_c(hook='type')
        op = "{2} {0} = {0}0 {1} {0}1;".format(
            result_name, self.name, dc2['code'])
        rows.append(op)
        rows.append("// {0}-{1} - done".format(id(self), self.name))
        return {'code': "\n".join(rows), 'result_name': result_name}


class MLActionUnary(MLAction):
    """
    Any binary operation.
    """

    def __init__(self, act1, name):
        """
        @param  act1        element
        @param  name        operator name
        """
        if not isinstance(act1, MLAction):
            raise TypeError("act1 must be MLAction.")  # pragma: no cover
        MLAction.__init__(self, [act1.output], act1.output, name,
                          children=[act1])

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(  # pragma: no cover
                "result_name must not be None")
        dc = MLAction._export_c(self, hook=hook, result_name=result_name)
        rows = [dc['code']]
        op = "auto {0} = {1} {0}0;".format(result_name, self.name)
        rows.append(op)
        rows.append("// {0}-{1} - done".format(id(self), self.name))
        return {'code': "\n".join(rows), 'result_name': result_name}


class MLActionConcat(MLActionFunctionCall):
    """
    Concatenate number of arrays into an array.
    """

    def __init__(self, act1, act2):
        """
        @param  act1        first element
        @param  act2        second element
        """
        if not isinstance(act1, MLAction):
            raise TypeError("act1 must be MLAction.")  # pragma: no cover
        if not isinstance(act2, MLAction):
            raise TypeError("act2 must be MLAction.")  # pragma: no cover
        n1 = (1 if isinstance(act1.output, (MLNumTypeFloat32, MLNumTypeFloat64))
              else act1.output.dim[0])
        n2 = (1 if isinstance(act2.output, (MLNumTypeFloat32, MLNumTypeFloat64))
              else act2.output.dim[0])
        MLActionFunctionCall.__init__(self, "concat", MLTensor(
            act1.output.__class__(), (n1 + n2,)), act1, act2)

    def execute(self, **kwargs):
        """
        Concatenation
        """
        MLActionFunctionCall.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(numpy.array(res))


class MLActionCast(MLActionUnary):
    """
    Cast into another type.
    """

    def __init__(self, act1, new_type):
        """
        @param  act1        element
        @param  new_type    new type
        """
        MLActionUnary.__init__(self, act1, "cast")
        self.output = new_type

    def execute(self, **kwargs):
        MLActionUnary.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(self.output.cast(res[0]))

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        raise NotImplementedError(  # pragma: no cover
            "Not enough information to do it here.")


class MLActionIfElse(MLAction):
    """
    Addition
    """

    def __init__(self, cond, act1, act2, check_type=True, comment=None):
        """
        @param  cond        condition
        @param  act1        first action
        @param  ect2        second action
        @param  check_type  check ype
        @param  comment     comment
        """
        if not isinstance(act1, MLAction):
            raise TypeError("act1 must be MLAction.")  # pragma: no cover
        if not isinstance(act2, MLAction):
            raise TypeError("act2 must be MLAction.")  # pragma: no cover
        if not isinstance(cond, MLAction):
            raise TypeError("cond must be MLAction.")  # pragma: no cover
        if not isinstance(cond.output, MLNumTypeBool):
            raise TypeError(  # pragma: no cover
                "No boolean condition {0}".format(type(cond.output)))
        if check_type and type(act1.output) != type(act2.output):
            raise TypeError("Not the same input type {0} != {1}".format(  # pragma: no cover
                type(act1.output), type(act2.output)))
        MLAction.__init__(self, [cond.output, act1.output, act2.output], act2.output, "if",
                          children=[cond, act1, act2])
        self.comment = comment

    def execute(self, **kwargs):
        self.children_results_ = [
            self.children[0].execute(**kwargs), None, None]
        self.inputs[0].validate(self.children_results_[0])
        if self.children_results_[0]:
            self.children_results_[1] = self.children[1].execute(**kwargs)
            self.inputs[1].validate(self.children_results_[1])
            res = self.children_results_[1]
        else:
            self.children_results_[2] = self.children[2].execute(**kwargs)
            self.inputs[2].validate(self.children_results_[2])
            res = self.children_results_[2]
        return self.output.validate(res)

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(
                "result_name must not be None")  # pragma: no cover
        dc = MLAction._export_c(self, hook=hook, result_name=result_name)
        rows = [dc['code']]
        dc2 = self.output._export_c(hook='type')
        op = "{1} {0} = {0}0 ? {0}1 : {0}2;".format(result_name, dc2['code'])
        rows.append(op)
        rows.append("// {0}-{1} - done".format(id(self), self.name))
        return {'code': "\n".join(rows), 'result_name': result_name}


class MLActionReturn(MLAction):
    """
    Returns a results.
    """

    def __init__(self, act):
        """
        @param  act     action to return
        """
        MLAction.__init__(self, [act.output],
                          act.output, "return", children=[act])

    def execute(self, **kwargs):
        MLAction.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(res[0])

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if len(self.children) != 1:
            raise ValueError(
                "Only one result can be returned.")  # pragma: no cover
        if result_name is None:
            raise ValueError(
                "result_name must not be None")  # pragma: no cover
        dc = self.children[0]._export_c(hook=hook, result_name=result_name)
        if not dc['cache']:
            code = dc['code']
        else:
            code = ''

        add = self.output._copy_c(
            result_name, result_name[:-1], hook="typeref")
        code += "\n" + add
        return {'code': code, 'result_name': result_name}


class MLActionFunction(MLActionUnary):
    """
    A function.
    """

    def __init__(self, act, name):
        """
        @param  act     action
        @param  name    name
        """
        if not isinstance(act, MLActionReturn):
            raise NotImplementedError(  # pragma: no cover
                "Last result must be MLActionReturn.")
        MLActionUnary.__init__(self, act, name)

    def execute(self, **kwargs):
        MLActionUnary.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(res[0])

    @AutoAction.cache
    def _export_c(self, hook=None, result_name=None):
        if result_name is None:
            raise ValueError(
                "result_name must not be None")  # pragma: no cover
        if len(self.children) != 1:
            raise ValueError(
                "The function must return one result.")  # pragma: no cover
        if result_name[-1] == '0':
            raise ValueError(  # pragma: no cover
                "result_name '{0}' cannot end with 0.".format(result_name))

        vars = {v.name: v for v in self.enumerate_variables()}
        vars = [_[1] for _ in list(sorted(vars.items()))]
        parameters = ", ".join("{0} {1}".format(
            v.output._export_c(hook='type')['code'], v.name_var) for v in vars)
        typename = self.children[0].output._export_c(
            hook='typeref', result_name=result_name)['code']
        signature = "int {1} ({0}, {2})".format(
            typename, self.name, parameters)
        dc = MLAction._export_c(self, hook=hook, result_name=result_name)
        code = dc['code']
        rows = [signature, "{"]
        rows.extend("    " + line for line in code.split("\n"))
        rows.extend(
            ['    return 0;', "    // {0}-{1} - done".format(id(self), self.name), '}'])
        return {'code': "\n".join(rows), 'result_name': result_name}
