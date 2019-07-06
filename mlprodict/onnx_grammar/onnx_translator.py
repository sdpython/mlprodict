"""
@file
@brief One class which visits a syntax tree.
"""
import pprint
import numpy


class CodeTranslator:
    """
    Class which converts a Python function into
    something else. It must implements
    methods *visit* and *depart*.
    """

    def __init__(self, visitor):
        """
        @param      visitor     @see cl CodeNodeVisitor
        """
        self._visitor = visitor

    def export(self, context=None, **kwargs):
        """
        Exports the parsed :epkg:`python` code
        into something.
        """
        raise NotImplementedError("This function should be overwritten.")

    def visit(self, node, info):
        """
        Visits a node.

        @param      node        visited node
        @param      info        info extracted by the visitor
        """
        raise NotImplementedError("This function should be overwritten.")

    def depart(self, node, info):
        """
        Leaves a node.

        @param      node        visited node
        @param      info        info extracted by the visitor
        """
        raise NotImplementedError("This function should be overwritten.")


class OnnxTranslator(CodeTranslator):
    """
    Class which converts a Python function into
    an :epkg:`ONNX` function. It must implements
    methods *visit* and *depart*.
    """
    _binary_operators = {'Add', 'Div', 'Mult', 'Sub'}

    _numpy2onnx_op = {'absolute': 'Abs'}

    def __init__(self, visitor):
        """
        @param      visitor     @see cl CodeNodeVisitor
        """
        CodeTranslator.__init__(self, visitor)
        self._stack = []
        self._code_fct = None

    def _is_stacked(self, name):
        for line in self._stack:
            if line[0] == name:
                return True
        return False

    def _get_last(self, name, info=None):
        if len(self._stack) == 0:
            raise RuntimeError("Stack is empty.")
        last = self._stack[-1]
        if ((isinstance(name, str) and last[0] != name) or
                (isinstance(name, tuple) and last[0] not in name)):
            raise RuntimeError("Last item is not '{}'\n{}\n---\n{}".format(
                name, pprint.pformat(self._stack),
                pprint.pformat(info) if info else ""))
        return last

    def export(self, context=None, format='code'):  # pylint: disable=W0221
        """
        Returns an :epkg:`ONNX` graph or a piece
        of code which could generate the graph.

        @param      format      ``'code'``
        @return                 string or :epkg:`onnx` graph
        """
        if self._code_fct is None:
            raise RuntimeError("No python code was parsed.")
        if context is None:
            context = {}

        def find_onnx_correspondance(fct):
            if isinstance(fct, numpy.ufunc):
                name = fct.__name__
                if name not in OnnxTranslator._numpy2onnx_op:
                    raise RuntimeError("Unable to find a correspondance to '{}' in \n{}".format(
                        name, "\n".join(sorted(OnnxTranslator._numpy2onnx_op))))
                return OnnxTranslator._numpy2onnx_op[name]
            raise RuntimeError(
                "Unable to find a correspondance for function '{}'.".format(fct))

        def write_expression(expr, indent):
            if isinstance(expr, str):
                # an argument
                return ['{}{}'.format(" " * indent * 4, expr)]
            rows = []
            if isinstance(expr, tuple):
                expr = [expr]
            for op, args in expr:
                if op == 'BinOp':
                    opname = args["op"]
                    opon = args["args"]
                    rows.append('{}Onnx{}('.format(" " * indent * 4, opname))
                    for _, expr2 in enumerate(opon):
                        rows.extend(write_expression(expr2, indent + 1))
                        if i < len(opon) - 1:
                            rows[-1] += ","
                    rows.append('{})'.format(" " * indent * 4))
                elif op == 'Call':
                    name = args['name']
                    if name not in context:
                        raise RuntimeError("Unable to find function '{}' in context\n{}".format(
                            name, '\n'.join(sorted(context))))
                    op_conv = find_onnx_correspondance(context[name])
                    opon = args["args"]
                    rows.append('{}Onnx{}('.format(" " * indent * 4, op_conv))
                    opon = opon[1:]
                    for i, expr2 in enumerate(opon):
                        rows.extend(write_expression(expr2, indent + 1))
                        if i < len(opon) - 1:
                            rows[-1] += ","
                    rows.append('{})'.format(" " * indent * 4))
            return rows

        def write_function(node):
            rows = []
            name, args = node
            if name != 'FunctionDef':
                raise RuntimeError("The code being translated should be a single function not "
                                   "'{}'.".format(name))
            fct_name = args['name']
            rows.append("def {}({}):".format(
                fct_name, ', '.join([_[0] for _ in args["args"]])))
            indent = 1
            code = args['code']
            for op, args in code:
                if op == "Assign":
                    name = args['Name']
                    args = args["args"]
                    rows.append("{}{} = (".format(" " * (indent * 4), name))
                    rows.extend(write_expression(args, indent + 1))
                    rows.append("{})".format(" " * (indent * 4)))
                elif op == "Return":
                    args = args["code"]
                    rows.append("{}return (".format(" " * (indent * 4)))
                    rows.extend(write_expression(args, indent + 1))
                    rows.append("{})".format(" " * (indent * 4)))
                else:
                    raise RuntimeError("Unable to process operator '{}'. Make sure it is "
                                       "either an affectation, either a return.".format(op))
            return rows

        rows = write_function(self._code_fct)
        return "\n".join(rows)

    def visit(self, node, info):
        """
        Visits a node.

        @param      node        visited node
        @param      info        info extracted by the visitor
        """
        if 'type' not in info:
            return

        kind = info['type']
        if kind == "Module":
            return
        if kind == "FunctionDef":
            if self._is_stacked('FunctionDef'):
                raise RuntimeError("Nested functions are not allowed.")
            self._stack.append(
                ('FunctionDef', {'args': [], 'code': [], 'name': info['name']}))
            return
        if kind == "arguments":
            _, buf = self._get_last('FunctionDef')
            return
        if kind == "arg":
            return
        if kind == "Assign":
            self._stack.append(('Assign', {'args': []}))
            return
        if kind == 'Name':
            self._get_last(('Assign', 'BinOp', 'Call'))
            return
        if kind == 'BinOp':
            self._stack.append(('BinOp', {'args': []}))
            return
        if kind in OnnxTranslator._binary_operators:
            _, buf = self._get_last('BinOp')
            buf['op'] = kind
            return
        if kind == 'Call':
            self._stack.append(('Call', {'name': info['str'], 'args': []}))
            return
        if kind == 'Attribute':
            _, buf = self._get_last('Call')
            return
        if kind == 'Return':
            self._get_last('FunctionDef')
            self._stack.append(('Return', {'code': []}))
            return
        raise NotImplementedError("Unable to interpret kind '{}'\n{}\n---\n{}".format(
            info.get('type', '?'), pprint.pformat(info),
            pprint.pformat(self._stack)))

    def depart(self, node, info):
        """
        Visits a node.

        @param      node        visited node
        @param      info        info extracted by the visitor
        """
        if 'type' not in info:
            return

        kind = info['type']
        if kind == "arg":
            return
        if kind == "arguments":
            _, buf = self._get_last('FunctionDef')
            for child in info['children']:
                buf['args'].append((child['str'], child['annotation']))
            return
        if kind == "Name":
            op, buf = self._get_last(('Assign', 'BinOp', 'Call'), info)
            if op == 'Assign':
                buf['Name'] = info['str']
                return
            elif op in ('BinOp', 'Call'):
                buf['args'].append(info['str'])
                return
        if kind in OnnxTranslator._binary_operators:
            _, buf = self._get_last('BinOp')
            return
        if kind in ('Call', 'BinOp', 'Assign', 'Return'):
            op, buf = self._get_last(('Call', 'BinOp', 'Assign', 'Return'))
            self._stack.pop()
            opp, parent = self._get_last(
                ('Call', 'BinOp', 'Assign', 'FunctionDef', 'Return'))
            if opp in ('FunctionDef', 'Return'):
                parent['code'].append((op, buf))
            else:
                parent['args'].append((op, buf))
            return
        if kind == 'FunctionDef':
            if len(self._stack) == 1:
                self._code_fct = self._stack[-1]
                self._stack = []
                return
        if kind == 'Module':
            return

        raise NotImplementedError("Unable to interpret kind '{}'\n{}\n---\n{}".format(
            info.get('type', '?'), pprint.pformat(info),
            pprint.pformat(self._stack)))
