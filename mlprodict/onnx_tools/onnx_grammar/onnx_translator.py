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
        raise NotImplementedError(  # pragma: no cover
            "This function should be overwritten.")

    def visit(self, node, info):
        """
        Visits a node.

        @param      node        visited node
        @param      info        info extracted by the visitor
        """
        raise NotImplementedError(  # pragma: no cover
            "This function should be overwritten.")

    def depart(self, node, info):
        """
        Leaves a node.

        @param      node        visited node
        @param      info        info extracted by the visitor
        """
        raise NotImplementedError(  # pragma: no cover
            "This function should be overwritten.")


class OnnxTranslator(CodeTranslator):
    """
    Class which converts a Python function into
    an :epkg:`ONNX` function. It must implements
    methods *visit* and *depart*.
    """
    _binary_operators = {
        'Add': 'Add', 'Div': 'Div',
        'Mult': 'Mul', 'Sub': 'Sub',
        'Pow': 'Pow', 'MatMult': 'MatMul',
    }

    _unary_operators = {
        'Sub': 'Neg',
    }

    _numpy2onnx_op = {
        'absolute': 'Abs',
        'cos': 'Cos',
        'exp': 'Exp',
        'power': 'Pow',
        'transpose': 'Transpose',
        'sin': 'Sin',
        # complex function
        'inner': 'inner',
    }

    _parameter_mapping = {
        'Transpose': {'axes': 'perm'}
    }

    class Parameter:
        """
        Holds parameter information.
        """

        def __init__(self, name, value=('#NODEFAULT#', ), annotation=None):
            """
            @param      name        parameter name
            @param      value       parameter value
            """
            self.name = name
            self.value = value
            self.annotation = annotation

        @staticmethod
        def format_value(value):
            """
            Returns a formatted value in python code.
            """
            if isinstance(value, str):
                return '"{}"'.format(value.replace('"', '\\"').replace('\\', '\\\\'))
            if isinstance(value, list):
                return "[{}]".format(", ".join(map(OnnxTranslator.Parameter.format_value, value)))
            if isinstance(value, tuple):
                if value == ('#NODEFAULT#', ):
                    return None
                return "({})".format(", ".join(map(OnnxTranslator.Parameter.format_value, value)))
            return str(value)

        @property
        def formatted_value(self):
            """
            Returns a formatted value in python code.
            """
            return OnnxTranslator.Parameter.format_value(self.value)

        def __str__(self):
            """
            Into python syntax.
            """
            rows = [self.name]
            if self.value != ('#NODEFAULT#', ):
                rows.append('=')
                rows.append(self.formatted_value)
            return ''.join(rows)

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
            raise RuntimeError("Stack is empty.")  # pragma: no cover
        last = self._stack[-1]
        if ((isinstance(name, str) and last[0] != name) or
                (isinstance(name, tuple) and last[0] not in name)):
            raise RuntimeError(  # pragma: no cover
                "Last item is not '{}'\n{}\n---\n{}".format(
                    name, pprint.pformat(self._stack),
                    pprint.pformat(info) if info else ""))
        return last

    def make_msg(self, info):
        """
        Make a message with line and column information.
        """
        lineno = '?'
        col_offset = '?'
        if isinstance(info, dict):
            if 'node' in info:
                node = info['node']
                lineno = node.lineno
                col_offset = node.col_offset
            else:
                if 'lineno' in info:
                    lineno = info['lineno']
                if 'col_offset' in info:
                    col_offset = info['col_offset']
        else:
            if hasattr(info, 'lineno'):
                lineno = info.lineno
            if hasattr(info, 'col_offset'):
                col_offset = info.col_offset

        return "line {}, col {}".format(lineno, col_offset)

    def export(self, context=None, format='code',  # pylint: disable=W0221
               output_names=None):
        """
        Returns an :epkg:`ONNX` graph or a piece
        of code which could generate the graph.

        @param      context         function used in the function code
        @param      format          ``'code'``
        @param      output_names    add code in the final function
                                    to overwrite the names of the
                                    outputs in the :epkg:`ONNX` graph
        @return                     string or :epkg:`onnx` graph

        This method is used in function @see fn translate_fct2onnx.
        An example of code can be found there.
        """
        if self._code_fct is None:
            raise RuntimeError(  # pragma: no cover
                "No python code was parsed.")
        if context is None:
            context = {}

        def find_onnx_correspondance(fct, info):
            if isinstance(fct, numpy.ufunc):
                name = fct.__name__
            elif callable(fct) and getattr(fct, '__module__', '') in (
                    'numpy', 'numpy.core.fromnumeric'):
                name = fct.__name__
            elif callable(fct) and fct.__name__.startswith("py_"):
                return fct
            else:
                name = None
            if name is not None and name not in OnnxTranslator._numpy2onnx_op:
                raise RuntimeError(  # pragma: no cover
                    "Unable to find a correspondance to '{}' at {} in \n{}".format(
                        name, self.make_msg(info),
                        "\n".join(sorted(OnnxTranslator._numpy2onnx_op))))
            if name is not None:
                return OnnxTranslator._numpy2onnx_op[name]
            if isinstance(fct, str):
                return fct
            raise RuntimeError(  # pragma: no cover
                "Unable to find a correspondance for function name '{}' in module '{}', "
                "'{}' (type {}) at {}.".format(
                    name, getattr(fct, '__module__', ''),
                    fct, type(fct), self.make_msg(info)))

        def write_expression(stack_fct_used, expr, indent, parameter_mapping=None):
            if isinstance(expr, str):
                # an argument
                return ['{}{}'.format(" " * indent * 4, expr)]
            if isinstance(expr, (int, float)):
                # an argument
                return ['{}{}'.format(" " * indent * 4, expr)]
            if isinstance(expr, OnnxTranslator.Parameter):
                if parameter_mapping is None:
                    name = expr.name
                else:
                    name = parameter_mapping.get(expr.name, expr.name)
                return ["{}{}={}".format(" " * indent * 4, name,
                                         expr.formatted_value)]
            rows = []
            if isinstance(expr, tuple):
                expr = [expr]
            for op, args in expr:
                if op == 'BinOp':
                    opname = args["op"]
                    opon = args["args"]
                    onnx_name = OnnxTranslator._binary_operators[opname]
                    rows.append(
                        '{}Onnx{}('.format(" " * indent * 4, onnx_name))
                    for expr2 in opon:
                        sexpr2 = write_expression(
                            stack_fct_used, expr2, indent + 1)
                        if any(filter(lambda s: 'op_version="op_version"' in s, sexpr2)):
                            continue  # pragma: no cover
                        rows.extend(sexpr2)
                        rows[-1] += ","
                    rows.append('{}op_version=op_version'.format(
                        " " * (indent + 1) * 4))
                    rows.append('{})'.format(" " * indent * 4))
                elif op == 'UnaryOp':
                    opname = args["op"]
                    opon = args["args"]
                    onnx_name = OnnxTranslator._unary_operators[opname]
                    rows.append(
                        '{}Onnx{}('.format(" " * indent * 4, onnx_name))
                    for expr2 in opon:
                        sexpr2 = write_expression(
                            stack_fct_used, expr2, indent + 1)
                        if any(filter(lambda s: 'op_version="op_version"' in s, sexpr2)):
                            continue
                        rows.extend(sexpr2)
                        rows[-1] += ","
                    rows.append('{}op_version=op_version'.format(
                        " " * (indent + 1) * 4))
                    rows.append('{})'.format(" " * indent * 4))
                elif op == 'Call':
                    name = args['name']
                    if name.startswith("onnx_"):
                        raise RuntimeError("The code must not use a function prefixed by 'onnx_' (%s). "
                                           "It indicates that function manipulate ONNX node and "
                                           "the fonction to convert must only deal with arrays." % name)
                    if name not in context:
                        raise RuntimeError(
                            "Unable to find function '{}' at {} in context\n{}\n--\n{}".format(
                                name, self.make_msg(args),
                                '\n'.join(sorted(context)),
                                pprint.pformat(args)))
                    op_conv = find_onnx_correspondance(context[name], args)
                    if callable(op_conv) and op_conv.__name__.startswith('py_'):
                        rows.append(
                            '{}{}('.format(" " * indent * 4, op_conv.__name__))
                    elif callable(op_conv) and op_conv.__name__.startswith('onnx_'):
                        stack_fct_used.append(op_conv.__name__)
                        rows.append(
                            '{}{}('.format(" " * indent * 4, op_conv))
                    else:
                        prefix = "onnx_" if 'a' <= op_conv[0] <= 'z' else 'Onnx'
                        if prefix == "onnx_":
                            stack_fct_used.append(
                                "{}{}".format(prefix, op_conv))
                            prefix = '_' + prefix
                        rows.append(
                            '{}{}{}('.format(" " * indent * 4, prefix, op_conv))

                    opon = args["args"]
                    opon = opon[1:]
                    for expr2 in opon:
                        sexpr2 = write_expression(
                            stack_fct_used, expr2, indent + 1,
                            OnnxTranslator._parameter_mapping.get(op_conv, None))
                        if any(filter(lambda s: 'op_version="op_version"' in s, sexpr2)):
                            continue
                        rows.extend(sexpr2)
                        rows[-1] += ","
                    rows.append('{}op_version=op_version'.format(
                        " " * (indent + 1) * 4))
                    rows.append('{})'.format(" " * indent * 4))
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to interpret '{}'.".format(expr))
            return rows

        def write_function(stack_fct_used, to_replaces, node):
            rows = []
            name, args = node
            if name != 'FunctionDef':
                raise RuntimeError(  # pragma: no cover
                    "The code being translated should be a single function not "
                    "'{}' at {}.".format(name, self.make_msg(args)))
            list_args = list(map(str, args['args']))
            if all(map(lambda s: 'dtype=' not in s, list_args)):
                list_args.append("dtype=numpy.float32")
            if all(map(lambda s: 'op_version=' not in s, list_args)):
                list_args.append("op_version=None")
            fct_name = args['name']
            rows.append("def {}({}):".format(
                fct_name, ', '.join(list_args)))
            indent = 1

            to_replace = "# __HEADER__{}".format(id(node))
            to_replaces.append(to_replace)
            rows.append("{}{}".format(" " * (indent * 4), to_replace))

            code = args['code']
            for op, args in code:
                if op == "Assign":
                    name = args['name']
                    args = args["args"]
                    rows.append("{}{} = (".format(" " * (indent * 4), name))
                    rows.extend(write_expression(
                        stack_fct_used, args, indent + 1))
                    rows.append("{})".format(" " * (indent * 4)))
                elif op == "Return":
                    args = args["code"]
                    if output_names is None:
                        rows.append("{}return (".format(" " * (indent * 4)))
                        rows.extend(write_expression(
                            stack_fct_used, args, indent + 1))
                        rows.append("{})".format(" " * (indent * 4)))
                    else:
                        rows.append(
                            "{}return OnnxIdentity(".format(" " * (indent * 4)))
                        subrows = write_expression(
                            stack_fct_used, args, indent + 1)
                        subrows[-1] += ","
                        rows.extend(subrows)
                        rows.append("{}output_names={},".format(
                            " " * ((indent + 1) * 4), str(output_names)))
                        rows.append("{}op_version=op_version".format(
                            " " * ((indent + 1) * 4)))
                        rows.append("{})".format(" " * (indent * 4)))
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to process operator '{}' at {}. "
                        "Make sure it is either an affectation, "
                        "either a return.".format(op, self.make_msg(args)))
            return rows

        stack_fct_used = []
        to_replaces = []
        rows = write_function(stack_fct_used, to_replaces, self._code_fct)

        # handling dtype parameter
        if len(to_replaces) != 1:
            raise RuntimeError(  # pragma: no cover
                "The following code misses a placeholder:\n{}".format(
                    "\n".join(rows)))
        index = -1
        for i, row in enumerate(rows):
            if to_replaces[0] in row:
                index = i
                break

        header = []
        for fct in stack_fct_used:
            header.append(
                "    _{0} = lambda *args, op_version=op_version, **kwargs: {0}(*args, dtype=dtype, "
                "op_version=op_version, **kwargs)".format(fct))
        if len(header) > 0:
            header.append('')
        rows[index:index + 1] = header

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
                raise RuntimeError("Nested functions are not allowed at {}.".format(
                    self.make_msg(node)))
            self._stack.append(
                ('FunctionDef', {'args': [], 'code': [], 'name': info['name'], 'default': [],
                                 'lineno': node.lineno, 'col_offset': node.col_offset}))
            return
        if kind == "arguments":
            _, buf = self._get_last('FunctionDef')
            return
        if kind == "arg":
            return
        if kind == "Assign":
            self._stack.append(
                ('Assign', {'args': [], 'lineno': node.lineno, 'col_offset': node.col_offset}))
            return
        if kind in ('Name', 'Cst'):
            self._get_last(
                ('Assign', 'BinOp', 'Call', 'Return', 'FunctionDef', 'keyword', 'UnaryOp'))
            return
        if kind == 'BinOp':
            self._stack.append(
                ('BinOp', {'args': [], 'lineno': node.lineno, 'col_offset': node.col_offset}))
            return
        if kind == 'UnaryOp':
            self._stack.append(
                ('UnaryOp', {'args': [], 'lineno': node.lineno, 'col_offset': node.col_offset}))
            return
        if kind in OnnxTranslator._binary_operators:
            _, buf = self._get_last(('BinOp', 'UnaryOp'))
            buf['op'] = kind
            return
        if kind == 'Call':
            self._stack.append(
                ('Call', {'name': info['str'], 'args': [], 'lineno': node.lineno,
                          'col_offset': node.col_offset}))
            return
        if kind == 'Return':
            self._get_last('FunctionDef')
            self._stack.append(
                ('Return', {'code': [], 'lineno': node.lineno, 'col_offset': node.col_offset}))
            return
        if kind == "Attribute":
            if info.get('str', '') == 'T':
                raise NotImplementedError(  # pragma: no cover
                    "Transpose should be done with numpy.transpose not with .T'{}' "
                    "at {}\n{}\n---\n{}".format(
                        info.get('type', '?'), self.make_msg(node),
                        pprint.pformat(info), pprint.pformat(self._stack)))
            self._get_last('Call')
            return
        if kind == 'keyword':
            self._get_last('Call')
            self._stack.append(
                ('keyword', {'name': "{0}".format(node.arg),
                             'lineno': getattr(node, 'lineno', '?'),
                             'col_offset': getattr(node, 'col_offset', '?')}))
            return
        if kind == 'List':
            self._get_last('keyword')
            self._stack.append(
                ('List', {'elts': [], 'lineno': getattr(node, 'lineno', '?'),
                          'col_offset': getattr(node, 'col_offset', '?')}))
            return
        if kind == 'Num':
            self._get_last(('List', 'UnaryOp', 'BinOp', 'FunctionDef', 'Call'))
            return
        if kind == 'Str':
            self._get_last('keyword')
            return

        raise NotImplementedError(  # pragma: no cover
            "Unable to interpret kind '{}' at {}\n{}\n---\n{}".format(
                info.get('type', '?'), self.make_msg(
                    node), pprint.pformat(info),
                pprint.pformat(self._stack)))

    def _fix_default_values(self, code_fct):
        """
        Maps default values with parameter names.
        """
        nbdef = len(code_fct[1]['default'])
        nbpar = len(code_fct[1]['args'])
        args = []
        for i in range(nbpar):
            name, annotation = code_fct[1]['args'][i]
            j = nbdef - (nbpar - i)
            if j >= 0:
                default = code_fct[1]['default'][j]
                p = OnnxTranslator.Parameter(
                    name, annotation=annotation, value=default)
            else:
                p = OnnxTranslator.Parameter(name, annotation=annotation)
            args.append(p)
        code_fct[1]['args'] = args

    def _post_process(self, op, node):
        """
        Simplifies some operator such as ``OnnxNeg(2)``.
        """
        if op is None and 'args' in node:
            for i in range(len(node['args'])):
                if not isinstance(node['args'][i], tuple):
                    continue
                o, v = node['args'][i]
                if (o == 'UnaryOp' and len(v['args']) == 1 and
                        isinstance(v['args'][0], (int, float, numpy.int64,
                                                  numpy.float32, numpy.float64))):
                    if v['op'] == 'Sub':
                        node['args'][i] = -v['args'][0]

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
                if child['type'] == 'Str':
                    buf['default'].append(child['str'])
                elif child['type'] in ('Num', 'Cst'):
                    buf['default'].append(child['n'])
                elif child['type'] == 'arg':
                    buf['args'].append(
                        (child['str'], child.get('annotation', None)))
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to interpret type '{}' in function definition."
                        "\n{}".format(
                            child['type'], pprint.pformat(info)))
            return

        if kind == "Name":
            op, buf = self._get_last(
                ('Assign', 'BinOp', 'Call', 'Return', 'FunctionDef', 'keyword',
                 'UnaryOp'),
                info)
            if op == 'Assign':
                buf['name'] = info['str']
                return
            elif op in ('BinOp', 'Call'):
                buf['args'].append(info['str'])
                return
            elif op == 'Return':
                buf['code'] = info['str']
                return
            elif op == 'keyword':
                buf['value'] = info['str']
                return
            elif op == 'UnaryOp':
                buf['args'].append(info['str'])
                return
            elif op == 'FunctionDef':
                raise RuntimeError("Default value must be constant, variable '{}' was "
                                   "detected.".format(info['str']))

        if kind in OnnxTranslator._binary_operators:
            _, buf = self._get_last(('BinOp', 'UnaryOp'))
            return
        if kind in ('Call', 'BinOp', 'Assign', 'Return', 'UnaryOp'):
            op, buf = self._get_last(
                ('Call', 'BinOp', 'Assign', 'Return', 'UnaryOp'))
            self._post_process(op, buf)
            self._stack.pop()
            opp, parent = self._get_last(
                ('Call', 'BinOp', 'Assign', 'FunctionDef', 'Return', 'UnaryOp'))
            if opp in ('FunctionDef', 'Return'):
                parent['code'].append((op, buf))
            else:
                parent['args'].append((op, buf))
            self._post_process(None, parent)
            return
        if kind == 'FunctionDef':
            if len(self._stack) == 1:
                self._code_fct = self._stack[-1]
                self._fix_default_values(self._code_fct)
                self._stack = []
                return
        if kind == 'Module':
            return
        if kind == 'Attribute':
            op, buf = self._get_last(('Call', 'BinOp'))

            if len(info["children"]) > 0:
                fir = info["children"][0]
                if fir["type"] == "Name":
                    parent = fir["node"].id
                    info["str"] = "{0}.{1}".format(parent, info["str"])
                    info["children"][0]["remove"] = True

            buf['name'] = info["str"]
            buf['args'][0] = info["str"]
            return
        if kind in ('Num', 'Cst'):
            op, buf = self._get_last(
                ('List', 'BinOp', 'UnaryOp', 'FunctionDef', 'Call'))
            if op == 'FunctionDef':
                return
            if op == 'List':
                buf['elts'].append(info['n'])
            else:
                buf['args'].append(info['n'])
            return
        if kind == 'Str':
            _, buf = self._get_last('keyword')
            buf['value'] = info['str']
            return
        if kind == 'List':
            op, buf = self._get_last('List')
            value = buf['elts']
            self._post_process(op, buf)
            self._stack.pop()
            opp, parent = self._get_last('keyword')
            parent['value'] = value
            self._post_process(None, parent)
            return
        if kind == 'keyword':
            op, buf = self._get_last('keyword')
            name = buf["name"]
            if 'value' not in buf:
                raise RuntimeError(str(buf))  # pragma: no cover
            value = buf['value']
            self._post_process(op, buf)
            self._stack.pop()
            opp, parent = self._get_last('Call')
            parent['args'].append(OnnxTranslator.Parameter(name, value))
            self._post_process(None, parent)
            return

        raise NotImplementedError(  # pragma: no cover
            "Unable to interpret kind '{}' at {}\n{}\n---\n{}".format(
                info.get('type', '?'), self.make_msg(
                    node), pprint.pformat(info),
                pprint.pformat(self._stack)))
