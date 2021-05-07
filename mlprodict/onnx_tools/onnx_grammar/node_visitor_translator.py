"""
@file
@brief One class which visits a syntax tree.
"""

import ast
from .onnx_translator import OnnxTranslator


class CodeNodeVisitor(ast.NodeVisitor):

    """
    Defines a visitor which walks though the syntax tree of the code.

    .. exref::
        :title: Get the tree of a simple function

        The following code uses Python syntax but follows a SQL logic.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning
            :process:
            :store_in_file: fct2onnx1.py

            import ast
            import inspect
            from textwrap import dedent
            from mlprodict.onnx_tools.onnx_grammar import CodeNodeVisitor

            def norm2(x, y):
                delta = x - y
                n = delta ** 2
                return n

            code = dedent(inspect.getsource(norm2))
            node = ast.parse(code)
            v = CodeNodeVisitor()
            v.visit(node)
            for r in v.Rows :
                print("{0}{1}: {2}".format("    " * r["indent"], r["type"], r["str"]))
    """

    def __init__(self, translator=None):
        """
        @param      translator      @see cl CodeTranslator

        By default the translator is @see cl OnnxTranslator.
        """
        ast.NodeVisitor.__init__(self)
        self._rows = []
        self._indent = 0
        self._stack = []
        self._translator = OnnxTranslator(
            self) if translator is None else translator

    def push(self, row):
        """
        Pushes an element into a list.
        """
        self._rows.append(row)

    def generic_visit(self, node):
        """
        Overrides ``generic_visit`` to check it is not used.
        """
        raise AttributeError(  # pragma: no cover
            "generic_visit_args should be used.")

    def generic_visit_args(self, node, row):
        """
        Overrides ``generic_visit`` to keep track of the indentation
        and the node parent. The function will add field
        ``row["children"] = visited`` nodes from here.

        @param      node        node which needs to be visited
        @param      row         row (a dictionary)
        @return                 See ``ast.NodeVisitor.generic_visit``
        """
        if hasattr(node, 'lineno'):
            row['lineno'] = node.lineno
        if hasattr(node, 'col_offset'):
            row['col_offset'] = node.col_offset
        self._indent += 1
        last = len(self._rows)
        self._translator.visit(node, row)
        res = ast.NodeVisitor.generic_visit(  # pylint: disable=E1111
            self, node)  # pylint: disable=E1111
        row["children"] = [
            _ for _ in self._rows[
                last:] if _["indent"] == self._indent]
        self._indent -= 1
        self._translator.depart(node, row)
        return res

    def make_msg(self, node):
        """
        Displays line and column information into a string.
        """
        return "line {}, col {}".format(  # pragma: no cover
            getattr(node, 'lineno', '?'), getattr(node, 'col_offset', '?'))

    def visit(self, node):
        """
        Visits a node, a method must exist for every object class.
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise TypeError(  # pragma: no cover
                "Unable to find a method '{}' at {}.".format(
                    method, self.make_msg(node)))
        res = visitor(node)
        # print(method, CodeNodeVisitor.print_node(node))
        return res

    def visit_(self, node):
        """
        If an element is not found...
        """
        raise NotImplementedError(  # pragma: no cover
            "Node '{}' ({}) not recognized at {}\nNode\n{}\n--"
            "Status--\n{}".format(
                node, type(node), self.make_msg(node),
                self.print_node(node), self.print_tree()))

    @staticmethod
    def print_node(node):
        """
        Debugging purpose.
        """
        r = []
        for att in sorted(set(["s", "name", "str", "id", "body", "n",
                               "arg", "targets", "attr", "returns", "ctx",
                               'col_offset', 'lineno',
                               'value'] + list(getattr(node, '_attributes', [])))):
            v = getattr(node, att, None)
            if v is not None or att in getattr(node, '_fields', []):
                r.append("{0}={1}".format(att, v))
        return " ".join(r)

    def print_tree(self):
        """
        Displays the tree of instructions.

        @return     string
        """
        rows = []
        for r in self.Rows:
            rows.append(
                ("{0}{1}: {2}".format(
                    "    " *
                    r["indent"],
                    r["type"],
                    r["str"])))
        return "\n".join(rows)

    @property
    def Rows(self):
        """
        returns a list of dictionaries with all the elements of the code
        """
        return [_ for _ in self._rows if not _.get("remove", False)]

    def export(self, context=None, **kwargs):
        """
        Calls method *export* from the translator class.

        @param      context     known :epkg:`python` needed to run
                                the translated function
        @param      kwargs      whatever the method *export* from
                                the translator class ingests
        @return                 whatever the method *export* from
                                the translator class returns
        """
        return self._translator.export(context=context, **kwargs)

    ###########
    # Methods for python code elements
    ###########

    def visit_Str(self, node):  # pylint: disable=C0111
        cont = {
            "indent": self._indent,
            "type": "Str",
            "str": node.s,
            "node": node,
            "value": node.s}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Name(self, node):  # pylint: disable=C0111
        cont = {
            "indent": self._indent,
            "type": "Name",
            "str": node.id,
            "node": node,
            "id": node.id,
            "ctx": node.ctx}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Module(self, node):  # pylint: disable=C0111
        cont = {
            "indent": self._indent,
            "type": "Module",
            "str": "",
            "body": node.body,
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_FunctionDef(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "FunctionDef", "str": node.name, "name": node.name, "body": node.body,
                "node": node, "returns": node.returns}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_List(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "List",
                "str": "", "elts": node.elts,
                "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_arguments(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "arguments", "str": "",
                "node": node, "args": node.args}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_arg(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "arg", "str": node.arg,
                "node": node,
                "arg": node.arg, "annotation": node.annotation}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Assign(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Assign", "str": "", "node": node,
                "targets": node.targets, "value": node.value}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Store(self, node):  # pylint: disable=C0111
        #cont = { "indent":self._indent, "type": "Store", "str": "" }
        # self.push(cont)
        cont = {}
        return self.generic_visit_args(node, cont)

    def visit_Call(self, node):  # pylint: disable=C0111
        if "attr" in node.func.__dict__:
            cont = {"indent": self._indent, "type": "Call", "str": node.func.attr,
                    "node": node, "func": node.func}
        else:
            cont = {"indent": self._indent, "type": "Call", "str": node.func.id,
                    "node": node, "func": node.func}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Attribute(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Attribute", "str": node.attr,
                "node": node, "value": node.value, "ctx": node.ctx, "attr": node.attr}
        self.push(cont)
        # last = len(self._rows)
        res = self.generic_visit_args(node, cont)

        if len(cont["children"]) > 0:
            fir = cont["children"][0]
            if fir["type"] == "Name":
                parent = fir["node"].id
                cont["str"] = "{0}.{1}".format(parent, cont["str"])
                cont["children"][0]["remove"] = True
        return res

    def visit_Load(self, node):  # pylint: disable=C0111
        cont = {}
        return self.generic_visit_args(node, cont)

    def visit_keyword(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "keyword", "str": "{0}".format(node.arg),
                "node": node, "arg": node.arg, "value": node.value}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_BinOp(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "BinOp",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Div(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Div",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Sub(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Sub",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_USub(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Sub",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Add(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Add",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Pow(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Pow",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Mult(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Mult",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_MatMult(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "MatMult",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Compare(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Compare",
                "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Gt(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Gt", "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Lt(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Lt", "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_UnaryOp(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent,
                "type": "UnaryOp", "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Num(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Num",
                "node": node, "str": "{0}".format(node.n),
                'n': node.n}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Return(self, node):  # pylint: disable=C0111
        cont = {"indent": self._indent, "type": "Return", "node": node, "str": "",
                'value': node.value}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_NameConstant(self, node):
        """
        A name.
        """
        if node.value is None:
            cont = {"indent": self._indent, "type": "Cst",
                    "node": node, "str": "None",
                    'n': None}
            self.push(cont)
            return self.generic_visit_args(node, cont)
        return self.visit_(node)  # pragma: no cover
