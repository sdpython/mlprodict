"""
Extension for sphinx to display the onnx nodes.
"""
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
import sphinx
from sphinx.util.nodes import nested_parse_with_titles
from tabulate import tabulate
from mlprodict.npy.xop import _dynamic_class_creation


class SupportedOnnxOpsDirective(Directive):
    """
    Automatically displays the list of supported ONNX models
    *skl2onnx* can use to build converters.
    """
    required_arguments = False
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        cls = _dynamic_class_creation()
        cls_name = [(c.__name__, c) for c in cls]
        rows = []
        sorted_cls_name = list(sorted(cls_name))
        main = nodes.container()

        def make_ref(cl):
            return ":ref:`l-xop-onnx-{}`".format(cl.__name__)

        table = []
        cut = len(sorted_cls_name) // 3 + \
            (1 if len(sorted_cls_name) % 3 else 0)
        for i in range(cut):
            row = []
            row.append(make_ref(sorted_cls_name[i][1]))
            if i + cut < len(sorted_cls_name):
                row.append(make_ref(sorted_cls_name[i + cut][1]))
                if i + cut * 2 < len(sorted_cls_name):
                    row.append(make_ref(sorted_cls_name[i + cut * 2][1]))
                else:
                    row.append('')
            else:
                row.append('')
                row.append('')
            table.append(row)

        rst = tabulate(table, tablefmt="rst")
        rows = rst.split("\n")

        node = nodes.container()
        st = StringList(rows)
        nested_parse_with_titles(self.state, st, node)
        main += node

        rows.append('')
        for name, cl in sorted_cls_name:
            rows = []
            rows.append('.. _l-xop-onnx-{}:'.format(cl.__name__))
            rows.append('')
            rows.append(cl.__name__)
            rows.append('=' * len(cl.__name__))
            rows.append('')
            rows.append(
                ".. autoclass:: mlprodict.npy.xop_auto_import_.{}".format(name))
            st = StringList(rows)
            node = nodes.container()
            nested_parse_with_titles(self.state, st, node)
            main += node

        return [main]


def setup(app):
    app.add_directive('supported-onnx-ops', SupportedOnnxOpsDirective)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
