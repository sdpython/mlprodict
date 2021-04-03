"""
@file
@brief Documentation helper.
"""
import keyword
import textwrap
import re
from jinja2 import Template
from jinja2.runtime import Undefined
from onnx.defs import OpSchema
from ...tools import change_style


def type_mapping(name):
    """
    Mapping between types name and type integer value.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.doc.doc_helper import type_mapping
        import pprint
        pprint.pprint(type_mapping(None))
        print(type_mapping("INT"))
        print(type_mapping(2))
    """
    di = dict(FLOAT=1, FLOATS=6, GRAPH=5, GRAPHS=10, INT=2,
              INTS=7, STRING=3, STRINGS=8, TENSOR=4,
              TENSORS=9, UNDEFINED=0, SPARSE_TENSOR=11)
    if name is None:
        return di
    if isinstance(name, str):
        return di[name]
    rev = {v: k for k, v in di.items()}
    return rev[name]


def _get_doc_template():

    return Template(textwrap.dedent("""
        {% for sch in schemas %}

        {{format_name_with_domain(sch)}}
        {{'=' * len(format_name_with_domain(sch))}}

        {{process_documentation(sch.doc)}}

        {% if sch.attributes %}
        **Attributes**

        {% for _, attr in sorted(sch.attributes.items()) %}* *{{attr.name}}*{%
          if attr.required %} (required){% endif %}: {{
          process_attribute_doc(attr.description)}} {%
          if attr.default_value %} {{
          process_default_value(attr.default_value)
          }} ({{type_mapping(attr.type)}}){% endif %}
        {% endfor %}
        {% endif %}

        {% if sch.inputs %}
        **Inputs**

        {% if sch.min_input != sch.max_input %}Between {{sch.min_input
        }} and {{sch.max_input}} inputs.
        {% endif %}
        {% for ii, inp in enumerate(sch.inputs) %}
        * *{{getname(inp, ii)}}*{{format_option(inp)}}{{inp.typeStr}}: {{
        inp.description}}{% endfor %}
        {% endif %}

        {% if sch.outputs %}
        **Outputs**

        {% if sch.min_output != sch.max_output %}Between {{sch.min_output
        }} and {{sch.max_output}} outputs.
        {% endif %}
        {% for ii, out in enumerate(sch.outputs) %}
        * *{{getname(out, ii)}}*{{format_option(out)}}{{out.typeStr}}: {{
        out.description}}{% endfor %}
        {% endif %}

        {% if sch.type_constraints %}
        **Type Constraints**

        {% for ii, type_constraint in enumerate(sch.type_constraints)
        %}* {{getconstraint(type_constraint, ii)}}: {{
        type_constraint.description}}
        {% endfor %}
        {% endif %}

        **Version**

        *Onnx name:* `{{sch.name}} <{{build_doc_url(sch)}}{{sch.name}}>`_

        {% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}
        No versioning maintained for experimental ops.
        {% else %}
        This version of the operator has been {% if
        sch.deprecated %}deprecated{% else %}available{% endif %} since
        version {{sch.since_version}}{% if
        sch.domain %} of domain {{sch.domain}}{% endif %}.
        {% if len(sch.versions) > 1 %}
        Other versions of this operator:
        {% for v in sch.version[:-1] %} {{v}} {% endfor %}
        {% endif %}
        {% endif %}

        **Runtime implementation:**
        :class:`{{sch.name}}
        <mlprodict.onnxrt.ops_cpu.op_{{change_style(sch.name)}}.{{sch.name}}>`

        {% endfor %}
    """))


_template_operator = _get_doc_template()


class NewOperatorSchema:
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleRegressorDouble.
    """

    def __init__(self, name):
        self.name = name
        self.domain = 'mlprodict'


def get_rst_doc(op_name):
    """
    Returns a documentation in RST format
    for all :class:`OnnxOperator`.

    :param op_name: operator name of None for all
    :return: string

    The function relies on module :epkg:`jinja2` or replaces it
    with a simple rendering if not present.
    """
    from ..ops_cpu._op import _schemas
    schemas = [_schemas.get(op_name, NewOperatorSchema(op_name))]

    def format_name_with_domain(sch):
        if sch.domain:
            return '{} ({})'.format(sch.name, sch.domain)
        return sch.name

    def format_option(obj):
        opts = []
        if OpSchema.FormalParameterOption.Optional == obj.option:
            opts.append('optional')
        elif OpSchema.FormalParameterOption.Variadic == obj.option:
            opts.append('variadic')
        if getattr(obj, 'isHomogeneous', False):
            opts.append('heterogeneous')
        if opts:
            return " (%s)" % ", ".join(opts)
        return ""  # pragma: no cover

    def getconstraint(const, ii):
        if const.type_param_str:
            name = const.type_param_str
        else:
            name = str(ii)  # pragma: no cover
        if const.allowed_type_strs:
            name += " " + ", ".join(const.allowed_type_strs)
        return name

    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)  # pragma: no cover
        return name

    def process_documentation(doc):
        if doc is None:
            doc = ''  # pragma: no cover
        if isinstance(doc, Undefined):
            doc = ''  # pragma: no cover
        if not isinstance(doc, str):
            raise TypeError(  # pragma: no cover
                "Unexpected type {} for {}".format(type(doc), doc))
        doc = textwrap.dedent(doc)
        main_docs_url = "https://github.com/onnx/onnx/blob/master/"
        rep = {
            '[the doc](IR.md)': '`ONNX <{0}docs/IR.md>`_',
            '[the doc](Broadcasting.md)':
                '`Broadcasting in ONNX <{0}docs/Broadcasting.md>`_',
            '<dl>': '',
            '</dl>': '',
            '<dt>': '* ',
            '<dd>': '  ',
            '</dt>': '',
            '</dd>': '',
            '<tt>': '``',
            '</tt>': '``',
            '<br>': '\n',
        }
        for k, v in rep.items():
            doc = doc.replace(k, v.format(main_docs_url))
        move = 0
        lines = []
        for line in doc.split('\n'):
            if line.startswith("```"):
                if move > 0:
                    move -= 4
                    lines.append("\n")
                else:
                    lines.append("::\n")
                    move += 4
            elif move > 0:
                lines.append(" " * move + line)
            else:
                lines.append(line)
        return "\n".join(lines)

    def process_attribute_doc(doc):
        return doc.replace("<br>", " ")

    def build_doc_url(sch):
        doc_url = "https://github.com/onnx/onnx/blob/master/docs/Operators"
        if "ml" in sch.domain:
            doc_url += "-ml"
        doc_url += ".md"
        doc_url += "#"
        if sch.domain not in (None, '', 'ai.onnx'):
            doc_url += sch.domain + "."
        return doc_url

    def process_default_value(value):
        if value is None:
            return ''  # pragma: no cover
        res = []
        for c in str(value):
            if ((c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z') or
                    (c >= '0' and c <= '9')):
                res.append(c)
                continue
            if c in '[]-+(),.?':
                res.append(c)
                continue
        if len(res) == 0:
            return "*default value cannot be automatically retrieved*"
        return "Default value is ``" + ''.join(res) + "``"

    fnwd = format_name_with_domain
    tmpl = _template_operator
    docs = tmpl.render(schemas=schemas, OpSchema=OpSchema,
                       len=len, getattr=getattr, sorted=sorted,
                       format_option=format_option,
                       getconstraint=getconstraint,
                       getname=getname, enumerate=enumerate,
                       format_name_with_domain=fnwd,
                       process_documentation=process_documentation,
                       build_doc_url=build_doc_url, str=str,
                       type_mapping=type_mapping,
                       process_attribute_doc=process_attribute_doc,
                       process_default_value=process_default_value,
                       change_style=change_style)
    return docs.replace(" Default value is ````", "")


def debug_onnx_object(obj, depth=3):
    """
    ``__dict__`` is not in most of :epkg:`onnx` objects.
    This function uses function *dir* to explore this object.
    """
    def iterable(o):
        try:
            iter(o)
            return True
        except TypeError:
            return False

    if depth <= 0:
        return None

    rows = [str(type(obj))]
    if not isinstance(obj, (int, str, float, bool)):

        for k in sorted(dir(obj)):
            try:
                val = getattr(obj, k)
                sval = str(val).replace("\n", " ")
            except (AttributeError, ValueError) as e:  # pragma: no cover
                sval = "ERRROR-" + str(e)
                val = None

            if 'method-wrapper' in sval or "built-in method" in sval:
                continue

            rows.append("- {}: {}".format(k, sval))
            if k.startswith('__') and k.endswith('__'):
                continue
            if val is None:
                continue

            if isinstance(val, dict):
                try:
                    sorted_list = list(sorted(val.items()))
                except TypeError:  # pragma: no cover
                    sorted_list = list(val.items())
                for kk, vv in sorted_list:
                    rows.append("  - [%s]: %s" % (str(kk), str(vv)))
                    res = debug_onnx_object(vv, depth - 1)
                    if res is None:
                        continue
                    for line in res.split("\n"):
                        rows.append("    " + line)
            elif iterable(val):
                if all(map(lambda o: isinstance(o, (str, bytes)) and len(o) == 1, val)):
                    continue
                for i, vv in enumerate(val):
                    rows.append("  - [%d]: %s" % (i, str(vv)))
                    res = debug_onnx_object(vv, depth - 1)
                    if res is None:
                        continue
                    for line in res.split("\n"):
                        rows.append("    " + line)
            elif not callable(val):
                res = debug_onnx_object(val, depth - 1)
                if res is None:
                    continue
                for line in res.split("\n"):
                    rows.append("  " + line)

    return "\n".join(rows)


def visual_rst_template():
    """
    Returns a :epkg:`jinja2` template to display DOT graph for each
    converter from :epkg:`sklearn-onnx`.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.doc.doc_helper import visual_rst_template
        print(visual_rst_template())
    """
    return textwrap.dedent("""

    .. _l-{{link}}:

    {{ title }}
    {{ '=' * len(title) }}

    Fitted on a problem type *{{ kind }}*
    (see :func:`find_suitable_problem
    <mlprodict.onnxrt.validate.validate_problems.find_suitable_problem>`),
    method {{ method }} matches output {{ output_index }}.
    {{ optim_param }}

    ::

        {{ indent(model, "    ") }}

    {{ table }}

    .. gdot::

        {{ indent(dot, "    ") }}
    """)
