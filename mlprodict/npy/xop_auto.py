"""
@file
@brief Automates the generation of operators for the
documentation for the Xop API.

::

    def setup(app):
        app.connect('builder-inited', generate_op_doc)

.. versionadded:: 0.9
"""
import textwrap
import importlib
import inspect
import onnx
import onnx.defs
from onnx.backend.test.case.base import _Exporter
from onnx.defs import OpSchema


def _get_doc_template():
    try:
        from jinja2 import Template
    except ImportError:  # pragma no cover
        class Template:
            "Docstring template"

            def __init__(self, *args):
                pass

            def render(self, **context):
                "render"
                schemas = context['schemas']
                rows = []
                for sch in schemas:
                    doc = sch.doc or ''
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name),
                                 "", doc, ""])
                return "\n".join(rows)

    return Template(textwrap.dedent("""
        {% for sch in schemas %}

        .. tag-diff-insert.

        .. _l-onnx-op{{sch.domain.lower().replace(".", "-")}}-{{sch.name.lower()}}-{{str(sch.since_version)}}:

        {{format_name_with_domain(sch)}}
        {{'=' * len(format_name_with_domain(sch))}}

        **Version**

        * **name**: `{{sch.name}} (GitHub) <{{build_doc_url(sch)}}{{sch.name}}>`_
        * **domain**: **{{sch.domain}}**
        * **since_version**: **{{sch.since_version}}**
        * **function**: {{sch.has_function}}**
        * **support_level**: {{sch.support_level}}**
        * **shape inference**: {{sch.has_type_and_shape_inference_function}}**

        {% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}
        No versioning maintained for experimental ops.
        {% else %}
        This version of the operator has been {% if
        sch.deprecated %}deprecated{% else %}available{% endif %}
        **since version {{sch.since_version}}{% if
        sch.domain %} of domain {{sch.domain}}{% endif %}**.
        {% if len(sch.versions) > 1 %}
        Other versions of this operator:
        {% for v in sch.version[:-1] %} {{v}} {% endfor %}
        {% endif %}
        {% endif %}

        **Summary**

        {{process_documentation(sch.doc)}}

        {% if sch.attributes %}
        **Attributes**

        {% for _, attr in sorted(sch.attributes.items()) %}* *{{attr.name}}*{%
          if attr.required %} (required){% endif %}:
        {{text_wrap(attr.description, 2)}} {%
          if attr.default_value %}{{clean_default_value(attr.default_value)}}{%
          endif %}
        {% endfor %}
        {% endif %}

        {% if sch.inputs %}
        **Inputs**

        {% if sch.min_input != sch.max_input %}Between {{sch.min_input
        }} and {{sch.max_input}} inputs.
        {% endif %}
        {% for ii, inp in enumerate(sch.inputs) %}
        * *{{getname(inp, ii)}}*{{format_option(inp)}} - **{{inp.typeStr}}**:
        {{text_wrap(inp.description, 2)}}{% endfor %}
        {% endif %}

        {% if sch.outputs %}
        **Outputs**

        {% if sch.min_output != sch.max_output %}Between {{sch.min_output
        }} and {{sch.max_output}} outputs.
        {% endif %}
        {% for ii, out in enumerate(sch.outputs) %}
        * *{{getname(out, ii)}}*{{format_option(out)}} - **{{out.typeStr}}**:
        {{text_wrap(out.description, 2)}}{% endfor %}
        {% endif %}

        {% if sch.type_constraints %}
        **Type Constraints**

        {% for ii, type_constraint in enumerate(sch.type_constraints)
        %}* {{get_constraint(type_constraint, ii)}}:
        {{text_wrap(type_constraint.description, 2)}}
        {% endfor %}
        {% endif %}

        {% if get_onnx_example and is_last_schema(sch): %}
        **Examples**

        {% for example, code in get_onnx_example(sch.name).items(): %}
        **{{ example }}**

        ::

        {{ format_example(code) }}

        {% endfor %}
        {% endif %}

        {% endfor %}
    """))


_template_operator = _get_doc_template()
_get_all_schemas_with_history = None


def _populate__get_all_schemas_with_history():
    res = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        domain = schema.domain
        version = schema.since_version
        name = schema.name
        if domain not in res:
            res[domain] = {}
        if name not in res[domain]:
            res[domain][name] = {}
        res[domain][name][version] = schema
    return res


def get_domain_list():
    """
    Returns the list of available domains.
    """
    return list(sorted(set(map(lambda s: s.domain,
                               onnx.defs.get_all_schemas_with_history()))))


def get_operator_schemas(op_name, version=None, domain=None):
    """
    Returns all schemas mapped to an operator name.

    :param op_name: name of the operator
    :param version: version
    :param domain: domain
    :return: list of schemas
    """
    global _get_all_schemas_with_history  # pylint: disable=W0603
    if version == 'last' and op_name is not None:
        if domain is not None:
            return [onnx.defs.get_schema(op_name, domain=domain)]
    if _get_all_schemas_with_history is None:
        _get_all_schemas_with_history = _populate__get_all_schemas_with_history()
    if domain is None:
        domains = []
        for dom, ops in _get_all_schemas_with_history.items():
            if op_name is None or op_name in ops:
                domains.append(dom)
    else:
        domains = [domain]

    # schemas
    sch = []
    for dom in domains:
        ops = _get_all_schemas_with_history[dom]
        if op_name is None:
            for op, v in ops.items():
                if version is None:
                    sch.extend(v.values())
                elif version == 'last':
                    sch.append(
                        onnx.defs.get_schema(op, domain=dom))
                else:
                    sch.append(v[version])
        elif op_name in ops:
            if version is None:
                sch.extend(ops[op_name].values())
            elif version in ops[op_name]:
                sch.append(ops[op_name][version])

    # sort
    vals = [(s.domain, s.name, -s.since_version, s) for s in sch]
    vals.sort()
    return [v[-1] for v in vals]


def get_rst_doc(op_name=None, domain=None, version='last', clean=True,
                diff=False, example=False):
    """
    Returns a documentation in RST format
    for all :class:`OnnxOperator`.

    :param op_name: operator name of None for all
    :param domain: domain
    :param version: version, None for all, `'last'` for the most recent one
    :param clean: clean empty lines
    :param diff: highlights differences between two versions
    :param example: add example to the documentation
    :return: string

    The function relies on module :epkg:`jinja2` or replaces it
    with a simple rendering if not present.
    """
    schemas = get_operator_schemas(op_name, domain=domain, version=version)

    # from onnx.backend.sample.ops import collect_sample_implementations
    # from onnx.backend.test.case import collect_snippets
    # SNIPPETS = collect_snippets()
    # SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
    def format_name_with_domain(sch):
        if version == 'last':
            if sch.domain:
                return '{} ({})'.format(sch.name, sch.domain)
            return sch.name
        if sch.domain:
            return '{} - {} ({})'.format(sch.name, sch.since_version, sch.domain)
        return '%s - %d' % (sch.name, sch.since_version)

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
        return ""

    def format_example(code):
        code = textwrap.indent(code, '    ')
        return code

    def get_constraint(const, ii):
        if const.type_param_str:
            name = const.type_param_str
        else:
            name = str(ii)
        name = "**%s** in (" % name
        if const.allowed_type_strs:
            text = ",\n  ".join(sorted(const.allowed_type_strs))
            name += "\n  " + text + "\n  )"
        return name

    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)
        return name

    def process_documentation(doc):
        if doc is None:
            doc = ''
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

    def build_doc_url(sch):
        doc_url = "https://github.com/onnx/onnx/blob/main/docs/Operators"
        if "ml" in sch.domain:
            doc_url += "-ml"
        doc_url += ".md"
        doc_url += "#"
        if sch.domain not in (None, '', 'ai.onnx'):
            doc_url += sch.domain + "."
        return doc_url

    def clean_default_value(value):
        res = str(value).replace('\\n', ' ').strip()
        if len(res) > 0:
            return "Default value is ``%s``." % res
        return ""

    def text_wrap(text, indent):
        s = ' ' * indent
        lines = textwrap.wrap(text, initial_indent=s, subsequent_indent=s)
        return '\n'.join(lines)

    fnwd = format_name_with_domain
    tmpl = _template_operator
    docs = tmpl.render(schemas=schemas, OpSchema=OpSchema,
                       len=len, getattr=getattr, sorted=sorted,
                       format_option=format_option,
                       get_constraint=get_constraint,
                       getname=getname, enumerate=enumerate,
                       format_name_with_domain=fnwd,
                       process_documentation=process_documentation,
                       build_doc_url=build_doc_url, text_wrap=text_wrap,
                       str=str, clean_default_value=clean_default_value,
                       get_onnx_example=get_onnx_example if example else None,
                       format_example=format_example,
                       is_last_schema=is_last_schema)
    if diff:
        lines = docs.split('\n')
        new_lines = ['']
        for line in lines:
            line = line.rstrip('\r\t ')
            if len(line) == 0 and len(new_lines[-1]) == 0:
                continue
            new_lines.append(line)
        docs = '\n'.join(new_lines)
        docs = _insert_diff(docs, '.. tag-diff-insert.')

    if clean:
        lines = docs.split('\n')
        new_lines = ['']
        for line in lines:
            line = line.rstrip('\r\t ')
            if len(line) == 0 and len(new_lines[-1]) == 0:
                continue
            new_lines.append(line)
        docs = '\n'.join(new_lines)

    return docs


def _insert_diff(docs, split='.. tag-diff-insert.'):
    """
    Splits a using `split`, insert HTML differences between pieces.
    The function relies on package :epkg:`pyquickhelper`.
    """
    spl = docs.split(split)
    if len(spl) <= 1:
        return docs

    from pyquickhelper.texthelper.edit_text_diff import (
        edit_distance_text, diff2html)

    pieces = [spl[0]]
    for i in range(1, len(spl)):
        spl1 = spl[i - 1].strip('\n ')
        spl2 = spl[i].strip('\n ')
        if len(spl1) < 5 or len(spl2) < 5:
            pieces.append(spl[i])
            continue

        _, aligned, final = edit_distance_text(  # pylint: disable=W0632
            spl1, spl2, threshold=0.5)
        ht = diff2html(spl1, spl2, aligned, final, two_columns=True)
        ht = '    ' + '\n    '.join(ht.split('\n'))
        pieces.extend(['', '.. html::', '', ht, '', spl[i]])

    return '\n'.join(pieces)


def get_onnx_example(op_name):
    """
    Retrieves examples associated to one operator
    stored in onnx packages.

    :param op_name: operator name
    :param fmt: rendering format
    :return: dictionary
    """
    module = 'onnx.backend.test.case.node.%s' % op_name.lower()
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return {}
    results = {}
    for v in mod.__dict__.values():
        if not isinstance(v, _Exporter):
            continue
        code_cls = inspect.getsource(v)
        codes = code_cls.split('@staticmethod')
        for me in v.__dict__:
            if not me.startswith('export_'):
                continue
            sub = ' %s()' % me
            found = None
            for code in codes:
                if sub in code:
                    found = code
            if found is None:
                raise RuntimeError(
                    "Unable to find %r in\n%s" % (sub, code_cls))
            found = textwrap.dedent(found)
            lines = found.split('\n')
            first = 0
            for i in range(len(lines)):
                if lines[i].startswith('def '):
                    first = i + 1
            found = textwrap.dedent('\n'.join(lines[first:]))
            results[me[len('export_'):]] = found
    return results


def is_last_schema(sch):
    """
    Tells if this is the most recent schema for this operator.

    :param sch: schema
    :return: True
    """
    last = onnx.defs.get_schema(sch.name, domain=sch.domain)
    return last.since_version == sch.since_version
