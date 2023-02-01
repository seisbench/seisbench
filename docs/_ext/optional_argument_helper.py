"""
Directive for documenting optional arguments of WaveformModels
"""

from importlib import import_module

from docutils import nodes
from docutils.parsers.rst import Directive


class ArgsDirective(Directive):
    """
    Directive for writing out documentation
    """

    required_arguments = 2
    optional_arguments = 0

    def run(self):
        mod, cls = self.arguments

        cls = getattr(import_module(mod), cls)

        annotate_args = cls._annotate_args

        table = nodes.table()
        tgroup = nodes.tgroup(cols=3)
        tgroup.append(nodes.colspec(colwidth=1))  # Argument
        tgroup.append(nodes.colspec(colwidth=2))  # Description
        tgroup.append(nodes.colspec(colwidth=1))  # Default value
        table += tgroup

        thead = nodes.thead()
        tgroup += thead
        row = nodes.row()
        entry = nodes.entry()
        entry += nodes.paragraph(text="Argument")
        row += entry
        entry = nodes.entry()
        entry += nodes.paragraph(text="Description")
        row += entry
        entry = nodes.entry()
        entry += nodes.paragraph(text="Default value")
        row += entry

        thead.append(row)

        rows = []
        for key, value in annotate_args.items():
            row = nodes.row()
            rows.append(row)

            entry = nodes.entry()
            entry += nodes.paragraph(text=key)
            row += entry

            entry = nodes.entry()
            entry += nodes.paragraph(text=str(value[0]))
            row += entry

            entry = nodes.entry()
            entry += nodes.paragraph(text=str(value[1]))
            row += entry

        tbody = nodes.tbody()
        tbody.extend(rows)
        tgroup += tbody

        caption = nodes.Text(
            "The following parameters are available for the annotate/classify functions:"
        )
        remark = nodes.hint()
        remark += nodes.Text(
            "Please note that the default parameters can be superseded by the pretrained "
            "model weights. Check model.default_args to see which parameters are overwritten."
        )

        return [caption, table, remark]


def setup(app):
    app.add_directive("document_args", ArgsDirective)
