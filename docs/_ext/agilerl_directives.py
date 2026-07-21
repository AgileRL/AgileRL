"""Custom Sphinx directives for AgileRL documentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx


class TutorialNode(nodes.Admonition, nodes.Element):
    """Tutorial callout node (styled as an admonition with a book icon)."""


def _visit_tutorial(self: Any, node: nodes.Element) -> None:  # noqa: ANN401 -- sphinx visitor; translator type varies by builder
    """Visitor bound to whichever translator the active builder uses."""
    self.visit_admonition(node)


def _depart_tutorial(self: Any, node: nodes.Element) -> None:  # noqa: ANN401 -- sphinx visitor; translator type varies by builder
    """Visitor bound to whichever translator the active builder uses."""
    self.depart_admonition(node)


class TutorialDirective(SphinxDirective):
    """Highlight tutorial content with a purple admonition-style callout.

    Usage::

        .. tutorial::

           Content uses the default title "Tutorial".

        .. tutorial:: Training on Arena

           :ref:`my_tutorial`
              Short description on the next line, indented (definition list).

           :ref:`another_tutorial`
              Same layout when there is only one tutorial.
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec: ClassVar[dict] = {}

    def run(self) -> list[nodes.Node]:
        """Parse directive content into a tutorial admonition node."""
        self.assert_has_content()

        title_text = self.arguments[0] if self.arguments else "Tutorial"

        admonition_node = TutorialNode(title_text)
        admonition_node["classes"] = ["tutorial"]

        title = nodes.title(
            title_text,
            "",
            nodes.Text(title_text),
        )
        admonition_node += title

        self.state.nested_parse(self.content, self.content_offset, admonition_node)
        return [admonition_node]


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the tutorial directive and node with Sphinx."""
    app.add_node(
        TutorialNode,
        html=(_visit_tutorial, _depart_tutorial),
        latex=(_visit_tutorial, _depart_tutorial),
        text=(_visit_tutorial, _depart_tutorial),
        man=(_visit_tutorial, _depart_tutorial),
        texinfo=(_visit_tutorial, _depart_tutorial),
    )
    app.add_directive("tutorial", TutorialDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
