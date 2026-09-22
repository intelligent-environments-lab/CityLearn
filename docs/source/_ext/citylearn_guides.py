"""Publish the canonical Markdown guides without maintaining duplicate copies."""

from pathlib import Path
import ast
import re
from urllib.parse import urlsplit


def scorecard_names(root):
    tree = ast.parse((root / "citylearn/internal/kpi.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "SCORECARD_DEFAULT_KPIS":
                return ast.literal_eval(node.value)
    raise RuntimeError("SCORECARD_DEFAULT_KPIS was not found")


def website_guide_content(content):
    """Keep repository translations out of the English-only website."""
    content = re.sub(r"^Portuguese version:.*\n\n?", "", content, flags=re.MULTILINE)
    # The website copy lives two levels deeper than the canonical Markdown file.
    return content.replace("(../assets/", "(../../../assets/")


def prepare_guides(app):
    root = Path(app.confdir).parents[1]
    source = root / "docs"
    destination = Path(app.srcdir) / "guides"
    destination.mkdir(exist_ok=True)
    for path in sorted(source.glob("*.md")):
        if path.name == "retired_datasets.md":
            continue
        content = website_guide_content(path.read_text(encoding="utf-8"))
        (destination / path.name).write_text(content, encoding="utf-8")
    names = scorecard_names(root)
    lines = [
        "# Default scorecard KPI names", "",
        "This list is generated from the simulator's `SCORECARD_DEFAULT_KPIS`",
        "selection at documentation build time. See [How to read the scorecard](scorecard.md)",
        "for interpretation and [KPI reference](kpis_reference.md) for definitions.",
        "The full evaluation table contains additional KPIs.", "",
    ]
    for level in ("district", "building"):
        lines += ["## " + level.capitalize(), ""]
        lines += ["- `" + name + "`" for name in names if name.startswith(level + "_")]
        lines += [""]
    (destination / "scorecard_kpi_names.md").write_text("\n".join(lines), encoding="utf-8")


def setup(app):
    app.connect("builder-inited", prepare_guides)
    app.connect("html-page-context", guide_source_link)
    app.connect("doctree-resolved", local_site_links)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}


def guide_source_link(app, pagename, templatename, context, doctree):
    """Point edit links at maintained sources, not generated guide copies."""
    if not pagename.startswith("guides/"):
        return
    if pagename == "guides/scorecard_kpi_names":
        context["display_github"] = False
    else:
        context["meta"] = dict(context.get("meta") or {})
        context["meta"]["github_url"] = (
            "https://github.com/citylearn-project/CityLearn/blob/master/docs/"
            + pagename.removeprefix("guides/") + ".md"
        )


def local_site_links(app, doctree, docname):
    """Keep canonical website links inside local/offline documentation builds."""
    from docutils import nodes

    if app.builder.format != "html":
        return
    for node in doctree.findall(nodes.reference):
        url = urlsplit(node.get("refuri", ""))
        if url.netloc not in ("www.citylearn.net", "citylearn.net"):
            continue
        target = url.path.lstrip("/")
        target = target[:-5] if target.endswith(".html") else target
        target = target or "index"
        if target in app.env.found_docs and not url.query:
            node["refuri"] = app.builder.get_relative_uri(docname, target)
            if url.fragment:
                node["refuri"] += "#" + url.fragment
