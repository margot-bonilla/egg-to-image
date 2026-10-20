"""Convert the Markdown docs to LaTeX (and PDF), with the Mermaid diagrams as TikZ figures.

    python scripts/docs_to_latex.py [docs/how_it_works.md docs/architecture_deep_dive.md] [--out docs/latex] [--no-pdf]

pandoc does the Markdown -> LaTeX conversion; it leaves Mermaid blocks as code, so each one is
translated here into a TikZ graph laid out automatically by TikZ's graph-drawing engine
(``layered layout``, LuaLaTeX only). Supports the Mermaid subset our docs use: ``flowchart TB|LR``,
``ID["label"]`` nodes, chains ``A --> B --> C``, ``-->|"label"|``, ``-.->``, ``-. "label" .-``,
``== "label" ==>``; ``subgraph`` boxes are dropped, their title shown in bold on their first node. Every diagram is laid
out top to bottom, which reads better than wide left-to-right graphs on a portrait page.

pandoc: ``$PANDOC``, else ``pandoc`` on PATH, else the copy bundled with RStudio.
Needs LuaLaTeX (MacTeX / TeX Live) for the PDF; the .tex files are standalone (compile them with
``lualatex``, twice for the table of contents: TikZ's graph drawing needs LuaLaTeX).
"""
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RSTUDIO_PANDOC = "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc"

HEADER = r"""
\usepackage{fontspec}
% Symbols missing from the text fonts (arrows, maths signs) come from these fallbacks.
\directlua{luaotfload.add_fallback("symbols", {"STIX Two Math:mode=harf;", "DejaVu Sans:mode=harf;"})}
\setmainfont{STIX Two Text}[RawFeature={fallback=symbols}]
\setmonofont{DejaVu Sans Mono}[Scale=0.85, RawFeature={fallback=symbols}]
\usepackage{unicode-math}
\setmathfont{STIX Two Math}
\usepackage{tikz}
\usetikzlibrary{graphs, graphdrawing, quotes, arrows.meta}
\usegdlibrary{layered}
\usepackage{adjustbox}
\usepackage{float}
\usepackage{fvextra}
\fvset{breaklines=true, breakanywhere=true}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true, breakanywhere=true}
\renewcommand{\arraystretch}{1.2}
% Let identifiers like learn_temperature=True break after underscores (e.g. in table cells).
\let\oldunderscore\_
\renewcommand{\_}{\oldunderscore\allowbreak}
"""

NODE = re.compile(r'([A-Za-z0-9_]+)(?:\["((?:[^"\\]|\\.)*)"\])?')
EDGES = [  # (regex, style, arrow)
    (re.compile(r'-->\|"([^"]*)"\|'), "solid", True),
    (re.compile(r'-\.->\|"([^"]*)"\|'), "dashed", True),
    (re.compile(r'-\.\s*"([^"]*)"\s*\.(->|-)'), "dashed", None),  # arrow iff it ends with ->
    (re.compile(r'==\s*"([^"]*)"\s*==>'), "thick", True),
    (re.compile(r'-->'), "solid", True),
    (re.compile(r'-\.->'), "dashed", True),
    (re.compile(r'==>'), "thick", True),
    (re.compile(r'---'), "solid", False),
]


def latex_escape(text: str) -> str:
    out = []
    for ch in text:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(ch, ch))
    return "".join(out)


def label(text: str) -> str:
    return r" \\ ".join(latex_escape(part) for part in re.split(r"<br\s*/?>", text))


def parse_mermaid(src: str):
    """-> (direction, {id: (group title or None, label)}, [(a, b, style, arrow, label)])"""
    lines = [l.strip() for l in src.strip().splitlines()]
    direction = lines[0].split()[1]
    nodes, edges, group = {}, [], None  # group: title of the subgraph whose first node is pending
    for line in lines[1:]:
        if line.startswith("subgraph"):
            m = re.match(r'subgraph\s+\w+\["([^"]*)"\]', line)
            group = m.group(1) if m else line[len("subgraph"):].strip()
            continue
        if not line or line == "end" or line.startswith("%%"):
            continue
        pos, prev, pending = 0, None, None
        while pos < len(line):
            if line[pos].isspace():
                pos += 1
                continue
            m = NODE.match(line, pos)  # node ids start with a letter/digit, edges with "-" or "="
            if m:
                node_id, node_label = m.group(1), m.group(2)
                if node_id not in nodes:
                    nodes[node_id] = (group, node_label if node_label is not None else node_id)
                    group = None
                elif node_label is not None:
                    nodes[node_id] = (nodes[node_id][0], node_label)
                if pending is not None:
                    edges.append((prev, node_id, *pending))
                prev, pending, pos = node_id, None, m.end()
                continue
            for regex, style, arrow in EDGES:
                e = regex.match(line, pos)
                if e:
                    text = e.group(1) if e.groups() else None
                    if arrow is None:
                        arrow = e.group(2) == "->"
                    pending, pos = (style, arrow, text), e.end()
                    break
            else:
                raise ValueError(f"can't parse Mermaid line: {line!r} at {line[pos:]!r}")
    return direction, nodes, edges


def mermaid_to_tikz(src: str) -> str:
    _direction, nodes, edges = parse_mermaid(src)
    body = [f"  {n} [as={{{(r'\textbf{' + label(title) + r'} \\ ') if title else ''}{label(text)}}}];"
            for n, (title, text) in nodes.items()]
    for a, b, style, arrow, text in edges:
        opts = {"dashed": ["dashed"], "thick": ["very thick"], "solid": []}[style]
        if text:
            opts.append(f'"{{{label(text)}}}"')
        body.append(f"  {a} {'->' if arrow else '--'}[{', '.join(opts)}] {b};")
    labelled = any(text for *_, text in edges)  # edge labels need room between nodes
    sep = "level sep=16mm, sibling sep=32mm" if labelled else "level sep=9mm, sibling sep=6mm"
    return "\n".join([
        r"\begin{figure}[H]\centering",
        r"\begin{adjustbox}{max width=\linewidth, max totalheight=0.8\textheight}",
        r"\begin{tikzpicture}[>={Stealth[length=2mm]},"
        r" box/.style={draw, rounded corners=2pt, align=center, font=\footnotesize, inner sep=4pt, fill=black!4},"
        r" every edge quotes/.style={font=\scriptsize, fill=white, inner sep=1pt, align=center, auto}]",
        rf"\graph [layered layout, grow=down, {sep}, nodes={{box}},"
        r" edges={semithick}] {",
        *body,
        "};",
        r"\end{tikzpicture}",
        r"\end{adjustbox}",
        r"\end{figure}",
    ])


def size_table_columns(text: str) -> str:
    """Give each pipe-table column a width proportional to its cell lengths (pandoc reads relative
    widths from the dashes of the separator row), so wide tables wrap instead of running off the page."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [[c.strip() for c in re.split(r"(?<!\\)\|", l.strip())[1:-1]] for l in lines[i:j]]
            widths = []
            for k in range(len(rows[1])):
                lengths = [len(r[k]) if k < len(r) else 0 for r in rows[:1] + rows[2:]]
                # mostly the typical cell length, so one long cell doesn't take the whole row
                widths.append(round(min(0.7 * sum(lengths) / len(lengths) + 0.3 * max(lengths), 50)))
            seps = []
            for cell, w in zip(rows[1], widths):
                dashes = "-" * max(w, 6)
                seps.append((":" if cell.startswith(":") else "") + dashes + (":" if cell.endswith(":") else ""))
            out += [lines[i], "|" + "|".join(seps) + "|"] + lines[i + 2:j]
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def find_pandoc() -> str:
    for candidate in (os.environ.get("PANDOC"), shutil.which("pandoc"), RSTUDIO_PANDOC):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("pandoc not found: install it, or set PANDOC=/path/to/pandoc")


def convert(md: Path, out: Path, pandoc: str, pdf: bool) -> None:
    text = md.read_text()
    text = re.sub(r"```mermaid\n(.*?)```", lambda m: "```{=latex}\n" + mermaid_to_tikz(m.group(1)) + "\n```",
                  text, flags=re.S)
    text = re.sub(r"\]\(([\w_]+)\.md(#[^)]*)?\)", r"](\1.pdf\2)", text)  # links between the docs -> PDFs
    text = size_table_columns(text)
    tmp = out / f"{md.stem}.pandoc.md"
    tmp.write_text(text)
    tex = out / f"{md.stem}.tex"
    (out / "header.tex").write_text(HEADER)
    subprocess.run([pandoc, str(tmp), "-s", "-o", str(tex), "--columns", "40",
                    # pandoc's Markdown (it wraps wide tables), with GitHub's habits: lists right after a
                    # paragraph, bare URLs, ``` fences, $...$ maths.
                    "--from", "markdown+lists_without_preceding_blankline+autolink_bare_uris+tex_math_dollars"
                              "-implicit_figures-smart",
                    "--pdf-engine=lualatex", "--include-in-header", str(out / "header.tex"),
                    "-V", "geometry:margin=2.2cm", "-V", "documentclass=article", "-V", "fontsize=11pt",
                    "-V", "colorlinks=true", "--toc", "--syntax-highlighting=tango"],
                   check=True)
    tmp.unlink()
    (out / "header.tex").unlink()  # inlined into the .tex by pandoc
    print(f"wrote {tex.relative_to(REPO)}")
    if pdf:
        for _ in range(2):  # twice, for the table of contents
            run = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                                 cwd=out, capture_output=True, text=True)
            if run.returncode:
                raise SystemExit(f"lualatex failed on {tex.name}:\n" + "\n".join(run.stdout.splitlines()[-25:]))
        for ext in (".aux", ".log", ".out", ".toc"):
            (out / f"{md.stem}{ext}").unlink(missing_ok=True)
        print(f"wrote {(out / f'{md.stem}.pdf').relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("docs", nargs="*", type=Path,
                        default=[REPO / "docs/how_it_works.md", REPO / "docs/architecture_deep_dive.md"])
    parser.add_argument("--out", type=Path, default=REPO / "docs/latex")
    parser.add_argument("--no-pdf", action="store_true", help="only write the .tex files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pandoc = find_pandoc()
    for md in args.docs:
        convert(md.resolve(), args.out.resolve(), pandoc, not args.no_pdf)


if __name__ == "__main__":
    main()
