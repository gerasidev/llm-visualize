"""Build a self-contained HTML debugger from a trace.json file."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def build_visualizer(
    trace_path: str | Path,
    output_path: str | Path,
    frontend_dir: str | Path = FRONTEND,
) -> Path:
    frontend_dir = Path(frontend_dir)
    trace_path = Path(trace_path)
    output_path = Path(output_path)

    html = (frontend_dir / "index.html").read_text(encoding="utf-8")
    css = (frontend_dir / "styles.css").read_text(encoding="utf-8")
    js = (frontend_dir / "app.js").read_text(encoding="utf-8")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        "<script>window.TRACE_DATA = null;</script>",
        "<script>window.TRACE_DATA = "
        + json.dumps(trace, ensure_ascii=False)
        + ";</script>",
    )
    html = html.replace(
        '<script src="app.js"></script>',
        f"<script>\n{js}\n</script>",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="Path to trace.json")
    parser.add_argument("output", help="Path to generated HTML")
    args = parser.parse_args()

    result = build_visualizer(args.trace, args.output)
    print(result)
