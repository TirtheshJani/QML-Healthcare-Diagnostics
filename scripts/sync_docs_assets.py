"""Refresh every generated docs-site asset in one command.

Run this before ``mkdocs serve`` / ``mkdocs build`` (the deploy workflow runs it too):

  1. Regenerate the results table + chart data   (scripts/build_docs_tables.py)
  2. Regenerate the client-side demo model        (scripts/export_demo_model.py)
  3. Copy reports/figures/*.png into docs/assets/figures/

Figures live in reports/figures/ as the single source of truth. MkDocs only serves files
under its docs_dir, so they are copied (not committed twice) into docs/assets/figures/, which
is gitignored.
"""

from __future__ import annotations

import shutil

from qml_healthcare.config import FIGURES_DIR, PROJECT_ROOT

DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "assets" / "figures"


def sync_figures() -> int:
    """Copy every PNG from reports/figures/ into docs/assets/figures/. Returns the count."""
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pngs = sorted(FIGURES_DIR.glob("*.png"))
    for src in pngs:
        shutil.copy2(src, DOCS_FIGURES_DIR / src.name)
    return len(pngs)


def main() -> None:
    # Imported here because scripts/ is only on sys.path[0] when run as a script.
    import build_docs_tables
    import export_demo_model

    build_docs_tables.main()
    export_demo_model.main()
    n = sync_figures()
    print(f"Copied {n} figures into {DOCS_FIGURES_DIR}.")


if __name__ == "__main__":
    main()
