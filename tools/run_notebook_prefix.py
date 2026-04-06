from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def should_stop(cell_source: str) -> bool:
    # Stop before long-running training cells.
    markers = [
        "## 5. Model Training",
        "Load local BERT model",
        "# Training loop",
        "num_epochs =",
    ]
    return any(m in cell_source for m in markers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    nb_path: Path = args.notebook
    if not nb_path.exists():
        raise SystemExit(f"Notebook not found: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)

    # Create a truncated copy of the notebook up to (but excluding) training.
    new_cells = []
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source") or ""
            if should_stop(src):
                break
        new_cells.append(cell)

    nb.cells = new_cells

    # Execute in the notebook's directory so cells like `cd ..` behave as intended.
    resources = {"metadata": {"path": str(nb_path.parent)}}
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources=resources)
    try:
        client.execute()
    except CellExecutionError:
        # Re-raise with the original rich error message.
        raise

    out_path = args.out or (nb_path.with_suffix(".executed_prefix.ipynb"))
    nbformat.write(nb, out_path)
    print(f"Executed prefix notebook saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
