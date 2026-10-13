"""Interactive SAM2 mask selection UI for Google Colab."""

import json
import subprocess
import sys
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from PIL import Image


def show_mask_selector(
    image_path,
    candidates_dir,
    checkpoint,
    repo_dir=None,
    device="cuda",
):
    """Display candidates and run refinement after one user selection."""

    image_path = Path(image_path)
    candidates_dir = Path(candidates_dir)
    checkpoint = Path(checkpoint)
    repo_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1]

    metadata_path = candidates_dir / "candidates.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = metadata["candidates"]

    if not entries:
        raise ValueError("No mask candidates available.")

    image = np.asarray(Image.open(image_path).convert("RGB")).copy()

    # Display all candidates.
    cols = 4
    rows = (len(entries) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(15, rows * 5),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for i, entry in enumerate(entries):
        mask_path = candidates_dir / "candidates" / entry["file"]
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0

        if mask.shape != image.shape[:2]:
            raise ValueError(f"Mask size mismatch: {mask_path}")

        overlay = image.copy()
        overlay[mask] = (
            0.45 * overlay[mask] +
            0.55 * np.array([0, 220, 80])
        ).astype(np.uint8)

        axes.flat[i].imshow(overlay)
        axes.flat[i].set_title(f"Mask {entry['id']}")

    plt.tight_layout()
    plt.show()

    dropdown = widgets.Dropdown(
        options=[
            (f"Mask {entry['id']}", entry["id"])
            for entry in entries
        ],
        description="Object:",
        value=None,
    )

    button = widgets.Button(
        description="Select object & refine",
        button_style="success",
        icon="check",
    )

    result_area = widgets.Output()

    def on_select_clicked(_):
        with result_area:
            clear_output(wait=True)

            selected_id = dropdown.value

            if selected_id is None:
                print("Please select a mask first.")
                return

            button.disabled = True

            try:
                cmd = [
                    sys.executable,
                    str(repo_dir / "scripts/select_object_sam2.py"),
                    "--image", str(image_path),
                    "--checkpoint", str(checkpoint),
                    "--output", str(candidates_dir),
                    "--select-mask", str(selected_id),
                    "--device", device,
                ]

                print(f"Selected Mask {selected_id}")
                print("Running automatic refinement...", flush=True)

                result = subprocess.run(
                    cmd,
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                )

                print(result.stdout)

                if result.returncode != 0:
                    print(result.stderr)
                    raise RuntimeError("Automatic refinement failed.")

                selection = json.loads(
                    (candidates_dir / "selection.json").read_text(
                        encoding="utf-8"
                    )
                )

                print("Automatic candidate:",
                      selection["refinement_candidate"])
                print("Final mask:", selection["final_mask"])

                final_mask = np.asarray(
                    Image.open(candidates_dir / "final_mask.png").convert("L")
                ) > 0

                isolated = np.full_like(image, 255)
                isolated[final_mask] = image[final_mask]

                preview = Image.fromarray(isolated)
                preview.thumbnail((480, 640))
                display(preview)

            except Exception as exc:
                print(f"Selection failed: {exc}")

            finally:
                button.disabled = False

    button.on_click(on_select_clicked)

    display(widgets.VBox([
        widgets.HTML("<h3>Select the object to reconstruct</h3>"),
        dropdown,
        button,
        result_area,
    ]))

    return dropdown, button, result_area
