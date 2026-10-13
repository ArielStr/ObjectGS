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


def show_box_selector(
    image_path,
    candidates_dir,
    checkpoint,
    repo_dir=None,
    device="cuda",
):
    """Interactive box prompting and SAM2 candidate selection in Colab."""
    import torch
    from matplotlib.patches import Rectangle

    from select_object_sam2 import generate_box_candidates

    image_path = Path(image_path)
    candidates_dir = Path(candidates_dir)
    checkpoint = Path(checkpoint)

    image = np.asarray(Image.open(image_path).convert("RGB")).copy()
    height, width = image.shape[:2]

    style = {"description_width": "45px"}

    x1 = widgets.IntSlider(
        value=width // 4, min=0, max=width - 1,
        description="x1", style=style,
    )
    y1 = widgets.IntSlider(
        value=height // 4, min=0, max=height - 1,
        description="y1", style=style,
    )
    x2 = widgets.IntSlider(
        value=3 * width // 4, min=0, max=width - 1,
        description="x2", style=style,
    )
    y2 = widgets.IntSlider(
        value=3 * height // 4, min=0, max=height - 1,
        description="y2", style=style,
    )

    # Editable coordinates, synchronized with sliders
    coordinate_inputs = [
        widgets.BoundedIntText(
            value=slider.value,
            min=slider.min,
            max=slider.max,
            description=slider.description,
            layout=widgets.Layout(width="165px"),
        )
        for slider in (x1, y1, x2, y2)
    ]

    for slider, number in zip(
        (x1, y1, x2, y2), coordinate_inputs
    ):
        widgets.link((slider, "value"), (number, "value"))

    preview_area = widgets.Output()
    results_area = widgets.Output()

    generate_button = widgets.Button(
        description="Generate SAM2 candidates",
        button_style="primary",
    )

    candidate_dropdown = widgets.Dropdown(
        options=[],
        description="Candidate:",
        disabled=True,
    )

    confirm_button = widgets.Button(
        description="Confirm mask",
        button_style="success",
        disabled=True,
    )

    state = {"masks": None, "scores": None, "box": None}

    def update_preview(change=None):
        with preview_area:
            clear_output(wait=True)

            left, right = sorted([x1.value, x2.value])
            top, bottom = sorted([y1.value, y2.value])

            fig, ax = plt.subplots(figsize=(7, 8))
            ax.imshow(image)
            ax.add_patch(Rectangle(
                (left, top),
                right - left,
                bottom - top,
                edgecolor="lime",
                facecolor="none",
                linewidth=2,
            ))
            # Coordinate grid over the source image
            import matplotlib.ticker as ticker

            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)

            ax.xaxis.set_major_locator(
                ticker.MultipleLocator(250)
            )
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(250)
            )

            ax.grid(
                True,
                color="cyan",
                alpha=0.45,
                linewidth=0.7,
            )

            ax.tick_params(
                axis="both",
                labelsize=8,
            )

            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")

            plt.show()
            plt.close(fig)

    def invalidate_candidates(change=None):
        state["masks"] = None
        state["scores"] = None
        state["box"] = None
        candidate_dropdown.options = []
        candidate_dropdown.disabled = True
        confirm_button.disabled = True
        update_preview()

    for slider in [x1, y1, x2, y2]:
        slider.observe(invalidate_candidates, names="value")

    def on_generate(_):
        with results_area:
            clear_output(wait=True)

            left, right = sorted([x1.value, x2.value])
            top, bottom = sorted([y1.value, y2.value])

            if right <= left or bottom <= top:
                print("Invalid bounding box.")
                return

            generate_button.disabled = True
            confirm_button.disabled = True

            try:
                box = np.array(
                    [left, top, right, bottom],
                    dtype=np.float32,
                )

                print("Generating SAM2 candidates...", flush=True)

                masks, scores = generate_box_candidates(
                    image=image,
                    checkpoint=checkpoint,
                    config="configs/sam2.1/sam2.1_hiera_s.yaml",
                    device=device,
                    box=box,
                )

                state["masks"] = masks
                state["scores"] = scores
                state["box"] = box

                fig, axes = plt.subplots(
                    1, len(masks),
                    figsize=(5 * len(masks), 7),
                    squeeze=False,
                )

                for i, (mask, score) in enumerate(zip(masks, scores)):
                    overlay = image.copy()
                    region = mask.astype(bool)

                    overlay[region] = (
                        0.45 * overlay[region]
                        + 0.55 * np.array([0, 220, 80])
                    ).astype(np.uint8)

                    ax = axes[0, i]
                    ax.imshow(overlay)
                    ax.set_title(
                        f"Candidate {i + 1} | Score {score:.3f}"
                    )
                    ax.axis("off")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

                candidate_dropdown.options = [
                    (
                        f"Candidate {i + 1} | Score {float(score):.3f}",
                        i,
                    )
                    for i, score in enumerate(scores)
                ]
                candidate_dropdown.value = int(np.argmax(scores))
                candidate_dropdown.disabled = False
                confirm_button.disabled = False

            except Exception as exc:
                print("SAM2 generation failed:", exc)

            finally:
                generate_button.disabled = False

    def on_confirm(_):
        with results_area:
            clear_output(wait=True)

            if state["masks"] is None:
                print("Generate candidates first.")
                return

            selected = candidate_dropdown.value
            mask = state["masks"][selected].astype(bool)

            if not mask.any():
                print("Selected mask is empty.")
                return

            candidates_dir.mkdir(parents=True, exist_ok=True)

            final_path = candidates_dir / "final_mask.png"

            Image.fromarray(
                mask.astype(np.uint8) * 255
            ).save(final_path)

            metadata = {
                "selection_method": "box_prompt",
                "selected_candidate": int(selected + 1),
                "sam_score": float(state["scores"][selected]),
                "box_xyxy": [
                    int(v) for v in state["box"]
                ],
                "final_mask": str(final_path),
            }

            (candidates_dir / "selection.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            isolated = np.full_like(image, 255)
            isolated[mask] = image[mask]

            preview = Image.fromarray(isolated)
            preview.thumbnail((480, 640))

            print("Mask confirmed:", final_path)
            display(preview)

    generate_button.on_click(on_generate)
    confirm_button.on_click(on_confirm)

    display(widgets.VBox([
        widgets.HTML("<h3>Define object bounding box</h3>"),
        widgets.HBox([x1, x2]),
        widgets.HBox([y1, y2]),
        widgets.HTML("<b>Exact pixel coordinates:</b>"),
        widgets.HBox(coordinate_inputs),
        preview_area,
        generate_button,
        results_area,
        candidate_dropdown,
        confirm_button,
    ]))

    update_preview()

    return {
        "sliders": (x1, y1, x2, y2),
        "generate_button": generate_button,
        "candidate_dropdown": candidate_dropdown,
        "confirm_button": confirm_button,
    }
