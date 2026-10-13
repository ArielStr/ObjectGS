# ObjectGS

### Mask-Guided Object-Centric Gaussian Splatting

ObjectGS is an experimental object-centric 3D Gaussian Splatting
pipeline for reconstructing compact 3D representations of individual
objects from multi-view RGB images and segmentation masks.

The project builds on the original Graphdeco 3D Gaussian Splatting
implementation and modifies both the training objective and
densification behavior for small foreground objects.


## What Makes ObjectGS Different?

ObjectGS is not simply Vanilla 3D Gaussian Splatting applied to
segmented images. It modifies the optimization objective, Gaussian
densification behavior, and sparse initialization to focus the
reconstruction on a selected foreground object.

### Vanilla 3DGS vs. ObjectGS

| Component | Vanilla 3DGS | ObjectGS |
|---|---|---|
| Reconstruction target | Full scene | Selected foreground object |
| Photometric supervision | Full-image loss | Foreground-normalized masked loss |
| Background handling | Background is reconstructed | Outside-mask penalty suppresses unwanted rendering |
| Densification | Original screen-space gradients | Mask-fraction-normalized densification gradients |
| Sparse initialization | COLMAP scene points | Mask-consistent object-only SfM points |
| Representation | Scene-level Gaussians | Compact object-centric Gaussians |

### 1. Object-Focused Photometric Loss

Standard 3DGS optimizes image reconstruction across the entire view.
ObjectGS instead concentrates reconstruction error on the selected
object, using a binary foreground mask.

For a rendered image $\hat{I}$, target image $I$, and binary mask $M$,
the foreground-normalized pixelwise reconstruction term is:

$$
L_{\mathrm{object}} =
\frac{\sum_p M_p \, \ell(I_p,\hat{I}_p)}
{\sum_p M_p}
$$

Here, $\ell$ denotes the pixelwise photometric error. The foreground
normalization prevents a small object from being overwhelmed by the
much larger background area.

### 2. Outside-Mask Penalty

A foreground-only reconstruction loss does not by itself discourage
Gaussians from rendering outside the object silhouette.

ObjectGS adds an outside-mask penalty:

$$
L = L_{\mathrm{object}}
+ \lambda_{\mathrm{outside}} L_{\mathrm{outside}}
$$

The outside-mask term is computed from the rendered RGB image:

$$
L_{\mathrm{outside}} =
\frac{
\sum_{p,c}(1-M_p)\,|\hat{I}_{p,c}|
}{
C\sum_p(1-M_p)+\epsilon
}
$$

where $C$ is the number of image channels (3 for RGB) and
$\epsilon$ provides numerical stability.

The default configuration uses $\lambda_{\mathrm{outside}} = 1.0$.

Unlike an opacity penalty, this term penalizes nonzero rendered RGB
values outside the object mask. It encourages a dark background
outside the target silhouette, but does not directly guarantee
zero Gaussian opacity there.

### 3. Mask-Normalized Densification

**This is a key algorithmic modification.**

Foreground-normalized loss changes the scale of the gradients used
during optimization. For objects occupying only a small fraction of
the image, the original 3DGS densification heuristic can receive
disproportionately large screen-space gradient signals.

In our experiments, masked training without this correction caused
severe over-densification.

ObjectGS rescales the densification gradient signal by the fraction
of foreground pixels in the training view:

$$
f_{\mathrm{mask}} = \frac{\sum_p M_p}{H W}
$$

$$
g_{\mathrm{densification}}^{\mathrm{ObjectGS}}
= f_{\mathrm{mask}} \cdot g_{\mathrm{densification}}
$$

This correction rescales the accumulated screen-space gradient
statistics used for Gaussian clone/split decisions. It does not
rescale the gradients used by the optimizer or replace the
object-focused reconstruction objective.

The ObjectGS training wrapper enables mask-normalized densification
by default, even though the patched upstream runtime exposes it
as an optional flag.

It preserves foreground-focused optimization while bringing the
densification statistics closer to the scale expected by the
original 3DGS heuristic.

### 4. Object-Only Sparse Initialization

ObjectGS also filters the COLMAP sparse reconstruction using
multi-view mask consistency.

A 3D point is retained when its observations agree with the
foreground masks across the registered views. By default, the
consistency threshold is 1.0, requiring every observation of a
retained point to lie inside the object mask.

This gives training an object-focused sparse initialization rather
than starting from all scene points.

### Why It Matters

The combination of foreground-focused supervision, outside-mask
regularization, object-only initialization, and corrected
densification produces a substantially more compact representation
in our experiments.

The visual comparisons below show reductions of 76.9% to 98.4% in
Gaussian count relative to the corresponding Vanilla 3DGS runs.

These results are preliminary and should not be interpreted as a
controlled reconstruction-quality benchmark.

---

## Visual Results

ObjectGS reconstructs the target object rather than the entire scene.
Each comparison uses a synchronized side-by-side animation.

### Iron Man — Multi-view Images

<table>
<tr>
<td align="center"><b>Vanilla 3DGS — 7K</b></td>
<td align="center"><b>ObjectGS — 7K</b></td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="assets/object_02/ironman_comparison_7k.gif" width="100%">
</td>
</tr>
<tr>
<td align="center">399,965 Gaussians</td>
<td align="center">6,295 Gaussians</td>
</tr>
</table>

**Result:** 63.5× fewer Gaussians (98.4% reduction).

### Cap — Video Input (54 Frames)

54 frames extracted from a video at 3 FPS, with segmentation
masks propagated using SAM2 Video Tracking.

<table>
<tr>
<td align="center"><b>Vanilla 3DGS — 7K</b></td>
<td align="center"><b>ObjectGS — 7K</b></td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="assets/object_03/cap_comparison_7k.gif" width="100%">
</td>
</tr>
<tr>
<td align="center">175,251 Gaussians</td>
<td align="center">24,312 Gaussians</td>
</tr>
</table>

**Result:** 7.2× fewer Gaussians (86.1% reduction).

### Book — Video Input (55 Frames)

55 frames reconstructed using COLMAP, with object masks generated
using SAM2. Both models trained for 7,000 iterations at quarter resolution.

<table>
<tr>
<td align="center"><b>Vanilla 3DGS — 7K</b></td>
<td align="center"><b>ObjectGS — 7K</b></td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="assets/object_04/book_comparison_7k.gif" width="100%">
</td>
</tr>
<tr>
<td align="center">223,867 Gaussians</td>
<td align="center">51,679 Gaussians</td>
</tr>
</table>

**Result:** 4.3× fewer Gaussians (76.9% reduction).
ObjectGS trained in 105.6 s versus 250.8 s for Vanilla.

### Quantitative Summary

| Metric | Iron Man Vanilla | Iron Man ObjectGS | Cap Vanilla | Cap ObjectGS | Book Vanilla | Book ObjectGS |
|---|---:|---:|---:|---:|---:|---:|
| Iterations | 7,000 | 7,000 | 7,000 | 7,000 | 7,000 | 7,000 |
| Gaussians | 399,965 | 6,295 | 175,251 | 24,312 | 223,867 | 51,679 |
| PLY size | — | — | 41.45 MB | 5.75 MB | 52.95 MB | 12.22 MB |
| Training time | Not measured | Not measured | Not measured | 285.18 s | 250.8 s | 105.6 s |

The cap baseline used `--eval`, whereas ObjectGS trained on all
registered frames. The training splits therefore differ, so this
is a qualitative comparison rather than a controlled benchmark.

Results are preliminary and cover three captured objects.

## Google Colab — Recommended

The easiest way to run ObjectGS is through the end-to-end Google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArielStr/ObjectGS/blob/main/notebooks/ObjectGS_End_to_End.ipynb)

### Requirements

- A Google account with access to Google Colab.
- A Colab runtime with an NVIDIA GPU (Runtime > Change runtime type > GPU).
- A video of a mostly stationary object filmed from multiple viewpoints.
- Sufficient GPU runtime, disk space, and time for reconstruction and training.

For better results, keep the object visible and in focus, move around it
slowly, capture overlapping viewpoints, and avoid strong motion blur.

### Video Capture Guidelines

For the video workflow, we recommend starting with a short,
continuous recording that makes approximately **one full orbit
(360 degrees) around a mostly stationary object**.

A practical starting configuration is:

| Parameter | Suggested starting point |
|---|---|
| Video duration | Approximately 15 seconds |
| Camera movement | One slow, continuous orbit around the object |
| Frame extraction | `FPS = 2.0` |
| Approximate extracted frames | 30 frames |
| Quick pipeline test | `ITERATIONS = 1000` |
| Longer reconstruction experiment | `ITERATIONS = 7000` |

These are practical starting points, not strict requirements.
The appropriate frame count depends on the object's texture,
visibility, viewpoint overlap, and camera motion.

**Capture recommendations:**

- Keep the object mostly stationary while moving the camera around it.
- Try to complete one full orbit, capturing multiple visible sides.
- Move slowly and maintain overlap between consecutive viewpoints.
- Keep the object sharp, well lit, and visible throughout the video.
- Avoid motion blur, sudden camera movements, and large occlusions.
- Prefer objects with visible texture and geometric detail.

**Frame count versus processing cost:** Increasing `FPS` extracts
more frames from the same video. This can improve viewpoint coverage,
but it also increases SAM2 tracking work, COLMAP feature extraction
and matching, reconstruction overhead, and the number of training
views. More frames do not automatically mean better reconstruction.

For example, a 15-second video sampled at 2 FPS yields approximately
30 frames. This is a reasonable first experiment, not a guarantee
that COLMAP will register every frame.

### Training Iterations and Reconstruction Quality

The notebook defaults to `ITERATIONS = 1000` to make the initial
end-to-end test relatively quick.

For a more developed reconstruction, **we recommend trying
`ITERATIONS = 7000`** after confirming that the pipeline and object
selection work correctly.

| Setting | Intended use |
|---|---|
| 1,000 iterations | Quick test, pipeline validation, early visual feedback |
| 7,000 iterations | Longer optimization, generally more developed reconstruction |

More iterations increase training time and do not guarantee
improved quality in every case. Reconstruction quality also depends
on camera coverage, mask accuracy, COLMAP registration, and the
object's visual characteristics.

### Step-by-step execution

Open the notebook, enable a GPU, and execute the cells in order.

**Step 1 — Setup (Cell 2)**

Clone the public ObjectGS repository and install the dependencies,
including COLMAP, SAM2, and the patched 3D Gaussian Splatting runtime.

No GitHub token is required for cloning.

Full setup logs are saved to `/content/ObjectGS/setup_full.log`.

**Step 2 — Upload video (Cell 4)**

The default input configuration is:

    INPUT_SOURCE = "upload"
    FPS = 1.0
    ITERATIONS = 1000

Running this cell opens a video upload dialog.

`FPS` controls how many frames are extracted per second. Increasing
it may improve viewpoint coverage, but increases computation.

`ITERATIONS` controls the training duration. Start with 1,000 iterations
for a quick experiment, then consider 7,000 for a longer run.

The notebook also supports Drive input using `INPUT_SOURCE = "drive"`
and a user-configured `DRIVE_VIDEO` path.

**Step 3 — Generate object candidates (Cell 6)**

Run the candidate generation cell and wait for it to finish.

**Step 4 — Select the target object (Cell 8)**

Use the interactive object selection interface to select the object
you want to reconstruct. The default interface supports bounding-box
selection. Verify the selected mask before continuing.

You do not need to create masks manually for every frame.

**Step 5 — Run ObjectGS (Cell 10)**

Run the remaining end-to-end pipeline:

1. Extract video frames.
2. Track object masks using SAM2.
3. Estimate camera poses and sparse geometry using COLMAP.
4. Filter sparse points using multi-view mask consistency.
5. Prepare the undistorted ObjectGS scene.
6. Train the object-centric Gaussian Splatting model.

The pipeline may stop if too few frames are registered or the
reconstruction does not pass its quality checks.

**Step 6 — Review results (Cells 11 and 13)**

The experiment is saved under:

    /content/objectgs_runs/<run_id>/

For a completed 1,000-iteration run, the trained model is typically:

    /content/objectgs_runs/<run_id>/objectgs_1000/point_cloud/iteration_1000/point_cloud.ply

Open the PLY using [SuperSplat](https://superspl.at/editor).

**Step 7 — Download results (Cell 15)**

The default export configuration is:

    EXPORT_MODE = "download"

The cell creates a ZIP archive of the experiment and downloads it.

Alternatively, set `EXPORT_MODE = "drive"` and configure your
own `DRIVE_EXPORT_DIR`.

### Troubleshooting

| Problem | Suggested action |
|---|---|
| GPU unavailable | Enable a GPU runtime before setup |
| Setup failure | Inspect `/content/ObjectGS/setup_full.log` |
| Incorrect object selection | Rerun selection and inspect the mask |
| Too few COLMAP-registered frames | Capture overlapping viewpoints or increase FPS |
| GPU out of memory | Reduce the number of extracted frames or use a larger GPU |
| Colab disconnects | Export results before the runtime is deleted |

Colab runtime storage is temporary. Download or export results
before disconnecting.

---

## Quick Start

ObjectGS supports two workflows: video input and multi-view images.
Both run segmentation, COLMAP, object-only SfM, and ObjectGS training.

### Installation

    git clone https://github.com/ArielStr/ObjectGS.git
    cd ObjectGS
    bash setup.sh --mode gpu --install-colmap --install-sam2

A CUDA GPU is required for training.
The SAM2 checkpoint is downloaded to `checkpoints/sam2.1_hiera_small.pt`.

### Run from Video

    python scripts/run_pipeline.py --video data/my_object/video.mp4 --point 1150 1320 --fps 3 --sam-checkpoint checkpoints/sam2.1_hiera_small.pt --output outputs/my_video --iterations 7000 --sift-gpu off

The `--point X Y` coordinates must be inside the target object
in the first extracted frame, at its original resolution.
SAM2 Video Tracking propagates the mask through subsequent frames.

### Run from Images

    python scripts/run_pipeline.py --images data/my_object/images --prompts data/my_object/prompts.json --sam-checkpoint checkpoints/sam2.1_hiera_small.pt --output outputs/my_images --iterations 7000 --sift-gpu off

Provide per-image SAM2 box or point prompts in JSON format.
See [Generate masks with SAM2](#generate-masks-with-sam2).

### Outputs

The final model is saved under:

    outputs/my_video/objectgs_7000/point_cloud/iteration_7000/point_cloud.ply

The output directory also contains masks, COLMAP reconstruction,
object-only SfM, prepared scene data, logs, and pipeline summary.
Open the final PLY in [SuperSplat](https://superspl.at/editor).

### Resume

Existing stages are skipped when their outputs pass validation.
Use a fresh output directory when changing inputs or parameters.
The pipeline does not automatically detect configuration changes.
Use `--force` only with a new or empty output directory.

On headless Colab, `--sift-gpu off` avoids COLMAP OpenGL issues.
ObjectGS training still uses CUDA.

## Motivation

Standard 3D Gaussian Splatting is designed for full-scene
reconstruction.

For object-centric reconstruction, we instead want the photometric
objective to focus only on pixels belonging to the target object.

A straightforward masked loss introduces an important side effect:
when the reconstruction error is normalized by the much smaller
foreground area, the screen-space gradients used by the standard 3DGS
densification heuristic become substantially larger.

In our experiments this caused severe over-densification.


## Core idea

ObjectGS separates the scale of the reconstruction objective from the
scale used by the densification heuristic.

For each training view, the object reconstruction term is:

    object_loss =
        masked_photometric_error / foreground_area

An additional outside-mask penalty discourages rendered RGB outside
the object silhouette:

    loss =
        object_loss
        + lambda_outside * outside_loss

For densification only, the accumulated screen-space gradient can be
rescaled using:

    densification_gradient *= mask_fraction

where:

    mask_fraction =
        foreground_pixels / image_pixels

This keeps the reconstruction objective focused on the object while
bringing the densification signal back toward the scale expected by
the original 3DGS heuristic.

The correction can be disabled for ablation experiments.


## Pipeline

ObjectGS supports two input modes that share the same reconstruction
and training stages.

    Multi-view images              Input video
           |                           |
           |                    Frame extraction
           |                           |
    SAM2 image segmentation     SAM2 video tracking
           |                           |
           +------------+--------------+
                        |
                 RGB images + masks
                        |
                  COLMAP SfM
                        |
             Select largest registered
                reconstruction
                        |
                Image undistortion
                        |
             Multi-view mask consistency
                        |
                 Object-only SfM
                        |
                Scene preparation
                        |
                 ObjectGS training
                        |
             Object-centric 3D Gaussians

The COLMAP wrapper selects the reconstruction with the most registered
images and records it in `selected_model.txt`. Downstream stages use
that same reconstruction rather than assuming `sparse/0`.

Object-only SfM retains 3D points based on the consistency of their
multi-view observations with the segmentation masks.

The prepared scene contains undistorted RGB images, aligned binary
masks, camera parameters, and object-only SfM initialization.

ObjectGS then trains using mask-guided photometric supervision,
an outside-mask penalty, and mask-normalized densification.

## Object-only SfM initialization

`scripts/prepare_object_sfm.py` uses the original COLMAP reconstruction
and the object masks to classify sparse Point3D tracks.

For a 3D point `p`, mask consistency is defined as:

    C(p) =
        number of observations inside the object masks
        ------------------------------------------------
        total number of observations of p

A point is retained when:

    C(p) >= consistency_threshold

The default threshold is:

    consistency_threshold = 1.0

so every observation of a retained sparse point must lie inside the
object silhouette.

The selected Point3D IDs are then applied to the undistorted COLMAP
reconstruction, preserving the camera geometry and valid Point2D /
Point3D associations.


## Scene preparation

`scripts/prepare_scene.py` constructs the final ObjectGS training scene.

The RGB images are the COLMAP-undistorted registered images.

Object masks are transformed through the same COLMAP
`image_undistorter` geometry and then converted back to binary masks
using:

    grayscale > 127

This ensures that RGB images, masks, cameras, and sparse initialization
all share the same undistorted coordinate system.


## Training scene format

A prepared scene has the following structure:

    scene/
      images/
        IMG_0001.jpeg
        ...

      masks/
        IMG_0001.png
        ...

      sparse/
        0/
          cameras.bin
          images.bin
          points3D.bin

Masks use the same image stem as their corresponding RGB image.


## Scripts

### Generate masks with SAM2

ObjectGS expects one binary foreground mask per input image. Masks can be
generated with [SAM2](https://github.com/facebookresearch/sam2) using the
provided wrapper.

Install the official SAM2 package:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

Download a SAM2 checkpoint. The Object 02 experiment uses
`sam2.1_hiera_small.pt` with the corresponding
`configs/sam2.1/sam2.1_hiera_s.yaml` model configuration.

Prompts are supplied as JSON and may be keyed either by image stem or by
the full image filename:

```json
{
  "IMG_5544.jpeg": {
    "box": [1415, 1429, 2067, 1812]
  }
}
```

Box coordinates follow `[x0, y0, x1, y1]`. Point prompts are also
supported by the script.

Generate the masks:

```bash
python scripts/generate_masks_sam2.py \
    --images /path/to/images \
    --output /path/to/masks \
    --prompts /path/to/prompts.json \
    --model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
    --checkpoint /path/to/sam2.1_hiera_small.pt \
    --device cuda
```

The generated masks are binary PNG files named by image stem. Validate
them before continuing with the reconstruction pipeline.

### Generate video masks with SAM2

ObjectGS supports generating object masks from an ordered sequence
of video frames using SAM2 Video Tracking.

Unlike per-image SAM2 prompting, this workflow requires only one
positive point in the first frame. SAM2 propagates the segmentation
through the remaining frames.

#### Example

    python scripts/generate_masks_sam2_video.py \
      --images /path/to/frames \
      --output /path/to/masks \
      --checkpoint checkpoints/sam2.1_hiera_small.pt \
      --point 1080 1350 \
      --device cuda \
      --precision bf16

The point coordinates are specified in the original first-frame
image resolution.

#### Requirements

- Install SAM2 using `setup.sh --install-sam2`.
- Input frames must be JPEG images with numeric frame ordering.
  Examples: `frame_0000.jpg`, `frame_0001.jpg`.
- Frames must represent a temporally ordered video sequence.
- Use a CUDA GPU for BF16 inference.

The script automatically creates temporary numeric filenames
required by SAM2 without modifying the source images.

#### Output

For each input image, the script creates a binary grayscale PNG:

- Object pixels: 255 (white)
- Background pixels: 0 (black)
- Resolution: identical to the source image
- Filename: `<image_stem>.png`

The generated masks are compatible with ObjectGS scene preparation
and object-only SfM initialization.

#### Validation

    python scripts/validate_masks.py \
      --images /path/to/frames \
      --masks /path/to/masks

Inspect the generated masks before training.

### Validate masks

    python scripts/validate_masks.py \
        --images /path/to/images \
        --masks /path/to/masks

Checks image/mask pairing, resolution, mask format, empty masks, and
foreground statistics.


### Run COLMAP

    python scripts/run_colmap.py \
        --images /path/to/images \
        --workspace /path/to/colmap_workspace

The wrapper performs feature extraction, matching, sparse mapping, and
scene undistortion.


### Prepare object-only SfM

    python scripts/prepare_object_sfm.py \
        --original-model /path/to/sparse/0 \
        --undistorted-model /path/to/undistorted/sparse \
        --masks /path/to/masks \
        --output /path/to/object_sfm

By default, only Point3D tracks with mask consistency `1.0` are kept.


### Prepare the ObjectGS scene

    python scripts/prepare_scene.py \
        --original-model /path/to/sparse/0 \
        --undistorted-images /path/to/undistorted/images \
        --object-model /path/to/object_sfm \
        --masks /path/to/masks \
        --output /path/to/objectgs_scene


### Train ObjectGS

    python scripts/train_objectgs.py \
        --runtime /path/to/patched/gaussian-splatting \
        --scene /path/to/objectgs_scene \
        --output /path/to/output \
        --iterations 7000

The default ObjectGS configuration uses:

    lambda_outside = 1.0
    mask_normalized_densification = enabled


## Reproducibility check

The scene-preparation pipeline was reconstructed and checked against
the original reference experiment.

For the reference object:

    original images       : 39
    registered views      : 36

    full SfM points       : 2,536
    object SfM points     : 1,736

    full observations     : 9,478
    object observations   : 7,006

The object-only SfM preparation reproduced exactly:

    Point3D IDs
    Point3D geometry
    colors
    tracks
    camera parameters
    image poses
    Point2D associations

The complete scene preparation reproduced:

    RGB images            : 36 / 36 byte-identical
    object masks          : 36 / 36 pixel-identical
    differing mask pixels : 0

This provides an end-to-end reproducible path from the original
COLMAP reconstruction and masks to the ObjectGS training scene.


### Clean-runtime reproduction (October 2026)

| Metric | Original | Reproduced |
|---|---:|---:|
| Registered images | 36 | 36 |
| Full SfM points | 2,536 | 2,504 |
| Object SfM points | 1,736 | 1,724 |
| Gaussians after 7K | 5,585 | 5,623 |

The Gaussian count differs by 0.68% from the reference run.
SAM2 box prompts were derived from ground-truth masks.
This is a box-prompt-guided reproduction, not automatic detection.
Similar counts do not prove identical geometry or visual quality.


## Upstream 3DGS

ObjectGS is developed against a pinned version of the original
Graphdeco Gaussian Splatting repository.

See:

    UPSTREAM.md
    objectgs.patch
    setup.sh

for the pinned upstream commit and patch-based setup.

The ObjectGS changes are intentionally kept separate from the full
upstream repository.


## Repository status

ObjectGS is a research prototype for object-centric 3D Gaussian Splatting.

**Implemented:**

- End-to-end pipelines for multi-view images and video input
- SAM2 image segmentation and video mask propagation
- COLMAP reconstruction and largest-component selection
- Multi-view mask-consistency filtering for object-only SfM
- Reproducible undistorted scene preparation
- Mask-guided photometric supervision and outside-mask penalty
- Mask-normalized Gaussian densification
- Patched upstream 3DGS training integration
- Stage validation, resume support, and pipeline logging
- Synchronized visual comparisons against Vanilla 3DGS

**Current limitations and future work:**

- Expand clean-runtime validation across additional objects and capture conditions
- Perform controlled Vanilla 3DGS versus ObjectGS comparisons
- Measure reconstruction quality using image and silhouette metrics
- Evaluate robustness across additional objects and capture conditions
- Extend automated testing and reproducibility checks

## Acknowledgements

ObjectGS builds upon the original 3D Gaussian Splatting implementation
by Kerbl et al.

SAM 2 can optionally be used for object-mask generation.

See the upstream projects and repository licensing information before
redistributing or using the software.

### SAM2 checkpoint

When `setup.sh` is run with `--install-sam2`, it automatically downloads the official SAM2.1 Hiera Small checkpoint to `checkpoints/sam2.1_hiera_small.pt`.

The checkpoint is reused on subsequent setup runs and is excluded from Git. Pass its path to the pipeline using `--sam-checkpoint`.
