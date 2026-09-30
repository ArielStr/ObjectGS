# ObjectGS

### Mask-Guided Object-Centric Gaussian Splatting

ObjectGS is an experimental object-centric 3D Gaussian Splatting
pipeline for reconstructing compact 3D representations of individual
objects from multi-view RGB images and segmentation masks.

The project builds on the original Graphdeco 3D Gaussian Splatting
implementation and modifies both the training objective and
densification behavior for small foreground objects.


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


## Preliminary result

Reference experiment: 36 registered views of a single object.

| Method | Iterations | Gaussians | Masked PSNR | Masked L1 |
|---|---:|---:|---:|---:|
| Mask-loss baseline | 3K | 182,777 | 24.75 dB | 0.0302 |
| ObjectGS | 3K | 4,163 | 23.03 dB | 0.0390 |
| Mask-loss baseline | 7K | 409,451 | 28.23 dB | 0.0192 |
| ObjectGS | 7K | 6,295 | 24.85 dB | 0.0306 |

At 7K iterations, ObjectGS uses approximately **65x fewer Gaussians**
than the mask-loss baseline.

ObjectGS at 7K also reaches approximately the same masked PSNR as the
3K baseline while using approximately **29x fewer Gaussians**.

These results are preliminary and currently come from a single object.
Evaluation on additional objects is planned.


## Pipeline

The repository provides wrappers for the main reconstruction stages:

    Multi-view RGB images
              |
              +---- segmentation ----> object masks
              |
              v
            COLMAP
              |
              v
       full sparse SfM
              |
              v
    multi-view mask consistency
              |
              v
      object-only SfM
              |
              v
       scene preparation
        /            \
       /              \
undistorted RGB   undistorted masks
                         |
                    threshold > 127
       \              /
        \            /
              |
              v
       ObjectGS scene
              |
              v
       ObjectGS training
              |
              v
    compact object Gaussian model

Masks can be produced manually or by an external segmentation model.
SAM 2 support is being integrated as an optional mask-generation step.


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

Research prototype.

Implemented:

- mask validation
- COLMAP pipeline wrapper
- multi-view object-SfM filtering
- reproducible undistorted scene preparation
- object-only photometric training loss
- outside-mask RGB penalty
- mask-normalized densification
- ObjectGS training wrapper
- pinned upstream 3DGS integration

Current work:

- validate SAM 2 mask generation in a clean GPU runtime
- evaluate additional objects
- perform controlled ablations
- measure reconstruction quality vs. Gaussian count
- add qualitative renders and comparison figures


## Acknowledgements

ObjectGS builds upon the original 3D Gaussian Splatting implementation
by Kerbl et al.

SAM 2 can optionally be used for object-mask generation.

See the upstream projects and repository licensing information before
redistributing or using the software.
