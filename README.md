# ObjectGS

### Mask-Guided Object-Centric Gaussian Splatting

ObjectGS is an experimental object-centric 3D Gaussian Splatting
pipeline designed to reconstruct compact 3D representations of
individual objects from multi-view RGB images and segmentation masks.

## Motivation

Standard 3D Gaussian Splatting is designed for full-scene
reconstruction.

When the reconstruction objective is restricted to a small foreground
object and normalized by its mask area, the scale of the gradients used
by the standard densification heuristic changes substantially.

In our experiments this caused severe over-densification.

ObjectGS decouples the scale used for object reconstruction from the
scale used for densification.

## Core idea

For each training view:

    object_loss = masked_photometric_error / foreground_area

while the screen-space gradient accumulated for densification is
corrected using:

    densification_gradient *= mask_fraction

where:

    mask_fraction = foreground_pixels / image_pixels

The reconstruction objective therefore remains focused on the object,
while the densification heuristic operates at a more appropriate scale.

## Preliminary result

36-view object reconstruction.

| Method | Iterations | Gaussians | Masked PSNR | Masked L1 |
|---|---:|---:|---:|---:|
| Mask-loss baseline | 3K | 182,777 | 24.75 dB | 0.0302 |
| ObjectGS | 3K | 4,163 | 23.03 dB | 0.0390 |
| Mask-loss baseline | 7K | 409,451 | 28.23 dB | 0.0192 |
| ObjectGS | 7K | 6,295 | 24.85 dB | 0.0306 |

At 7K iterations, ObjectGS uses approximately **65x fewer Gaussians**.

ObjectGS at 7K also achieves approximately the same masked PSNR as the
3K baseline while using approximately **29x fewer Gaussians**.

These are preliminary results on a single object. Evaluation on
additional objects is in progress.

## Pipeline

    Multi-view RGB images
              |
              +---- segmentation ----> object masks
              |
              +---- COLMAP ----------> camera poses + sparse SfM
                                         |
                                         v
                              object-only SfM initialization
                                         |
                                         v
                                   ObjectGS training
                                         |
                                         v
                            compact object Gaussian model

Masks may come from SAM 2, another segmentation model, or manually
provided masks.

## Input

A prepared scene follows:

    scene/
      images/
      masks/
      sparse/

Masks use the same image stem as their corresponding RGB image.

Example:

    images/IMG_0001.jpeg
    masks/IMG_0001.png

## Project status

Research prototype.

Current priorities:

1. Validate the method on additional objects.
2. Measure the quality/complexity trade-off.
3. Package mask generation and object-SfM preparation.
4. Provide a reproducible end-to-end training pipeline.

## Acknowledgements

ObjectGS builds upon the original 3D Gaussian Splatting implementation
by Kerbl et al.

SAM 2 can optionally be used to generate object masks.

See the upstream projects and repository licensing information before
redistributing or using the software.
