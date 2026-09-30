# Preliminary Results

## Object 02 — Iron Man

Training views: 36

Initial object SfM points: 1,736

### Gaussian count

| Method | 3K | 7K |
|---|---:|---:|
| Mask-loss baseline | 182,777 | 409,451 |
| ObjectGS | 4,163 | 6,295 |

### Object-region reconstruction

| Method | Iterations | Masked PSNR | Masked L1 |
|---|---:|---:|---:|
| Baseline | 3K | 24.7491 | 0.0302 |
| ObjectGS | 3K | 23.0323 | 0.0390 |
| Baseline | 7K | 28.2282 | 0.0192 |
| ObjectGS | 7K | 24.8513 | 0.0306 |

### Main observations

At 7K iterations the number of Gaussians decreases from 409,451 to
6,295, a reduction of approximately 65x.

ObjectGS 7K and baseline 3K obtain similar masked PSNR while ObjectGS
uses approximately 29x fewer Gaussians.

The current results indicate a substantial quality/representation-size
trade-off. Additional objects are required before drawing conclusions
about generalization.
