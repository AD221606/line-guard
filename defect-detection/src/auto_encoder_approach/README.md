# Convolutional Autoencoder for Anomaly Detection (MVTec AD: bottle)

This approach trains a convolutional autoencoder on defect-free images only. At test time, a defective image should reconstruct poorly, because the model never learned to reproduce defects. The reconstruction error becomes the anomaly score.

## Code structure

```
auto_encoder_approach/
    __init__.py
    dataset.py   # MVTec AD loader
    model.py     # ConvAE architecture
    main.py      # training, scoring, evaluation
    checkpoints/
        ae_bottle.pt # saved weights
```

### `dataset.py`: `MVTecDataset`

Loads one MVTec category. The `split` argument controls what is returned:

- `"train"`: only `train/good/*.png`, all with label 0. The model never sees defects during training.
- `"test"`: every folder under `test/`. `good` images have label 0 and an all-zero mask. Defect images have label 1 and their pixel ground-truth mask from `ground_truth/<defect_type>/<name>_mask.png`.

Images are resized to `img_size` (256) and converted to tensors in [0, 1]. Masks use nearest-neighbour resizing so they stay binary. Each sample returns `(image, mask, label, defect_type)`.

### `model.py`: `ConvAE`

The network maps a 256×256×3 image through a bottleneck and back.

| Stage                                          | Output shape             |
| ---------------------------------------------- | ------------------------ |
| Input                                          | 3 × 256 × 256            |
| 5 × stride-2 conv (`down`)                     | 256 × 8 × 8              |
| 1×1 conv (bottleneck)                          | 16 × 8 × 8 (1024 values) |
| 1×1 conv + 4 × stride-2 transposed conv (`up`) | 32 × 128 × 128           |
| Final transposed conv + Sigmoid                | 3 × 256 × 256            |

`down` is a 4×4 stride-2 convolution with BatchNorm and LeakyReLU. `up` is a 4×4 stride-2 transposed convolution with BatchNorm and ReLU. The Sigmoid keeps outputs in [0, 1] to match the inputs.

The bottleneck size is the main capacity control. A wider bottleneck reconstructs defects too well. A narrower one blurs normal texture too.

### `main.py`

**Training** (`train`): Adam with learning rate 1e-3, MSE loss between reconstruction and input, default 100 epochs, batch size 16. Weights are saved to `--ckpt`. Training MSE is printed every 5 epochs.

**Scoring** (`anomaly_maps`): For each image, compute the per-pixel squared error averaged over RGB channels, then apply a Gaussian blur (kernel 11, σ = 4). The blur suppresses single-pixel edge noise, which would otherwise dominate the maximum.

**Evaluation** (`evaluate`): The image score is the maximum value of the blurred error map. Maximum is used rather than mean because defects are small and local. Averaging over 65k pixels would dilute them. The script reports:

- Image AUROC across all test images
- Pixel AUROC against the ground-truth masks
- Image AUROC per defect type, each compared against the good images

## Usage

Run from `src/auto_encoder_approach/`:

```bash
python main.py                              # train and evaluate
python main.py --eval_only                  # evaluate a saved checkpoint
python main.py --latent_channels 4          # sweep the bottleneck
```

Useful arguments: `--category`, `--img_size`, `--latent_channels`, `--epochs`, `--batch_size`, `--lr`, `--ckpt`, `--root`.

## Results (bottle, 256 px, 16 latent channels, 100 epochs)

| Metric               | Value  |
| -------------------- | ------ |
| Image AUROC          | 0.9405 |
| Pixel AUROC          | 0.8561 |
| broken_large (n=20)  | 1.0000 |
| broken_small (n=22)  | 1.0000 |
| contamination (n=21) | 0.8214 |

The test set has 20 good images and 63 defective images. Per-type AUROC compares each defect type against the 20 good images.

## Interpreting the results

**Broken defects are solved; contamination is the weak point.** Broken bottles have large, high-contrast regions where the reconstruction fails badly. Contamination is a foreign material that is often subtle, low-contrast, or close in appearance to normal glass texture. The autoencoder can reconstruct it partly, so its error is smaller and overlaps with normal images. Because contamination accounts for a third of the defect set, it pulls the overall image AUROC down to 0.94 even though the two broken types are perfectly separated.

**Pixel AUROC is lower than image AUROC.** The image score takes the maximum over the whole image, so one strong pixel is enough. Pixel AUROC requires the whole error map to rank correctly against the mask. Normal reconstruction error at edges, the bottle rim, and lighting gradients all contribute non-zero error across the image, and these compete with the low-magnitude errors inside small defects. The blur also spreads error outward, which reduces localization precision.

**Results are reasonable for a pixel-space baseline but not final.** The gap to feature-based methods is the point of comparison. PatchCore compares ImageNet features against stored normal patches, so subtle contamination has a distinct feature signature that the autoencoder's pixel-level reconstruction does not capture.

## Limitations

- Single category, single seed. Results on one run can shift by a few points, so repeat with several seeds before comparing methods.
- The threshold for a hard `is_defective` decision is not set. AUROC measures ranking, not a usable operating point. Choose the threshold from the test-set precision/recall curve, or hold out good images, and do not use training-set scores.
- The bottleneck was not swept. Try `--latent_channels` at 4, 16, and 64 and report AUROC for each.
- Inference cost is low, but the approach has a known ceiling on subtle texture defects.

## Next steps

1. Sweep latent channels and report the table.
2. Choose and document an operating threshold.
3. Compare against PatchCore on the same split, especially on contamination.
4. Record the failure cases (images of contamination with the lowest scores) for the write-up.
