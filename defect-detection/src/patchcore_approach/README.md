# PatchCore for Anomaly Detection (MVTec AD: bottle)

PatchCore does not train a network. It uses a frozen ImageNet-pretrained backbone to extract local patch features from defect-free images and stores them in a memory bank. At test time, each patch in an image is scored by its distance to the nearest stored normal patch. A patch that looks like nothing in the normal set gets a high score.

## Code structure

```
patchcore_approach/
    __init__.py
    dataset.py   # MVTec AD loader (same as the autoencoder approach)
    model.py     # PatchCore feature extractor, coreset, scoring
    main.py      # fit, score, evaluate
    checkpoints/
        patchcore_bottle.pt  # saved memory bank
```

### `dataset.py`

Identical to the autoencoder version. Images are returned in [0, 1] and normalization is applied inside the model, so the loader is shared between approaches.

### `model.py`: `PatchCore`

**Feature extraction** (`embed`). A frozen WideResNet50-2 pretrained on ImageNet is run up to `layer3`:

1. `layer2` produces 512-channel features at 1/8 resolution (32×32 for a 256 px input).
2. `layer3` produces 1024-channel features at 1/16 resolution (16×16).
3. Both are smoothed with a 3×3 average pool, so each patch describes its neighbourhood rather than a single pixel.
4. `layer3` is upsampled to the `layer2` grid and concatenated, giving 1536 channels per location.
5. The 1536-dim vector is reduced to 1024 dims with adaptive average pooling over the channel axis. This is a lightweight stand-in for the paper's random projection and keeps memory and distance computation manageable.

Each 256 px image gives 32×32 = 1024 patch vectors. For the 209 good training images, that is 214,016 patches.

**Memory bank** (`fit`, `_greedy_coreset`). Storing all 214k patches makes nearest-neighbour search slow. A greedy k-center coreset selects a subset that still covers the feature space:

1. Project the features to 128 dims with a fixed random matrix. Distances are computed in this space, which is faster.
2. Start from one random patch.
3. Repeatedly add the patch farthest from the current set.

With `--coreset_ratio 0.1`, the bank keeps 21,401 patches. The projection and seed are fixed, so the bank is reproducible.

**Scoring** (`anomaly_maps`). Each test patch is compared to every bank entry with `torch.cdist`, and its distance to the nearest one is kept. The 32×32 distance grid is bilinearly upsampled to 256×256 and smoothed with a 33×33 Gaussian (σ = 4). Patch-level smoothing is larger than in the autoencoder because the patch grid is coarse.

### `main.py`

**Fit.** Iterates over the good training images once to build the memory bank, then saves it to `--ckpt`. There is no training loop and no gradient computation.

**Evaluate.** Same protocol as the autoencoder approach so the numbers can be compared directly:

- Image score = max of the anomaly map.
- Image AUROC across all test images.
- Pixel AUROC against the ground-truth masks.
- Per-defect-type AUROC against the good images.
- Inference time per image, measured with `cuda.synchronize()` around the call so GPU timing is accurate.

## Usage

Run from `src/patchcore_approach/`:

```bash
python main.py                                  # build bank and evaluate
python main.py --eval_only                      # reuse a saved bank
python main.py --coreset_ratio 0.01             # faster, smaller bank
```

Other arguments: `--category`, `--img_size`, `--batch_size`, `--ckpt`, `--root`. The first run downloads the WideResNet50-2 weights (about 130 MB).

## Results (bottle, 256 px, coreset ratio 0.1)

| Metric               | Value         |
| -------------------- | ------------- |
| Image AUROC          | 1.0000        |
| Pixel AUROC          | 0.9870        |
| Inference            | 46.7 ms/image |
| broken_large (n=20)  | 1.0000        |
| broken_small (n=22)  | 1.0000        |
| contamination (n=21) | 1.0000        |

Patch features: 214,016 × 1024. Memory bank: 21,401 × 1024.

## Comparison with the autoencoder

| Metric        | Autoencoder  | PatchCore     |
| ------------- | ------------ | ------------- |
| Image AUROC   | 0.9405       | 1.0000        |
| Pixel AUROC   | 0.8561       | 0.9870        |
| contamination | 0.8214       | 1.0000        |
| Inference     | not measured | 46.7 ms/image |

The autoencoder's weakness was contamination. It learns to reconstruct pixels, so a subtle foreign material can be partly reproduced and its error overlaps with normal images. PatchCore compares features against stored normal patches, so a contamination patch has no close neighbour in the bank and receives a large distance. Both methods separate the broken types, but only PatchCore separates contamination.

## Limitations

- One category, one run. Image AUROC of exactly 1.0 is strong, but it is on `bottle` only. Harder categories such as `screw` or `capsule` are needed before claiming the method generalizes.
- The 1536 → 1024 dimension reduction is a simplification of the paper's approach. The paper's effect on accuracy has not been measured here.
- The paper's score re-weighting step is not implemented. The score is a plain maximum over the smoothed map.
- The memory bank is fixed after fit. Adding new normal products or lighting conditions requires rebuilding it.
- Memory grows with the number of training images and patches. The bank is about 88 MB in float32 at this setting.

## Next steps

1. Run the coreset sweep (0.01, 0.1, 0.25) and record AUROC and latency for each.
2. Inspect the lowest-scoring pixel maps to see whether pixel AUROC loss comes from boundaries or small defects.
3. Run both approaches on a second category (`metal_nut`) to test whether the ranking holds.
4. Choose an operating threshold on the test precision/recall curve for the agent layer.
