# Observations

Autotune needs to run the first time the system runs a specific configuratiion.

GPU usage is highest at the very start. After the MSE reaches and equilibrium, it drops to almost no usage.

The model performs substantially worse during the evaluation if it was stopped immediately after the validation stage, compared to being stopped midway through the training stage.

Normalising the input data seems to have no significant impact on the evaluation metrics. Run-to-run variance is substantially more prominent.

## Batch Size

Batch size is the biggest killer of VRAM.

### RGB

- Batch size of 4:
  - Low VRAM usage; ~3.75GB epoch 1, ~4.46GB epoch 2
  - Generally low GPU usage, regularly increases to moderate usage, sometimes spikes to very high, maxed at 100% once
  - Consistent performance

- Batch size of 8:
  - Moderate VRAM usage; ~5.53GB epoch 1, ~9.98GB epoch 2
  - Low GPU usage; mostly ~1-2%, could spike up to ~20-30%
  - Stutters every few batches

### Luma

- Batch size of 4:
  - Low VRAM usage; ~2.13GB epoch 1, ~3.44GB epoch 2
  - Moderate GPU usage, regularly above ~20%, reduced near the end of the first epoch - continuing into the second
  - Consistent performance

- Batch size of 8:
  - Moderate VRAM usage; ~5.52GB epoch 1, ~10.03GB epoch 2
  - Low GPU usage; mostly ~2%, sometimes spiking up to ~30%
  - Very minor stuttering

## Current Accuracy

### RGB

Seems to bottom out at ~0.33 MSE after ~30% of epoch 1 for batch size of 4.
Seems to bottom out at ~1.00 MSE after ~40% of epoch 1 for batch size of 8.

### Luma

Seems to bottom out at ~1.2 MSE after ~10% of epoch 1 for batch size of 4.
Seems to bottom out at ~3.31 MSE after ~40% of epoch 1 for batch size of 8.
