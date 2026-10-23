# Vision - CNN

A convolutional neural network training system.

To enable the pre-commit hook, run `git config core.hooksPath .githooks`.

## Usage Examples

Train a new model into `./output/model_1/`:

```sh
vision-cnn train new model_1
```

Resume in place, or resume into a new output folder:

```sh
vision-cnn train resume model_1
vision-cnn train resume model_1 -o model_2
```

In-place resume does not delete checkpoints and backs up an existing model of the same name with a UTC timestamp.

Choose the model export filename, evaluate after training, or train without
exporting the model:

```sh
vision-cnn train new model_1 --name model_1-half --evaluate
vision-cnn train new --dry-run
vision-cnn train new --dry-run --evaluate
```

Run inference on a model stored at `./output/model_1`:

```sh
vision-cnn infer model_1
```

Choose the model's filename, choose the number of items to use for evaluation, or evaluate all items:

```sh
vision-cnn infer model_1 --name model_1-half
vision-cnn infer model_1 -n 5
vision-cnn infer model_1 --evaluate-all
```

Full evaluation saves `metrics.json` and `samples.csv` into `./reports/MODEL_DIR/` by default.

Training configuration options can be modified by arguments. These options are valid for both training a new model or resuming an existing one. The options are `--num_epochs`, `--batch-size`, `--num-workers`, `--seed`, `--learning-rate`, `--decay-rate`, `--decay-steps`, and `--early-stopping-patience`.

When resuming, `num_epochs` is the number of additional epochs to run. E.g. A model that has already been trained for 3 epochs and resumed with `--num-epochs 2`, will train up to epoch 5. If no value is provided for `--num-epochs`, the default value is used for the number of additional epochs.

## Environment Variables

A number of options are also available as environment variables. The command-line arguments will take precedence over the environment variables.

The environment variables that are supported:

| Variable Name | Equivalent Argument | Description | Default |
| --- | --- | --- | --- |
| `VISION_CNN_OUTPUT_DIR` | `[OUTPUT_NAME]` | Base directory for trained models | `./output` |
| `VISION_CNN_REPORT_DIR` | N/A | Base directory for evaluation reports | `./reports` |
| `VISION_CNN_MODEL_NAME` | `--name` | Exported model filename (no extension) | `model` |
| `VISION_CNN_NUM_EPOCHS` | `--num-epochs` | Epochs to run in this invocation | `2` |
| `VISION_CNN_BATCH_SIZE` | `--batch-size` | Samples per batch | `8` |
| `VISION_CNN_NUM_WORKERS` | `--num-workers` | Data-loading workers | `4` |
| `VISION_CNN_SEED` | `--seed` | Random seed | `18` |
| `VISION_CNN_LEARNING_RATE` | `--learning-rate` | Initial learning rate | `0.01` |
| `VISION_CNN_DECAY_RATE` | `--decay-rate` | Inverse-time decay factor | `0.2` |
| `VISION_CNN_DECAY_STEPS` | `--decay-steps` | Learning-rate decay interval (`0` derives it automatically) | `0` |
| `VISION_CNN_EARLY_STOPPING_PATIENCE` | `--early-stopping-patience` | Validation epochs without improvement | `2` |
