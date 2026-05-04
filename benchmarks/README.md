# Model Performance Benchmark

`model_performance.py` benchmarks the existing `Graph_Build` executable for all
target networks:

- `alexnet_mnist`
- `googlenet`
- `densenet`
- `resnet`
- `yolo`

It measures wall time and peak RSS memory for two stages:

- `compile`: process start until `Graph_Build` prints `Starting inference...`
- `inference`: `Starting inference...` until `Inference completed successfully.`

The benchmark does not modify C++ code. It reads the executable output live and
samples process memory while the command is running.

Install `psutil` to measure RSS for the full process tree on every platform.
Without it, Linux uses `/proc`, while macOS and Windows use parent-process RSS
fallbacks.

## Usage

Build the project first:

```bash
cmake -S . -B build
cmake --build build --target Graph_Build --parallel
```

Run the default benchmark over every model with available JSON/input assets:

```bash
python3 benchmarks/model_performance.py
```

Run selected models and variants:

```bash
python3 benchmarks/model_performance.py \
  --model googlenet,resnet \
  --variant seq \
  --variant parallel-tbb \
  --repeat 3 \
  --warmup 1 \
  --csv-out benchmark_results/model_performance.csv
```

Use `--strict-assets` to fail when a model JSON or input image directory is
missing instead of skipping that model.
