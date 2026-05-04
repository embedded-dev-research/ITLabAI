# Model Performance Benchmark

`model_performance.py` benchmarks the existing `Graph_Build` executable for all
target networks:

- `alexnet_mnist`
- `googlenet`
- `densenet`
- `resnet`
- `yolo`

It measures wall time and RSS memory timeline for two stages:

- `compile`: process start until `Graph_Build` prints `Starting inference...`
- `inference`: `Starting inference...` until `Inference completed successfully.`

The benchmark does not modify C++ code. It reads the executable output live,
samples process memory while the command is running, stores the full RSS sample
series, and writes a memory plot for every measured run.

Install `matplotlib` to generate memory plots. Install `psutil` to measure RSS
for the full process tree on every platform. Without `psutil`, Linux uses
`/proc`, while macOS and Windows use parent-process RSS fallbacks.

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
  --warmup 1
```

The JSON report includes `memory_samples` for every run. PNG plots are written
to `benchmark_results/memory_plots` by default. Use `--samples-csv-out` to export
the memory timeline to CSV and `--plots-dir` to choose another plot directory.

Use `--strict-assets` to fail when a model JSON or input image directory is
missing instead of skipping that model.
