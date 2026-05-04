#!/usr/bin/env python3
"""Benchmark ITLabAI target models through the Graph_Build executable."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import errno
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pty
except ImportError:  # pragma: no cover - Windows
    pty = None


ROOT = Path(__file__).resolve().parents[1]
START_MARKER = "Starting inference..."
DONE_MARKER = "Inference completed successfully."

TARGET_MODELS = {
    "alexnet_mnist": ("docs/jsons/model_data_alexnet_1.json", "docs/input/28"),
    "googlenet": ("docs/jsons/googlenet_onnx_model.json", "docs/input/Imagenet_test"),
    "densenet": (
        "docs/jsons/densenet121_Opset16_onnx_model.json",
        "docs/input/Imagenet_test",
    ),
    "resnet": (
        "docs/jsons/resnest101e_Opset16_onnx_model.json",
        "docs/input/Imagenet_test",
    ),
    "yolo": ("docs/jsons/yolo11x-cls_onnx_model.json", "docs/input/Imagenet_test"),
}

VARIANT_ARGS = {
    "seq": [],
    "onednn": ["--onednn"],
    "parallel-tbb": ["--parallel", "tbb"],
    "parallel-threads": ["--parallel", "threads"],
    "parallel-omp": ["--parallel", "omp"],
    "parallel-kokkos": ["--parallel", "kokkos"],
    "seq-fusion-off": ["--fusion", "off"],
    "seq-fusion-on": ["--fusion", "convrelu"],
    "parallel-tbb-fusion-off": ["--parallel", "tbb", "--fusion", "off"],
    "parallel-tbb-fusion-on": ["--parallel", "tbb", "--fusion", "convrelu"],
    "parallel-threads-fusion-off": ["--parallel", "threads", "--fusion", "off"],
    "parallel-threads-fusion-on": [
        "--parallel",
        "threads",
        "--fusion",
        "convrelu",
    ],
    "parallel-omp-fusion-off": ["--parallel", "omp", "--fusion", "off"],
    "parallel-omp-fusion-on": ["--parallel", "omp", "--fusion", "convrelu"],
    "parallel-kokkos-fusion-off": ["--parallel", "kokkos", "--fusion", "off"],
    "parallel-kokkos-fusion-on": [
        "--parallel",
        "kokkos",
        "--fusion",
        "convrelu",
    ],
    "onednn-fusion-off": ["--onednn", "--fusion", "off"],
    "onednn-fusion-on": ["--onednn", "--fusion", "postops"],
}

VARIANT_GROUPS = {
    "all": list(VARIANT_ARGS),
    "target": [
        "seq-fusion-off",
        "seq-fusion-on",
        "parallel-tbb-fusion-off",
        "parallel-tbb-fusion-on",
        "parallel-threads-fusion-off",
        "parallel-threads-fusion-on",
        "parallel-omp-fusion-off",
        "parallel-omp-fusion-on",
        "parallel-kokkos-fusion-off",
        "parallel-kokkos-fusion-on",
        "onednn-fusion-off",
        "onednn-fusion-on",
    ],
}


@dataclasses.dataclass
class Phase:
    duration_s: float = 0.0
    peak_rss_bytes: int = 0
    markers: int = 0


@dataclasses.dataclass
class Sample:
    timestamp_s: float
    stage: str
    rss_bytes: int


@dataclasses.dataclass
class Result:
    model: str
    variant: str
    repeat: int
    command: list[str]
    returncode: int
    total_s: float
    peak_rss_bytes: int
    compile: Phase
    inference: Phase
    marker_split: bool
    samples: list[Sample]
    stdout_tail: list[str]
    plot_path: str = ""
    error: str = ""

    def row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "variant": self.variant,
            "repeat": self.repeat,
            "returncode": self.returncode,
            "total_s": round(self.total_s, 6),
            "compile_s": round(self.compile.duration_s, 6),
            "inference_s": round(self.inference.duration_s, 6),
            "peak_rss_mib": bytes_to_mib(self.peak_rss_bytes),
            "compile_peak_rss_mib": bytes_to_mib(self.compile.peak_rss_bytes),
            "inference_peak_rss_mib": bytes_to_mib(self.inference.peak_rss_bytes),
            "compile_markers": self.compile.markers,
            "inference_markers": self.inference.markers,
            "marker_split": self.marker_split,
            "command": " ".join(self.command),
            "error": self.error,
        }


def bytes_to_mib(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def exe_name() -> str:
    return "Graph_Build.exe" if os.name == "nt" else "Graph_Build"


def default_graph_build(build_dir: Path) -> Path:
    for candidate in (
        build_dir / "bin" / exe_name(),
        build_dir / "bin" / "Release" / exe_name(),
        build_dir / "bin" / "Debug" / exe_name(),
    ):
        if candidate.exists():
            return candidate
    return build_dir / "bin" / exe_name()


def expand_choices(values: Sequence[str], choices: dict[str, object], default: str) -> list[str]:
    if not values:
        return list(choices) if default == "all" else [default]

    expanded: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if item == "all":
                expanded.extend(choices)
            elif item in VARIANT_GROUPS and choices is VARIANT_ARGS:
                expanded.extend(VARIANT_GROUPS[item])
            elif item in choices:
                expanded.append(item)
            else:
                valid = ["all", *choices]
                if choices is VARIANT_ARGS:
                    valid.extend(name for name in VARIANT_GROUPS if name != "all")
                raise SystemExit(f"Unknown value '{item}'. Valid: {', '.join(valid)}")
    return dedupe(expanded)


def dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def has_image(input_dir: Path) -> bool:
    if not input_dir.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        for path in input_dir.iterdir()
    )


def check_assets(model: str, strict: bool) -> bool:
    json_rel, input_rel = TARGET_MODELS[model]
    missing = [
        str(path)
        for path, ok in (
            (ROOT / json_rel, (ROOT / json_rel).is_file()),
            (ROOT / input_rel, has_image(ROOT / input_rel)),
        )
        if not ok
    ]
    if not missing:
        return True
    message = f"missing benchmark assets for {model}: {', '.join(missing)}"
    if strict:
        raise SystemExit(message)
    print(f"Skipping {model}: {message}", file=sys.stderr)
    return False


def rss_tree(pid: int) -> int:
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return sum(
                child.memory_info().rss
                for child in [proc, *proc.children(recursive=True)]
                if child.is_running()
            )
        except psutil.Error:
            return 0
    if sys.platform.startswith("linux"):
        return linux_rss_tree(pid)
    if os.name == "nt":
        return powershell_rss(pid)
    return ps_rss(pid)


def linux_rss_tree(pid: int) -> int:
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        current = int(status.parent.name)
        parent = 0
        current_rss = 0
        try:
            for line in status.read_text(errors="ignore").splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    current_rss = int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
        children.setdefault(parent, []).append(current)
        rss[current] = current_rss

    total = 0
    stack = [pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += rss.get(current, 0)
        stack.extend(children.get(current, []))
    return total


def ps_rss(pid: int) -> int:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        return int(output.strip() or "0") * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def powershell_rss(pid: int) -> int:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64",
    ]
    try:
        return int(subprocess.check_output(command, text=True).strip() or "0")
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def sample_memory(
    pid: int,
    started_at: float,
    stage: dict[str, str],
    samples: list[Sample],
    stop: threading.Event,
    interval_s: float,
) -> None:
    while not stop.is_set():
        samples.append(Sample(time.perf_counter() - started_at, stage["name"], rss_tree(pid)))
        stop.wait(interval_s)
    samples.append(Sample(time.perf_counter() - started_at, stage["name"], rss_tree(pid)))


def record_line(
    line: str,
    lines: list[str],
    stage: dict[str, str],
    phase_started_at: dict[str, float],
    compile_phase: Phase,
    inference_phase: Phase,
) -> None:
    now = time.perf_counter()
    lines.append(line.rstrip())
    if START_MARKER in line:
        compile_phase.duration_s += now - phase_started_at["time"]
        compile_phase.markers += 1
        stage["name"] = "inference"
        phase_started_at["time"] = now
    elif DONE_MARKER in line and stage["name"] == "inference":
        inference_phase.duration_s += now - phase_started_at["time"]
        inference_phase.markers += 1
        stage["name"] = "compile"
        phase_started_at["time"] = now


def read_pipe(
    proc: subprocess.Popen[str],
    lines: list[str],
    stage: dict[str, str],
    phase_started_at: dict[str, float],
    compile_phase: Phase,
    inference_phase: Phase,
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        record_line(line, lines, stage, phase_started_at, compile_phase, inference_phase)
    proc.wait()


def read_pty(
    proc: subprocess.Popen[bytes],
    master_fd: int,
    lines: list[str],
    stage: dict[str, str],
    phase_started_at: dict[str, float],
    compile_phase: Phase,
    inference_phase: Phase,
) -> None:
    buffer = ""
    while True:
        try:
            chunk = os.read(master_fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        buffer += chunk.decode(errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            record_line(line, lines, stage, phase_started_at, compile_phase, inference_phase)
    if buffer:
        record_line(buffer, lines, stage, phase_started_at, compile_phase, inference_phase)
    proc.wait()


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


def peak_by_stage(samples: list[Sample], compile_phase: Phase, inference_phase: Phase) -> int:
    process_peak = 0
    for sample in samples:
        process_peak = max(process_peak, sample.rss_bytes)
        if sample.stage == "compile":
            compile_phase.peak_rss_bytes = max(compile_phase.peak_rss_bytes, sample.rss_bytes)
        elif sample.stage == "inference":
            inference_phase.peak_rss_bytes = max(
                inference_phase.peak_rss_bytes,
                sample.rss_bytes,
            )
    return process_peak


def run_command(command: list[str], timeout_s: float, interval_s: float) -> Result:
    start = time.perf_counter()
    phase_started_at = {"time": start}
    stage = {"name": "compile"}
    compile_phase = Phase()
    inference_phase = Phase()
    samples: list[Sample] = []
    lines: list[str] = []
    stop = threading.Event()
    error = ""
    master_fd: int | None = None

    use_pty = os.name != "nt" and pty is not None
    if use_pty:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        reader = threading.Thread(
            target=read_pty,
            args=(proc, master_fd, lines, stage, phase_started_at, compile_phase, inference_phase),
            daemon=True,
        )
    else:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        reader = threading.Thread(
            target=read_pipe,
            args=(proc, lines, stage, phase_started_at, compile_phase, inference_phase),
            daemon=True,
        )

    monitor = threading.Thread(
        target=sample_memory,
        args=(proc.pid, start, stage, samples, stop, interval_s),
        daemon=True,
    )
    monitor.start()
    reader.start()
    reader.join(timeout_s)
    if reader.is_alive():
        error = f"timeout after {timeout_s:.1f}s"
        terminate(proc)
        reader.join(5)
    stop.set()
    monitor.join(2)
    if master_fd is not None:
        try:
            os.close(master_fd)
        except OSError:
            pass

    end = time.perf_counter()
    if proc.returncode is None:
        proc.wait(timeout=5)
    if stage["name"] == "inference":
        inference_phase.duration_s += end - phase_started_at["time"]
    elif compile_phase.markers == 0:
        compile_phase.duration_s = end - start

    peak = peak_by_stage(samples, compile_phase, inference_phase)
    return Result(
        model="",
        variant="",
        repeat=0,
        command=command,
        returncode=int(proc.returncode or 0),
        total_s=end - start,
        peak_rss_bytes=peak,
        compile=compile_phase,
        inference=inference_phase,
        marker_split=compile_phase.markers > 0 and inference_phase.markers > 0,
        samples=samples,
        stdout_tail=lines[-80:],
        error=error or (f"process exited with {proc.returncode}" if proc.returncode else ""),
    )


def run_model(
    graph_build: Path,
    model: str,
    variant: str,
    repeat: int,
    timeout_s: float,
    interval_s: float,
    extra_args: Sequence[str],
) -> Result:
    command = [str(graph_build), "--model", model, *VARIANT_ARGS[variant], *extra_args]
    result = run_command(command, timeout_s, interval_s)
    result.model = model
    result.variant = variant
    result.repeat = repeat
    return result


def report_path(suffix: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "benchmark_results" / f"model_performance-{timestamp}.{suffix}"


def sample_rows(row: Result) -> list[dict[str, object]]:
    return [
        {
            "timestamp_s": round(sample.timestamp_s, 6),
            "stage": sample.stage,
            "rss_mib": bytes_to_mib(sample.rss_bytes),
        }
        for sample in row.samples
    ]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def plot_memory(row: Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / (
        f"{safe_name(row.model)}-{safe_name(row.variant)}-r{row.repeat}-memory.png"
    )
    times = [sample.timestamp_s for sample in row.samples]
    rss = [bytes_to_mib(sample.rss_bytes) for sample in row.samples]
    compile_times = [sample.timestamp_s for sample in row.samples if sample.stage == "compile"]
    compile_rss = [bytes_to_mib(sample.rss_bytes) for sample in row.samples if sample.stage == "compile"]
    inference_times = [sample.timestamp_s for sample in row.samples if sample.stage == "inference"]
    inference_rss = [bytes_to_mib(sample.rss_bytes) for sample in row.samples if sample.stage == "inference"]

    _, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, rss, color="#1f2937", linewidth=1.2, label="rss")
    if compile_times:
        axis.scatter(compile_times, compile_rss, color="#2563eb", s=8, label="compile")
    if inference_times:
        axis.scatter(inference_times, inference_rss, color="#dc2626", s=8, label="inference")
    axis.set_title(f"{row.model} / {row.variant} / repeat {row.repeat}")
    axis.set_xlabel("time, s")
    axis.set_ylabel("RSS, MiB")
    axis.grid(True, alpha=0.25)
    axis.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()
    row.plot_path = str(plot_path)


def write_samples_csv(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "variant", "repeat", "timestamp_s", "stage", "rss_mib"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for sample in sample_rows(row):
                writer.writerow(
                    {
                        "model": row.model,
                        "variant": row.variant,
                        "repeat": row.repeat,
                        **sample,
                    }
                )


def write_json(path: Path, args: argparse.Namespace, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "graph_build": str(args.graph_build),
            "sample_interval_s": args.sample_interval,
        },
        "results": [
            {
                **row.row(),
                "memory_plot": row.plot_path,
                "memory_samples": sample_rows(row),
                "stdout_tail": row.stdout_tail,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "variant",
        "repeat",
        "returncode",
        "total_s",
        "compile_s",
        "inference_s",
        "peak_rss_mib",
        "compile_peak_rss_mib",
        "inference_peak_rss_mib",
        "compile_markers",
        "inference_markers",
        "marker_split",
        "command",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.row())


def print_summary(rows: list[Result]) -> None:
    if not rows:
        print("No benchmark runs were executed.")
        return
    print("model | variant | rep | total_s | compile_s | infer_s | peak_mib | rc")
    print("----- | ------- | --- | ------- | --------- | ------- | -------- | --")
    for row in rows:
        print(
            f"{row.model} | {row.variant} | {row.repeat} | "
            f"{row.total_s:.3f} | {row.compile.duration_s:.3f} | "
            f"{row.inference.duration_s:.3f} | {bytes_to_mib(row.peak_rss_bytes):.1f} | "
            f"{row.returncode}"
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Graph_Build compile/inference time and RSS."
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--graph-build", type=Path)
    parser.add_argument("--model", action="append", default=[], help="Repeat or comma-list. Default: all.")
    parser.add_argument("--variant", action="append", default=[], help="Repeat or comma-list. Default: seq.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--strict-assets", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--samples-csv-out", type=Path)
    parser.add_argument("--plots-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    args.build_dir = args.build_dir.resolve()
    args.graph_build = (
        args.graph_build.resolve()
        if args.graph_build
        else default_graph_build(args.build_dir).resolve()
    )
    if not args.graph_build.exists():
        raise SystemExit(f"Graph_Build was not found: {args.graph_build}")

    models = [
        model
        for model in expand_choices(args.model, TARGET_MODELS, "all")
        if check_assets(model, args.strict_assets)
    ]
    variants = expand_choices(args.variant, VARIANT_ARGS, "seq")

    results: list[Result] = []
    for model in models:
        for variant in variants:
            for index in range(args.warmup):
                print(f"warmup model={model} variant={variant} {index + 1}/{args.warmup}", flush=True)
                run_model(
                    args.graph_build,
                    model,
                    variant,
                    -(index + 1),
                    args.timeout,
                    args.sample_interval,
                    args.extra_arg,
                )
            for repeat in range(args.repeat):
                print(f"benchmark model={model} variant={variant} {repeat + 1}/{args.repeat}", flush=True)
                results.append(
                    run_model(
                        args.graph_build,
                        model,
                        variant,
                        repeat,
                        args.timeout,
                        args.sample_interval,
                        args.extra_arg,
                    )
                )

    json_out = args.json_out or report_path("json")
    plots_dir = args.plots_dir or json_out.parent / "memory_plots"
    for row in results:
        plot_memory(row, plots_dir)

    print_summary(results)
    write_json(json_out, args, results)
    print(f"JSON report: {json_out}")
    if args.csv_out:
        write_csv(args.csv_out, results)
        print(f"CSV report: {args.csv_out}")
    if args.samples_csv_out:
        write_samples_csv(args.samples_csv_out, results)
        print(f"Samples CSV report: {args.samples_csv_out}")
    print(f"Memory plots: {plots_dir}")
    return 1 if any(row.returncode != 0 for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
