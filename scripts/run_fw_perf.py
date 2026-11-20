#!/usr/bin/env python3
"""
Run the functional warming of an experiment (./qflex fw) in the qflex container while perf, on the host,
measures the host cores the vCPUs are pinned to: perf stat -C <cores> -A -I <interval> from the start of QEMU to
the end of fw. If perf stat exits early (unknown event, no permissions, ...), QEMU is stopped right away.

The host cores are the host_core_idx column of <experiment>/run/core_info.csv, the file fw-qemu reads.

Usage in the host, not in the container.
  python3 scripts/run_fw_perf.py <container> <args file> [options]

To rebuild perf_totals.csv and perf_per_ksi.csv of an existing run from its perf_stat.csv:
  python3 scripts/run_fw_perf.py summarize <out folder> [--run-dir <experiment>/run]

Output in <out root>/<experiment>-<mode>-<timestamp>/. The default out root is <working folder>/perf_runs: the
experiment folder is created by root in the container, so the host user cannot write in it.

  run_info.txt        configuration and the events of the run
  run_fw_perf.log     output of this script
  fw.log, fw.args     output and arguments of ./qflex fw
  perf_stat.csv       raw perf stats
  perf_stat.err       stderr of perf stat
  perf_totals.csv     sum per core and event, over the whole run and from the warming start
  perf_per_ksi.csv    the totals per kilo SIMULATED instruction (WormCache Instruction of statistics.final.csv),
                      per host core (the vCPUs pinned to it in core_info.csv) and over all of them
  perf_per_ksi_groups.csv  mean and std of per_ksi_from_warm_start of each event over the server
                      cores (model_type ipc-model in core_info.csv) and over the client cores (constant)

The warming start is when run/statistics.csv first has a block with simulated instructions: before it, QEMU is
loading init_warmed. The totals "from warming start" drop the intervals before it. It is seen with a delay of at
most POLL_S + 10 s (the period of statistics.csv).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, NoReturn, Optional, Tuple

import pandas as pd

DEFAULT_EVENTS = (
    "{cycles,instructions,ref-cycles},"
    "{l2_rqsts.references,l2_rqsts.miss,l2_rqsts.all_demand_miss},"
    "{l1d.replacement,L1-icache-load-misses},"
    "{cycles,cycle_activity.stalls_l1d_pending,cycle_activity.stalls_l2_pending},"
    "{longest_lat_cache.reference,longest_lat_cache.miss},"
    "{mem_load_uops_l3_hit_retired.xsnp_hitm,mem_load_uops_l3_hit_retired.xsnp_hit},"
    "{mem_load_uops_l3_miss_retired.remote_hitm,mem_load_uops_l3_miss_retired.remote_dram},"
    "{dTLB-load-misses,iTLB-load-misses},"
    "context-switches,cpu-migrations"
)


PERF_STAT_COLUMNS = [
    "time",
    "cpu",
    "count",
    "unit",
    "event",
    "run_time",
    "pct_running",
    "metric_value",
    "metric_unit",
]


QEMU_BINARY = "qemu-system-aarch64"
QEMU_COMM = QEMU_BINARY[:15]  # the kernel truncates process names to 15 characters

DEFAULT_SAMPLE_SIZE = 30
DEFAULT_INTERVAL_MS = 10000

POLL_S = 5  # period of the monitoring loop
QEMU_START_TIMEOUT_S = 900  # from ./qflex fw to QEMU
FW_EXIT_TIMEOUT_S = 300  # from stopping QEMU to the end of ./qflex fw

LOG_FILE: Optional[str] = None


class Interrupted(Exception):
    pass


def say(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def die(msg: str) -> NoReturn:
    say(f"ERROR: {msg}")
    sys.exit(1)


def run(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Runs cmd and returns its output (stdout and stderr together)."""
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, stdout=f"{cmd[0]}: not found\n")
    except subprocess.TimeoutExpired as e:
        out = (
            e.output.decode(errors="replace")
            if isinstance(e.output, bytes)
            else (e.output or "")
        )
        return subprocess.CompletedProcess(
            cmd, 124, stdout=out + f"\ntimed out after {timeout} s\n"
        )


@dataclass
class Experiment:
    work: str
    name: str
    path: str
    run: str
    fw_args: List[str]
    sample_size: int


def arg_value(tokens: List[str], key: str) -> Optional[str]:
    for i, token in enumerate(tokens):
        if token == key and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(key + "="):
            return token.split("=", 1)[1]
    return None


def has_arg(tokens: List[str], key: str) -> bool:
    return any(token == key or token.startswith(key + "=") for token in tokens)


def read_args_file(path: str, cli_sample_size: Optional[int]) -> Experiment:
    """Arguments of ./qflex fw, split on whitespace, and the experiment they name."""
    try:
        with open(path) as f:
            tokens = f.read().split()
    except OSError as e:
        die(f"cannot read the args file: {e}")
    work = arg_value(tokens, "--working-folder")
    name = arg_value(tokens, "--experiment-name")
    if not work or not name:
        die(f"--working-folder or --experiment-name missing from {path}")
    work = os.path.normpath(work)

    file_sample_size = arg_value(tokens, "--sample-size")
    if file_sample_size is not None:
        try:
            sample_size = int(file_sample_size)
        except ValueError:
            die(f"invalid --sample-size in {path}: {file_sample_size}")
        if cli_sample_size is not None and cli_sample_size != sample_size:
            die(
                f"--sample-size {cli_sample_size} differs from the --sample-size {sample_size} of {path}"
            )
    else:
        sample_size = (
            DEFAULT_SAMPLE_SIZE if cli_sample_size is None else cli_sample_size
        )
        tokens += ["--sample-size", str(sample_size)]
    if not has_arg(tokens, "--loadvm-name"):
        tokens += ["--loadvm-name", "init_warmed"]
    if not has_arg(tokens, "--parallel"):
        say(
            f"WARNING: --parallel is not in {path}: this script is meant for parallel warming"
        )

    exp = os.path.join(work, "experiments", name)
    return Experiment(
        work=work,
        name=name,
        path=exp,
        run=os.path.join(exp, "run"),
        fw_args=tokens,
        sample_size=sample_size,
    )


def read_host_cores(path: str) -> pd.Series:
    """Host core of each vCPU (the index): row i of core_info.csv is vCPU i, the cpu_index of fw-qemu."""
    try:
        cores = pd.read_csv(path)["host_core_idx"]
    except (OSError, KeyError, ValueError) as e:
        die(f"cannot read the host_core_idx column of {path}: {e!r}")
    if cores.empty or not pd.api.types.is_integer_dtype(cores):
        die(f"{path}: host_core_idx must be one integer per vCPU")
    return cores


def read_core_info(run_dir: str) -> Tuple[List[int], int]:
    """Host cores (unique, sorted) and number of vCPUs of run/core_info.csv."""
    cores = read_host_cores(os.path.join(run_dir, "core_info.csv"))
    return sorted(set(cores.tolist())), len(cores)


def qemu_processes() -> List[Tuple[int, List[str]]]:
    """(pid, argv) of every qemu-system-aarch64 on the host, including the ones in containers."""
    procs = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if argv and os.path.basename(argv[0]) == QEMU_BINARY:
            procs.append((int(entry), argv))
    return procs


def find_qemu(exp_path: str) -> Optional[int]:
    """QEMU of this experiment: its -drive points into the experiment folder."""
    marker = exp_path.rstrip("/") + "/"
    for pid, argv in qemu_processes():
        if any(marker in a for a in argv[1:]):
            return pid
    return None


def clocks() -> str:
    try:
        with open("/proc/uptime") as f:
            uptime = f.read().split()[0]
    except OSError:
        uptime = ""
    return (
        f"epoch={time.time():.3f} mono={time.clock_gettime(time.CLOCK_MONOTONIC):.3f} "
        f"mono_raw={time.clock_gettime(time.CLOCK_MONOTONIC_RAW):.3f} uptime={uptime}"
    )


def md5(path: str) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        return f"unreadable ({e})"
    return h.hexdigest()


class RunInfo:
    """run_info.txt: key=value lines for the configuration, '<event>: <clocks> key=value...' for the events."""

    def __init__(self, path: str):
        self.path = path

    def _append(self, line: str) -> None:
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def write(self, key: str, value: object) -> None:
        self._append(f"{key}={value}")

    def event(self, name: str, **fields: object) -> None:
        extra = "".join(f" {k}={v}" for k, v in fields.items())
        self._append(f"{name}: {clocks()}{extra}")
        say(f"{name}{extra}")


class Statistics:
    """
    Tail of run/statistics.csv. The plugin rewrites the file when QEMU starts and appends one row per vCPU every
    10 s, all with the same ts; Instruction is cumulative. Only the complete blocks are returned.
    """

    def __init__(self, path: str, n_vcpus: int, not_before: float):
        self.path = path
        self.n_vcpus = n_vcpus
        self.not_before = not_before  # epoch: an older file is from a previous run

    def blocks(self) -> List[Tuple[int, int]]:
        """(ts, total Instruction) of the complete blocks of the file, oldest first."""
        try:
            if os.stat(self.path).st_mtime < self.not_before:
                return []
            with open(self.path, "rb") as f:
                data = f.read()
        except OSError:
            return []
        data = data[: data.rfind(b"\n") + 1]  # drop the line being written
        try:
            stats = pd.read_csv(
                io.BytesIO(data), usecols=["ts", "core_id", "Instruction"]
            )
        except ValueError:  # empty, header still being written, columns missing
            return []
        stats = stats.apply(pd.to_numeric, errors="coerce").dropna()
        per_ts = stats.groupby("ts", sort=False).agg(
            rows=("core_id", "size"), instructions=("Instruction", "sum")
        )
        complete = per_ts[per_ts["rows"] == self.n_vcpus]
        return [(int(ts), int(n)) for ts, n in complete["instructions"].items()]

    def first_warm(self) -> Optional[Tuple[int, int]]:
        for ts, instructions in self.blocks():
            if instructions > 0:
                return ts, instructions
        return None

    def fields(self) -> Dict[str, object]:
        blocks = self.blocks()
        if not blocks:
            return {}
        return {"stats_ts": blocks[-1][0], "stats_instructions": blocks[-1][1]}


class StatProfiler:
    """perf stat over the whole run, per host core and interval."""

    def __init__(
        self,
        cores: str,
        events: str,
        interval_ms: int,
        out: str,
        run_dir: str,
        info: RunInfo,
    ):
        self.cores = cores
        self.events = events
        self.interval_ms = interval_ms
        self.out = out
        self.run_dir = run_dir
        self.csv = os.path.join(out, "perf_stat.csv")
        self.err = os.path.join(out, "perf_stat.err")
        self.totals = os.path.join(out, "perf_totals.csv")
        self.info = info
        self.proc: Optional[subprocess.Popen] = None
        self.t0: Optional[float] = None
        self.warm_offset: Optional[float] = (
            None  # seconds from the start of perf to the warming start
        )
        self.stopped = False

    def start(self) -> None:
        cmd = [
            "perf",
            "stat",
            "-C",
            self.cores,
            "-A",
            "-I",
            str(self.interval_ms),
            "-x,",
            "-o",
            self.csv,
            "-e",
            self.events,
        ]
        self.info.write("perf_cmd", shlex.join(cmd))
        with open(self.err, "w") as err:
            self.t0 = time.monotonic()
            # Own session: a Ctrl+C in the terminal reaches only this script, which stops perf after QEMU.
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=err,
                stderr=err,
                start_new_session=True,
            )
        self.info.event("perf_stat_start", pid=self.proc.pid)

    def on_warm_start(self, now: float) -> None:
        if self.t0 is not None:
            self.warm_offset = now - self.t0
            self.info.write("perf_stat_warm_offset_s", f"{self.warm_offset:.3f}")

    def alive(self) -> bool:
        """False once perf stat has exited on its own (bad event, no permissions, ...)."""
        return self.proc is None or self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None or self.stopped:
            return
        self.stopped = True
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.info.event("perf_stat_stop", rc=self.proc.returncode)

    def summarize(self) -> None:
        if not os.path.isfile(self.csv):
            say("no perf_stat.csv to summarize")
            return
        summarize_stat(self.csv, self.warm_offset, self.interval_ms / 1000, self.totals)
        summarize_per_ksi(self.out, self.run_dir)


def cpu_number(cpu: str) -> int:
    digits = "".join(c for c in cpu if c.isdigit())
    return int(digits) if digits else -1


def sort_rows(df: pd.DataFrame, orders: Dict[str, List[str]]) -> pd.DataFrame:
    """Rows sorted by the columns of orders, in this order, each column in the order of its list of values."""
    rank = {col: {v: i for i, v in enumerate(values)} for col, values in orders.items()}
    return df.sort_values(
        list(orders), key=lambda col: col.map(rank[col.name])
    ).reset_index(drop=True)


# Fields of perf stat -x, -I -A (the last two are empty without a derived metric).


def summarize_stat(
    csv_path: str, warm_offset: Optional[float], interval_s: float, totals_path: str
) -> None:
    """Sums perf_stat.csv per core and event and writes perf_totals.csv."""
    try:
        raw = pd.read_csv(
            csv_path,
            comment="#",
            header=None,
            names=PERF_STAT_COLUMNS,
            usecols=range(7),
            low_memory=False,
        )
    except pd.errors.EmptyDataError:  # perf stat died before writing any interval
        say(f"no counts in {csv_path}")
        return
    raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
    raw["count"] = pd.to_numeric(
        raw["count"], errors="coerce"
    )  # <not counted>, <not supported> -> NaN
    raw["pct_running"] = pd.to_numeric(raw["pct_running"], errors="coerce").fillna(
        100.0
    )
    raw = raw.dropna(subset=["time"])
    # The same event in two groups (e.g. cycles) gives two rows per interval and CPU, in the order of -e: the
    # second one becomes "<event>#2". Numbered before dropping the <not counted> rows, so the names do not shift.
    nth = raw.groupby(["time", "cpu", "event"], sort=False).cumcount()
    raw["event"] = raw["event"].where(
        nth == 0, raw["event"] + "#" + (nth + 1).astype(str)
    )
    not_counted = raw.loc[raw["count"].isna(), "event"].value_counts()
    df = raw.dropna(subset=["count"])
    if df.empty:
        say(f"no counts in {csv_path}")
        return

    # Interval [t - interval, t]: counted from the warming start only if it starts after it.
    warm = (
        df["count"].where(df["time"] - interval_s >= warm_offset, 0.0)
        if warm_offset is not None
        else 0.0
    )
    totals = (
        df.assign(warm=warm)
        .groupby(["cpu", "event"], sort=False)
        .agg(
            total=("count", "sum"),
            total_from_warm_start=("warm", "sum"),
            intervals=("count", "size"),
            min_pct_running=("pct_running", "min"),
        )
        .reset_index()
    )
    totals["total"] = totals["total"].round().astype("int64")
    totals["total_from_warm_start"] = (
        totals["total_from_warm_start"].round().astype("Int64")
    )
    if warm_offset is None:
        totals["total_from_warm_start"] = pd.NA
    cpus = sorted(totals["cpu"].unique(), key=cpu_number)
    totals = sort_rows(totals, {"cpu": cpus, "event": list(df["event"].unique())})
    totals.to_csv(totals_path, index=False, float_format="%.2f")

    say(f"perf totals in {totals_path}")
    if not not_counted.empty:
        say(
            "WARNING: intervals not counted: "
            + ", ".join(f"{e} x{n}" for e, n in not_counted.items())
        )


def read_run_info(path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Configuration (key=value lines) and events ('<event>: k=v ...' lines, the last one of each name)."""
    config: Dict[str, str] = {}
    events: Dict[str, Dict[str, str]] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            head, sep, rest = line.partition(": ")
            if sep and "=" not in head:
                events[head] = dict(
                    token.split("=", 1) for token in rest.split() if "=" in token
                )
            elif "=" in line:
                key, value = line.split("=", 1)
                config[key] = value
    return config, events


def find_core_info(run_dir: str) -> Optional[str]:
    """core_info.csv of run/, or the one of cfg/ next to it (a copy of the results may have only that)."""
    for path in (
        os.path.join(run_dir, "core_info.csv"),
        os.path.join(
            os.path.dirname(os.path.normpath(run_dir)), "cfg", "core_info.csv"
        ),
    ):
        if os.path.isfile(path):
            return path
    return None


def fmt(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.3f}" if abs(value) < 1000 else f"{value:.0f}"


def read_final_instructions(
    run_dir: str, not_before: float
) -> Tuple[str, Optional[pd.Series]]:
    """
    Instruction of each vCPU (index core_id) at the end of the run, from statistics.final.csv.
    """
    path = os.path.join(run_dir, "statistics.final.csv")
    if not os.path.isfile(path) or os.stat(path).st_mtime < not_before:
        return path, None
    stats = pd.read_csv(path, usecols=["ts", "core_id", "Instruction"])
    last = stats[stats["ts"] == stats["ts"].max()]
    return path, last.set_index("core_id")["Instruction"].sort_index()


def summarize_per_ksi(out: str, run_dir: str) -> None:
    """
    Every event of perf_totals.csv per kilo simulated instruction (WormCache Instruction of statistics.final.csv),
    per host core and over all of them. The simulated
    instructions of a host core are those of the vCPUs pinned to it in core_info.csv. Writes perf_per_ksi.csv.
    """
    totals_path = os.path.join(out, "perf_totals.csv")
    ksi_path = os.path.join(out, "perf_per_ksi.csv")
    if not os.path.isfile(totals_path):
        say(f"no {totals_path}: no per kilo simulated instruction summary")
        return
    _, run_events = read_run_info(os.path.join(out, "run_info.txt"))
    totals = pd.read_csv(totals_path)
    events = list(totals["event"].unique())
    cpus = sorted(totals["cpu"].unique(), key=cpu_number)
    columns = cpus + ["all"]

    # Simulated instructions of each vCPU at the end of the run.
    core_info = find_core_info(run_dir)
    if core_info is None:
        say(
            f"WARNING: no core_info.csv for {run_dir}: no per kilo simulated instruction summary"
        )
        return
    host_core = read_host_cores(core_info)  # index: vCPU
    n_vcpus = len(host_core)
    launch = run_events.get("fw_launch", {}).get("epoch")
    source, final = read_final_instructions(run_dir, float(launch) if launch else 0.0)
    if final is None or list(final.index) != list(range(n_vcpus)) or final.sum() == 0:
        say(
            f"WARNING: no simulated instructions of vCPUs 0-{n_vcpus - 1} in {source} newer than the run "
        )
        return
    sim_total = int(final.sum())
    end_total = run_events.get("end", {}).get("stats_instructions")
    if end_total and abs(int(end_total) - sim_total) > 0.01 * sim_total:
        say(
            f"WARNING: {source} has {sim_total} instructions, statistics.csv at the end of the run {end_total}: "
            "is it the same run?"
        )

    # Window from the warming start: drop the instructions already simulated when it was seen, split among the
    # vCPUs in proportion to their final counts.
    warm_seen = run_events.get("warm_start", {}).get("stats_instructions")
    has_warm = warm_seen is not None and bool(
        totals["total_from_warm_start"].notna().all()
    )
    warm_scale = (sim_total - int(warm_seen)) / sim_total if has_warm else float("nan")
    if not has_warm:
        say(
            "WARNING: warming start not seen: the per kilo simulated instruction summary is over the whole run"
        )

    # Simulated instructions per host core (the vCPUs pinned to it) and over all of them.
    label = {cpu_number(cpu): cpu for cpu in cpus}
    # Total instructions simulated on each host core and corresponding vCPUs
    sim = (
        pd.DataFrame({"host_core": host_core, "instructions": final})
        .groupby("host_core")
        .agg(
            vcpus=("instructions", lambda s: ";".join(map(str, s.index))),
            sim_instructions=("instructions", "sum"),
        )
    )
    # Renames the first column items from i to CPUi
    sim.index = [label.get(core, f"CPU{core}") for core in sim.index]
    # Add a row with the total
    sim = pd.concat(
        [
            sim,
            pd.DataFrame(
                {
                    "vcpus": [";".join(map(str, final.index))],
                    "sim_instructions": [sim_total],
                },
                index=["all"],
            ),
        ]
    )
    # Add the amount of instructions simulated in the warm period
    sim["sim_instructions_from_warm_start"] = sim["sim_instructions"] * warm_scale

    # Host counts per core and over all of them, divided by the simulated instructions.
    all_counts = totals.groupby("event", sort=False)[
        ["total", "total_from_warm_start"]
    ].sum(min_count=1)

    # Get a dataframe with a row for each CPU and event with count and number of simulated instructions on
    # that CPU
    ksi = (
        pd.concat([totals, all_counts.reset_index().assign(cpu="all")])
        .rename(
            columns={"total": "count", "total_from_warm_start": "count_from_warm_start"}
        )
        .merge(sim, left_on="cpu", right_index=True, how="left")
    )
    ksi["per_ksi"] = 1000 * ksi["count"] / ksi["sim_instructions"]
    ksi["per_ksi_from_warm_start"] = (
        1000 * ksi["count_from_warm_start"] / ksi["sim_instructions_from_warm_start"]
    )
    if not has_warm:
        ksi[
            [
                "count_from_warm_start",
                "sim_instructions_from_warm_start",
                "per_ksi_from_warm_start",
            ]
        ] = float("nan")
    for col in (
        "count",
        "sim_instructions",
        "count_from_warm_start",
        "sim_instructions_from_warm_start",
    ):
        ksi[col] = ksi[col].round().astype("Int64")
    ksi = sort_rows(ksi, {"cpu": columns, "event": events})
    ksi.to_csv(
        ksi_path,
        index=False,
        float_format="%.6f",
        columns=[
            "cpu",
            "vcpus",
            "event",
            "count",
            "sim_instructions",
            "per_ksi",
            "count_from_warm_start",
            "sim_instructions_from_warm_start",
            "per_ksi_from_warm_start",
        ],
    )
    say(
        f"per kilo SIMULATED instruction ({'from the warming start' if has_warm else 'whole run'}) in {ksi_path}, "
        f"simulated instructions of {source}"
    )
    if totals["min_pct_running"].min() < 100:
        say(
            f"events multiplexed (min pct_running {totals['min_pct_running'].min():.1f}%): counts are scaled "
            "estimates"
        )

    if has_warm:
        summarize_groups(out, ksi, host_core, read_roles(core_info), label, events)
    else:
        say(
            "WARNING: warming start not seen: no server/client statistics (they use per_ksi_from_warm_start)"
        )


# model_type of core_info.csv -> role of the vCPU, for the server/client statistics.
ROLE_OF_MODEL = {"ipc-model": "server", "constant": "client"}
GROUPS = ["server", "client"]


def read_roles(path: str) -> Optional[pd.Series]:
    """Role (server, client) of each vCPU (the index) from the model_type of core_info.csv, None without it."""
    info = pd.read_csv(path)
    if "model_type" not in info.columns:
        return None
    return info["model_type"].astype(str).str.strip().map(ROLE_OF_MODEL)


def summarize_groups(
    out: str,
    ksi: pd.DataFrame,
    host_core: pd.Series,
    roles: Optional[pd.Series],
    label: Dict[int, str],
    events: List[str],
) -> None:
    """
    Mean and std (sample, n-1) of per_ksi_from_warm_start of each event over the server cores and over the client
    cores. The role of a host core is the one of the vCPUs pinned to it. Writes perf_per_ksi_groups.csv.
    """
    groups_path = os.path.join(out, "perf_per_ksi_groups.csv")
    if roles is None:
        say("WARNING: no model_type in core_info.csv: no server/client statistics")
        return

    # Role of each host core: the one of its vCPUs, when they all have the same known role.
    core_role = (
        pd.DataFrame({"host_core": host_core, "role": roles})
        .groupby("host_core")["role"]
        .agg(lambda r: r.iloc[0] if r.notna().all() and r.nunique() == 1 else None)
    )
    left_out = [
        label.get(core, f"CPU{core}")
        for core, role in core_role.items()
        if pd.isna(role)
    ]
    if left_out:
        say(
            f"WARNING: {', '.join(left_out)} left out of the server/client statistics: their vCPUs have an "
            "unknown or mixed model_type"
        )
    role_of_cpu = {
        label.get(core, f"CPU{core}"): role
        for core, role in core_role.items()
        if not pd.isna(role)
    }
    rows = ksi[ksi["cpu"].isin(list(role_of_cpu))].assign(
        group=lambda d: d["cpu"].map(role_of_cpu)
    )
    if rows.empty:
        say("WARNING: no server or client core: no server/client statistics")
        return

    stats = (
        rows.groupby(["group", "event"], sort=False)["per_ksi_from_warm_start"]
        .agg(n_cores="count", mean="mean", std="std")  # std of pandas: sample, n-1
        .reset_index()
    )
    stats = sort_rows(stats, {"group": GROUPS, "event": events})
    stats.to_csv(groups_path, index=False, float_format="%.6f")


def launch_fw(
    container: str, qflex_dir: str, fw_args: List[str], log_path: str
) -> subprocess.Popen:
    # Login shell: the PATH that finds uv (shebang of ./qflex) may be set only in the profile.
    inner = "exec ./qflex fw " + shlex.join(fw_args)
    cmd = [
        "docker",
        "exec",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-w",
        qflex_dir,
        container,
        "bash",
        "-lc",
        inner,
    ]
    with open(log_path, "w") as log:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def stop_qemu(container: str) -> None:
    """Killing the docker exec client does not stop the process in the container: stop QEMU there."""
    say(f"stopping QEMU in {container}")
    run(["docker", "exec", container, "pkill", "-TERM", "-x", QEMU_COMM], timeout=60)


def wait_for_qemu(exp_path: str, fw: subprocess.Popen) -> int:
    deadline = time.monotonic() + QEMU_START_TIMEOUT_S
    while True:
        pid = find_qemu(exp_path)
        if pid is not None:
            return pid
        if fw.poll() is not None:
            die(
                f"./qflex fw exited with {fw.returncode} before starting QEMU: see fw.log"
            )
        if time.monotonic() > deadline:
            die(f"QEMU did not start within {QEMU_START_TIMEOUT_S} s: see fw.log")
        time.sleep(1)


def monitor(
    fw: subprocess.Popen, profiler: StatProfiler, stats: Statistics, info: RunInfo
) -> Optional[str]:
    """
    Runs until fw exits. Returns early, with the reason, if perf stat exits on its own: the caller stops QEMU,
    which would otherwise run for hours with nothing measured.
    """
    warm = False
    while fw.poll() is None:
        time.sleep(POLL_S)
        now = time.monotonic()
        if not warm:
            first = stats.first_warm()
            if first is not None:
                warm = True
                info.event("warm_start", first_warm_stats_ts=first[0], **stats.fields())
                profiler.on_warm_start(now)
        if not profiler.alive():
            rc = profiler.proc.returncode if profiler.proc is not None else None
            info.event("stopped", reason="perf_stat_exited", rc=rc)
            return f"perf stat exited with {rc}, see {profiler.err}"
    if not warm:
        say("WARNING: the warming start was never seen in run/statistics.csv")
    return None


def on_signal(signum, frame) -> None:
    raise Interrupted(signal.Signals(signum).name)


def handle_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "container", help="name or id of the running qflex container (docker ps)"
    )
    ap.add_argument(
        "args_file",
        help="args file of ./qflex fw, e.g. /home/cs471-hmp/work/args/ws-4c/ws-4c-parallel.args",
    )
    # Only stat for now: --mode stat is still accepted, and the output folders and run_info.txt keep "stat".
    ap.add_argument(
        "--mode",
        choices=("stat",),
        default="stat",
        help="profiling mode (default %(default)s)",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        help=f"used when the args file has no --sample-size (default {DEFAULT_SAMPLE_SIZE})",
    )
    ap.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="perf stat -I interval (default %(default)s, the period of statistics.csv)",
    )
    ap.add_argument(
        "--events",
        default=DEFAULT_EVENTS,
        help="perf stat -e list (default %(default)s)",
    )
    ap.add_argument(
        "--out-root", help="output root folder (default <working folder>/perf_runs)"
    )
    ap.add_argument(
        "--qflex-dir",
        default="/home/dev/qflex",
        help="qflex folder in the container (default %(default)s)",
    )
    args = ap.parse_args()

    if args.interval_ms < 100:
        ap.error("--interval-ms must be at least 100")
    if args.sample_size is not None and args.sample_size < 1:
        ap.error("--sample-size must be >= 1")
    return args


def main() -> int:
    global LOG_FILE
    args = handle_arguments()

    # Checks
    exp = read_args_file(args.args_file, args.sample_size)
    host_cores, n_vcpus = read_core_info(exp.run)
    cores = ",".join(map(str, host_cores))
    say(
        f"experiment {exp.path}: {n_vcpus} vCPUs on host cores {cores}, {exp.sample_size} samples, mode {args.mode}"
    )

    out_root = args.out_root or os.path.join(exp.work, "perf_runs")
    out = os.path.join(
        out_root, f"{exp.name}-{args.mode}-{time.strftime('%Y%m%d_%H%M%S')}"
    )
    try:
        os.makedirs(out)
    except OSError as e:
        die(f"cannot create {out}: {e}")
    LOG_FILE = os.path.join(out, "run_fw_perf.log")
    info = RunInfo(os.path.join(out, "run_info.txt"))
    events = args.events

    for key, value in (
        ("mode", args.mode),
        ("experiment", exp.path),
        ("container", args.container),
        ("args_file", os.path.abspath(args.args_file)),
        ("sample_size", exp.sample_size),
        ("host_cores", cores),
        ("n_vcpus", n_vcpus),
        ("events", events),
        ("interval_ms", args.interval_ms),
        ("qemu_md5", md5(os.path.join(exp.run, QEMU_BINARY))),
    ):
        info.write(key, value)
    with open(os.path.join(out, "fw.args"), "w") as f:
        f.write("\n".join(exp.fw_args) + "\n")
    shutil.copy2(os.path.abspath(__file__), out)
    shutil.copy2(args.args_file, out)
    say(f"output in {out}")

    stats = Statistics(
        os.path.join(exp.run, "statistics.csv"), n_vcpus, not_before=time.time()
    )
    profiler = StatProfiler(cores, events, args.interval_ms, out, exp.run, info)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    fw: Optional[subprocess.Popen] = None
    interrupted = False
    stop_reason: Optional[str] = None
    t_start = time.monotonic()
    try:
        info.event("fw_launch")
        fw = launch_fw(
            args.container, args.qflex_dir, exp.fw_args, os.path.join(out, "fw.log")
        )
        qemu_pid = wait_for_qemu(exp.path, fw)
        info.event("qemu_start", pid=qemu_pid)
        profiler.start()
        stop_reason = monitor(fw, profiler, stats, info)
    except Interrupted as e:
        interrupted = True
        say(f"interrupted by {e}")
    finally:
        # Stop everything even on errors, and do not let a second signal cut the cleanup short.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if fw is not None and fw.poll() is None:
            stop_qemu(args.container)
            try:
                fw.wait(timeout=FW_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                say(
                    f"WARNING: ./qflex fw still running {FW_EXIT_TIMEOUT_S} s after stopping QEMU: "
                    f"killing the docker exec client, check {args.container}"
                )
                fw.kill()
                fw.wait()
        profiler.stop()

    wall = time.monotonic() - t_start
    rc = fw.returncode if fw is not None else None
    checkpoints = len(glob.glob(os.path.join(exp.run, "snapshot_*.bxdb-meta")))
    info.event(
        "end", fw_rc=rc, wall_s=f"{wall:.0f}", checkpoints=checkpoints, **stats.fields()
    )
    try:
        profiler.summarize()
    except Exception as e:  # the raw data is still in the output folder
        say(f"WARNING: summary failed: {e}")

    if interrupted:
        say(f"INTERRUPTED: partial data in {out}")
        return 130
    if stop_reason:
        say(f"ERROR: QEMU stopped early: {stop_reason}")
        return 1
    if rc != 0:
        say(f"ERROR: ./qflex fw exited with {rc}, see {out}/fw.log")
        return 1
    # ./qflex fw exits with 0 even when QEMU fails: check the checkpoints, as run_repeat.sh does.
    if checkpoints != exp.sample_size:
        say(
            f"ERROR: fw created {checkpoints} checkpoints instead of {exp.sample_size}, see {exp.run}/fp_gen_speed"
        )
        return 1
    say(f"DONE in {wall / 60:.0f} min: {out}")
    return 0


def summarize_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="run_fw_perf.py summarize",
        description="Rebuilds perf_totals.csv and then perf_per_ksi.csv of a stat run from its perf_stat.csv.",
    )
    ap.add_argument(
        "out", help="output folder of the run (perf_stat.csv, run_info.txt)"
    )
    ap.add_argument(
        "--run-dir",
        help="run/ folder of the experiment, with statistics.final.csv and core_info.csv "
        "(default: <experiment>/run, experiment from run_info.txt)",
    )
    args = ap.parse_args(argv)
    info_path = os.path.join(args.out, "run_info.txt")
    if not os.path.isfile(info_path):
        die(f"{info_path} not found")
    config, _ = read_run_info(info_path)
    if config.get("mode") != "stat":
        die(f"{args.out} is not a --mode stat run")
    run_dir = args.run_dir or os.path.join(config.get("experiment", ""), "run")
    if not os.path.isdir(run_dir):
        die(f"{run_dir} not found: pass --run-dir")
    perf_stat = os.path.join(args.out, "perf_stat.csv")
    if not os.path.isfile(perf_stat):
        die(
            f"{perf_stat} not found: the summaries are rebuilt from the raw perf stat counts"
        )

    # Same steps as at the end of a run: perf_stat.csv -> perf_totals.csv -> perf_per_ksi.csv.
    warm = config.get("perf_stat_warm_offset_s")
    warm_offset = float(warm) if warm else None
    interval_s = int(config.get("interval_ms") or DEFAULT_INTERVAL_MS) / 1000
    summarize_stat(
        perf_stat, warm_offset, interval_s, os.path.join(args.out, "perf_totals.csv")
    )
    summarize_per_ksi(args.out, run_dir)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["summarize"]:
        sys.exit(summarize_main(sys.argv[2:]))
    sys.exit(main())
