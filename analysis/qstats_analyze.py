"""
Analyze the per-quantum host-time breakdown written by fw-qemu with quantum orchestration

Input files in <dir>:
  host_<tid>.csv  generation,t_start_ns,exec_ns,barrier_ns,ctx_switch_ns,bql_wait_ns,
                  exec_cpu_ns,barrier_cpu_ns,n_switches,flags,
                  release_ns,barrier_bql_ns,vol_csw,invol_csw
  vcpu_<tid>.csv  generation,cpu_idx,insns,host_exec_ns,bql_wait_ns,wfi_ps,idle_entries,
                  busy_ps,bql_wait_exec_ns,bql_acq,yield_exits
  threads.csv     tid,host_core,kernel_tid,n_flushes,flush_ns,sched_*,..._ctxt_switches
  meta.txt        quantum_size_ns, process_wall_ns, host_threads

Generations in which any host thread has the STOP, RESUMED or FLUSHED flag are
excluded from the "steady" metrics. The "total" metrics use
the wall-clock time of the whole process (boot, -loadvm, checkpoints, pauses),
as the Cost of the PROPOSAL. All times in the files are host nanoseconds except
`wfi_ps`/`busy_ps` (simulated picoseconds); generation g covers simulated time
[g*Q, (g+1)*Q].

Usage:
scripts/qstats_analyze.py <run>/qstats [--json out.json] [--per-generation out.csv]
                        [--skip-first-s 0.1] [--statistics-final <run>/statistics.final.csv]
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Optional

import pandas as pd

METADATA_FILE = "meta.txt"
THREADS_FILE = "threads.csv"

LAST_ARRIVER, STOP, WAKE_INSIDE, FLUSHED, RESUMED = 1, 2, 4, 8, 16
EXCLUDED_FLAGS = STOP | FLUSHED | RESUMED


def read_metadata(qdir: str) -> dict:
    meta = {}
    path = os.path.join(qdir, METADATA_FILE)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    meta[key] = int(value)
    return meta


def read_threads(qdir: str) -> Optional[pd.DataFrame]:
    """One row per host thread, or None if the file is missing."""
    path = os.path.join(qdir, THREADS_FILE)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def read_per_thread(qdir: str, prefix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(qdir, f"{prefix}_*.csv"))):
        tid = int(re.search(rf"{prefix}_(\d+)\.csv$", path).group(1))
        df = pd.read_csv(path)
        df.insert(0, "tid", tid)
        frames.append(df)
    if not frames:
        sys.exit(f"Error: no {prefix}_*.csv in {qdir}")
    return pd.concat(frames, ignore_index=True)


def row_total(df: pd.DataFrame) -> pd.Series:
    return df["exec_ns"] + df["barrier_ns"] + df["ctx_switch_ns"] + df["bql_wait_ns"]


def consistency_checks(host: pd.DataFrame, vcpu: pd.DataFrame, q_ps: int) -> list[str]:
    """Sanity checks on the raw records (before any exclusion)."""
    issues = []

    # Consecutive generations in every thread (the last one may be missing at exit).
    for tid, g in host.groupby("tid"):
        gaps = (g["generation"].sort_values().diff().dropna() != 1).sum()
        if gaps:
            issues.append(f"host thread {tid}: {gaps} gaps in the generation sequence")

    # Exactly one thread releases the barrier in each generation.
    last = host[(host["flags"] & LAST_ARRIVER) != 0].groupby("generation").size()
    gens = host["generation"].unique()
    multiple = int((last > 1).sum())
    missing = len(gens) - len(last)
    if multiple or missing > 1:  # the very last generation may be cut at exit
        issues.append(
            f"last arriver: {multiple} generations with >1, {missing} with none"
        )

    # The serial section of the barrier is recorded only by the last arriver and is part of its wait.
    is_last = (host["flags"] & LAST_ARRIVER) != 0
    bad = int(((host["release_ns"] > 0) & ~is_last).sum())
    if bad:
        issues.append(
            f"{bad} host records with release_ns > 0 without LAST_ARRIVER flag"
        )
    bad = int((host["release_ns"] > host["barrier_ns"]).sum())
    if bad:
        issues.append(f"{bad} host records with release_ns > barrier_ns")
    bad = int((host["barrier_bql_ns"] > host["barrier_ns"]).sum())
    if bad:
        issues.append(f"{bad} host records with barrier_bql_ns > barrier_ns")

    # WFI cannot exceed the quantum; the BQL waits inside TCG are part of all BQL waits.
    over = int((vcpu["wfi_ps"] > q_ps).sum())
    if over:
        issues.append(f"{over} vCPU records with wfi_ps > quantum")
    bad = int((vcpu["bql_wait_exec_ns"] > vcpu["bql_wait_ns"]).sum())
    if bad:
        issues.append(f"{bad} vCPU records with bql_wait_exec_ns > bql_wait_ns")

    # exec + barrier + ctx + bql == time between two consecutive quantum starts.
    errs = []
    for _, g in host.sort_values(["tid", "generation"]).groupby("tid"):
        next = g.shift(-1)
        next_flags = (
            next["flags"].fillna(0).astype("int64")
        )  # shift() turns the last row into NaN
        ok = (next["generation"] == g["generation"] + 1) & (
            (next_flags & (FLUSHED | RESUMED)) == 0
        )
        errs.append(((next["t_start_ns"] - g["t_start_ns"]) - row_total(g))[ok].abs())
    errs = pd.concat(errs)
    if len(errs) and errs.median() > 10_000:  # more than 10 us off
        issues.append(f"time identity off: median |error| = {errs.median():.0f} ns")
    return issues


def exclude_incorrect_generations(
    host: pd.DataFrame, vcpu: pd.DataFrame, q_ns: int, skip_first_s: float
) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    Drops whole generations in which any host thread has the STOP, FLUSHED or
    RESUMED flag, plus the first `skip_first_s` simulated seconds of the run.
    """
    excluded = set(host.loc[(host["flags"] & EXCLUDED_FLAGS) != 0, "generation"])
    if skip_first_s > 0:
        g_min = int(host["generation"].min())
        first_ok = g_min + int(skip_first_s * 1e9 / q_ns)
        excluded |= set(range(g_min, first_ok))
    h = host[~host["generation"].isin(excluded)].copy()
    v = vcpu[~vcpu["generation"].isin(excluded)].copy()
    h["total_ns"] = row_total(h)
    return h, v, excluded


def add_host_derived_columns(hosts: pd.DataFrame) -> pd.DataFrame:
    """
    Adds to each host record:
      release_g             serial section of the barrier in that generation
      barrier_imbalance_ns  barrier wait due to the imbalance between threads
      exec_noise_ns         off-CPU time before the barrier that is not a BQL sleep
      barrier_noise_ns      off-CPU time inside the barrier that is not a BQL sleep
    """
    # The serial section of the barrier is recorded only by the last arriver, but every
    # thread waits for it. We copy it on every record of the generation.
    release_g = hosts.groupby("generation")["release_ns"].max().rename("release_g")
    hosts = hosts.join(release_g, on="generation")
    hosts["barrier_imbalance_ns"] = (hosts["barrier_ns"] - hosts["release_g"]).clip(
        lower=0
    )
    # Off-CPU time that is not a BQL sleep
    hosts["exec_noise_ns"] = (
        hosts["exec_ns"] + hosts["ctx_switch_ns"] - hosts["exec_cpu_ns"]
    ).clip(lower=0)
    hosts["barrier_noise_ns"] = (
        hosts["barrier_ns"] - hosts["barrier_cpu_ns"] - hosts["barrier_bql_ns"]
    ).clip(lower=0)
    return hosts


def compute_per_host_metrics(
    hosts: pd.DataFrame, threads: pd.DataFrame
) -> pd.DataFrame:
    """One row per host thread: sums over the included generations, fractions of its time."""
    pt = hosts.groupby("tid").agg(
        n_quanta=("generation", "size"),
        exec_ns=("exec_ns", "sum"),
        barrier_ns=("barrier_ns", "sum"),
        barrier_imbalance_ns=("barrier_imbalance_ns", "sum"),
        release_ns=("release_ns", "sum"),
        ctx_switch_ns=("ctx_switch_ns", "sum"),
        bql_wait_ns=("bql_wait_ns", "sum"),
        barrier_bql_ns=("barrier_bql_ns", "sum"),
        exec_cpu_ns=("exec_cpu_ns", "sum"),
        barrier_cpu_ns=("barrier_cpu_ns", "sum"),
        exec_noise_ns=("exec_noise_ns", "sum"),
        barrier_noise_ns=("barrier_noise_ns", "sum"),
        vol_csw=("vol_csw", "sum"),
        invol_csw=("invol_csw", "sum"),
        total_ns=("total_ns", "sum"),
    )
    pt["exec_frac"] = pt["exec_ns"] / pt["total_ns"]
    pt["ctx_switch_frac"] = pt["ctx_switch_ns"] / pt["total_ns"]
    pt["barrier_frac"] = pt["barrier_ns"] / pt["total_ns"]
    pt["imbalance_frac"] = pt["barrier_imbalance_ns"] / pt["total_ns"]
    pt["bql_frac"] = pt["bql_wait_ns"] / pt["total_ns"]
    return pt.join(
        threads.set_index("tid")[
            ["host_core", "flush_ns", "nonvoluntary_ctxt_switches"]
        ]
    )


def compute_per_vcpu_metrics(vcpus: pd.DataFrame, q_ps: int) -> pd.DataFrame:
    """One row per vCPU: sums over the included generations, simulated-time fractions, host cost."""
    pv = vcpus.groupby("cpu_idx").agg(
        n_quanta=("generation", "size"),
        insns=("insns", "sum"),
        busy_ps=("busy_ps", "sum"),
        wfi_ps=("wfi_ps", "sum"),
        host_exec_ns=("host_exec_ns", "sum"),
        bql_wait_ns=("bql_wait_ns", "sum"),
        bql_wait_exec_ns=("bql_wait_exec_ns", "sum"),
        bql_acq=("bql_acq", "sum"),
        yield_exits=("yield_exits", "sum"),
        idle_entries=("idle_entries", "sum"),
    )
    sim_ps = pv["n_quanta"] * q_ps
    pv["busy_frac"] = pv["busy_ps"] / sim_ps
    pv["wfi_frac"] = pv["wfi_ps"] / sim_ps
    pv["other_frac"] = 1 - pv["busy_frac"] - pv["wfi_frac"]
    pv["ns_per_insn"] = pv["host_exec_ns"] / pv["insns"]
    pv["mips_while_executing"] = pv["insns"] / pv["host_exec_ns"] * 1e3
    pv["bql_wait_loop_ns"] = pv["bql_wait_ns"] - pv["bql_wait_exec_ns"]
    pv["bql_acq_per_quantum"] = pv["bql_acq"] / pv["n_quanta"]
    pv["bql_wait_per_acq_ns"] = pv["bql_wait_ns"] / pv["bql_acq"].where(
        pv["bql_acq"] > 0
    )
    pv["yield_per_quantum"] = pv["yield_exits"] / pv["n_quanta"]
    return pv


def compute_per_generation(
    filtered_hosts: pd.DataFrame, filtered_vcpu: pd.DataFrame, q_ps: int
) -> pd.DataFrame:
    """
    One row per included generation: host times summed over the threads, plus the
    WFI and busy fractions of each vCPU.
    """
    per_gen = filtered_hosts.groupby("generation").agg(
        wall_ns=("total_ns", "max"),
        barrier_ns=("barrier_ns", "sum"),
        barrier_cpu_ns=("barrier_cpu_ns", "sum"),
        exec_ns=("exec_ns", "sum"),
        bql_wait_ns=("bql_wait_ns", "sum"),
        release_ns=("release_ns", "max"),
        barrier_imbalance_ns=("barrier_imbalance_ns", "sum"),
        barrier_bql_ns=("barrier_bql_ns", "sum"),
        invol_csw=("invol_csw", "sum"),
    )
    wfi = (
        filtered_vcpu.pivot_table(
            index="generation", columns="cpu_idx", values="wfi_ps"
        )
        / q_ps
    )
    wfi.columns = [f"wfi_frac_{c}" for c in wfi.columns]
    busy = (
        filtered_vcpu.pivot_table(
            index="generation", columns="cpu_idx", values="busy_ps"
        )
        / q_ps
    )
    busy.columns = [f"busy_frac_{c}" for c in busy.columns]
    return per_gen.join(wfi).join(busy)


def ratio(a, b) -> float:
    return float(a / b) if b else 0.0


def compute_summary(
    host: pd.DataFrame,
    vcpu: pd.DataFrame,
    filtered_hosts: pd.DataFrame,
    filtered_vcpu: pd.DataFrame,
    excluded: set,
    threads: pd.DataFrame,
    q_ns: int,
    process_wall_ns: int,
) -> dict:
    """
    Global metrics of the run. The "total" ones use all the records and the process
    wall-clock time, the "steady" ones only the included generations.
    """
    q_ps = q_ns * 1000
    g_min, g_max = host["generation"].min(), host["generation"].max()
    n_gen = filtered_hosts["generation"].nunique()
    t_sim_s = n_gen * q_ns / 1e9
    t_sim_total_s = (g_max - g_min + 1) * q_ns / 1e9
    # Host wall time of the included generations: every thread spans the same interval.
    t_wall_s = filtered_hosts.groupby("generation")["total_ns"].max().sum() / 1e9
    process_wall_s = process_wall_ns / 1e9
    insns = int(filtered_vcpu["insns"].sum())
    insns_all = int(vcpu["insns"].sum())
    release_sum_s = (
        filtered_hosts.groupby("generation")["release_g"].first().sum() / 1e9
    )
    M = int(threads["host_core"].nunique())

    summary = {
        "quantum_size_ns": q_ns,
        "host_threads": int(host["tid"].nunique()),
        "host_cores_M": M,
        "generations_total": int(g_max - g_min + 1),
        "generations_included": int(n_gen),
        "generations_excluded": int(len(excluded)),
        "sim_time_total_s": t_sim_total_s,
        "sim_time_included_s": t_sim_s,
        "process_wall_s": process_wall_s,
        "wall_included_s": t_wall_s,
        "cost_total": ratio(M * process_wall_s, t_sim_total_s),
        "mips_per_host_core_total": ratio(insns_all, M * process_wall_s) / 1e6,
        "cost_steady": ratio(M * t_wall_s, t_sim_s),
        "mips_per_host_core_steady": ratio(insns, M * t_wall_s) / 1e6,
        "total_insns": insns_all,
        # Thread view: share of each thread's time spent at the barrier.
        "barrier_fraction_threads": ratio(
            filtered_hosts["barrier_ns"].sum(), filtered_hosts["total_ns"].sum()
        ),
        "barrier_fraction_cores_wall": ratio(
            filtered_hosts["barrier_ns"].sum() / 1e9, M * t_wall_s
        ),
        "barrier_fraction_cores_cpu": ratio(
            filtered_hosts["barrier_cpu_ns"].sum() / 1e9, M * t_wall_s
        ),
        "imbalance_fraction_cores": ratio(
            filtered_hosts["barrier_imbalance_ns"].sum() / 1e9, M * t_wall_s
        ),
        "release_fraction_wall": ratio(release_sum_s, t_wall_s),
        "barrier_bql_fraction_wall": ratio(
            filtered_hosts["barrier_bql_ns"].sum() / 1e9, t_wall_s
        ),
        "bql_fraction_threads": ratio(
            filtered_hosts["bql_wait_ns"].sum(), filtered_hosts["total_ns"].sum()
        ),
        "noise_fraction_threads": ratio(
            filtered_hosts["exec_noise_ns"].sum()
            + filtered_hosts["barrier_noise_ns"].sum(),
            filtered_hosts["total_ns"].sum(),
        ),
        "invol_csw_per_quantum": ratio(
            filtered_hosts["invol_csw"].sum(), len(filtered_hosts)
        ),
        "busy_fraction_mean": ratio(
            filtered_vcpu["busy_ps"].sum(), len(filtered_vcpu) * q_ps
        ),
        "wfi_fraction_mean": ratio(
            filtered_vcpu["wfi_ps"].sum(), len(filtered_vcpu) * q_ps
        ),
    }
    return summary


def compare_with_wormcache(
    vcpu: pd.DataFrame, statistics_final_path: str
) -> pd.DataFrame:
    """
    For each core compares the sum insns of qstats over all generations and the final Instruction counter of
    WormCache found in statistics.final.csv. rel_diff is expected small and positive.
    """
    sf = pd.read_csv(statistics_final_path)
    wc = sf.groupby("core_id")["Instruction"].last()
    qs = vcpu.groupby("cpu_idx")["insns"].sum()
    insn_cmp = pd.DataFrame({"qstats_insns": qs, "wormcache_instruction": wc})
    insn_cmp["rel_diff"] = (
        insn_cmp["qstats_insns"] / insn_cmp["wormcache_instruction"] - 1
    )
    return insn_cmp


def print_report(
    qdir: str,
    summary: dict,
    pt: pd.DataFrame,
    pv: pd.DataFrame,
    insn_cmp: Optional[pd.DataFrame],
) -> None:
    """Prints the summary, the per-thread and per-vCPU tables, the WormCache comparison and the issues."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"== {qdir}")
    for key, value in summary.items():
        if key != "consistency_issues":
            print(
                f"{key:28s} {value:.6g}"
                if isinstance(value, float)
                else f"{key:28s} {value}"
            )
    print("\n== per host thread (ns)")
    print(pt.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\n== per vCPU")
    print(pv.to_string(float_format=lambda x: f"{x:.4g}"))
    if insn_cmp is not None:
        print(
            "\n== instructions: qstats (all generations) vs WormCache statistics.final.csv"
        )
        print(insn_cmp.to_string(float_format=lambda x: f"{x:.4g}"))
    issues = summary["consistency_issues"]
    print("\n== consistency checks")
    print("\n".join(f"  ! {i}" for i in issues) if issues else "  all OK")


def write_json(
    path: str,
    summary: dict,
    pt: pd.DataFrame,
    pv: pd.DataFrame,
    insn_cmp: Optional[pd.DataFrame],
) -> None:
    """Writes the summary, the per-thread and per-vCPU tables and the WormCache comparison."""
    out = {
        "summary": summary,
        "per_thread": pt.reset_index().to_dict(orient="records"),
        "per_vcpu": pv.reset_index().to_dict(orient="records"),
    }
    if insn_cmp is not None:
        out["instructions_vs_wormcache"] = (
            insn_cmp.rename_axis("cpu_idx").reset_index().to_dict(orient="records")
        )
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


def handle_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("qstats_dir")
    ap.add_argument("--json", help="write the summary to this JSON file")
    ap.add_argument("--per-generation", help="write a per-generation table (CSV)")
    ap.add_argument(
        "--wfi-corr",
        help="write the vCPU x vCPU correlation of the per-quantum WFI fraction (CSV)",
    )
    ap.add_argument(
        "--skip-first-s",
        type=float,
        default=0.0,
        help="also exclude the first N simulated seconds of the run",
    )
    ap.add_argument(
        "--statistics-final",
        help="WormCache statistics.final.csv: compare its Instruction counter with sum(insns)",
    )
    return ap.parse_args()


def main():
    args = handle_arguments()

    qdir = args.qstats_dir
    meta = read_metadata(qdir)
    host = read_per_thread(qdir, "host")
    vcpu = read_per_thread(qdir, "vcpu")
    threads = read_threads(qdir)
    if not meta or threads is None:
        print(
            f"Warning: {METADATA_FILE}/{THREADS_FILE} missing",
            file=sys.stderr,
        )
        sys.exit(1)

    q_ns = meta.get("quantum_size_ns", 10_000)
    q_ps = q_ns * 1000
    process_wall_ns = meta.get("process_wall_ns", meta.get("wall_ns", 0))

    issues = consistency_checks(host, vcpu, q_ps)

    filtered_hosts, filtered_vcpu, excluded = exclude_incorrect_generations(
        host, vcpu, q_ns, args.skip_first_s
    )
    filtered_hosts = add_host_derived_columns(filtered_hosts)

    pt = compute_per_host_metrics(filtered_hosts, threads)

    pv = compute_per_vcpu_metrics(filtered_vcpu, q_ps)

    # Simulated time: busy + wfi == Q on average
    for cpu, frac in (pv["busy_frac"] + pv["wfi_frac"]).items():
        if abs(frac - 1) > 0.02:
            issues.append(
                f"vCPU {cpu}: (busy_ps + wfi_ps) / quantum = {frac:.4f} on average"
            )

    summary = compute_summary(
        host,
        vcpu,
        filtered_hosts,
        filtered_vcpu,
        excluded,
        threads,
        q_ns,
        process_wall_ns,
    )
    summary["consistency_issues"] = issues

    insn_cmp = (
        compare_with_wormcache(vcpu, args.statistics_final)
        if args.statistics_final
        else None
    )

    print_report(qdir, summary, pt, pv, insn_cmp)

    if args.json:
        write_json(args.json, summary, pt, pv, insn_cmp)

    if args.per_generation:
        compute_per_generation(filtered_hosts, filtered_vcpu, q_ps).to_csv(
            args.per_generation
        )


if __name__ == "__main__":
    main()
