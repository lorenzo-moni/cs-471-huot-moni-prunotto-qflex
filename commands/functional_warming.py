import math
import os
import re
import shlex

from commands import Executor
from .config import ExperimentContext
from commands.qemu import QemuCommonArgParser

# Sample size requested from the last functional warming, in the run folder.
FW_SAMPLE_SIZE_FILE = "fw_sample_size"


def check_qstats_dir(qstats_dir: str) -> str:
    """
    Validates --qstats-dir and returns it normalized ("" when qstats is off).
    The folder is deleted before the run, so it must stay inside the run folder;
    a comma would split the -quantum option of QEMU.
    """
    if not qstats_dir:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", qstats_dir):
        raise ValueError(f"--qstats-dir {qstats_dir!r}: use only letters, digits and . _ - / (a comma would split the -quantum option).")
    path = os.path.normpath(qstats_dir)
    if os.path.isabs(path) or path in (".", "..") or path.startswith("../"):
        raise ValueError(f"--qstats-dir {qstats_dir!r}: give a folder inside the run folder, e.g. qstats (it is emptied before the run).")
    return path


class FunctionalWarming(Executor):
    """
    This class handles the functional warming phase of the experiment and generates checkpoints.
    """

    def __init__(self,
                 experiment_context: ExperimentContext,
                 sample_size: int,
                 qstats_dir: str = ""):
        self.experiment_context = experiment_context
        self.simulation_context = self.experiment_context.simulation_context
        self.qemu_common_parser = QemuCommonArgParser(experiment_context)
        self.sample_size = sample_size
        population = self.experiment_context.workload.population
        if population is None:
            raise ValueError("Error: functional warming needs --population-seconds (the config.json of this prebuilt workload has no suggested population).")
        self.sampling_interval = math.ceil(
            (population + self.sample_size - 1) / self.sample_size
        )
        # Folder of the per-quantum host-time breakdown (qstats), relative to the run folder; "" = off.
        self.qstats_dir = check_qstats_dir(qstats_dir)
        if self.qstats_dir and not self.simulation_context.is_parallel:
            raise ValueError("--qstats-dir needs the parallel mode: fw-qemu ignores it with --sequential, and --no-parallel uses -icount.")





    def cmd(self) -> str | list[str]:
        fw_cmd = f"""
            ./qemu-system-aarch64 \
            {self.qemu_common_parser.get_qemu_base_args()} \
            {self.qemu_common_parser.quantum_args(stats_dir=self.qstats_dir)} \
            -plugin ../lib/libworm_cache.so,mode=warm,init_threshold={self.sampling_interval},interval={self.sampling_interval},count={self.sample_size} \
        """
        print("fw command:")
        print(fw_cmd)
        run_folder = f"{self.experiment_context.get_experiment_folder_address()}/run"
        qstats_before, qstats_after = [], []
        if self.qstats_dir:
            # fw-qemu creates the folder but never deletes old files (e.g. host_<tid>.csv of a run
            # with more threads), which the analyzer would mix with the new ones.
            qstats_before = [f"rm -rf ./{self.qstats_dir}"]
            qstats_after = ["echo " + shlex.quote(
                f"qstats written to {run_folder}/{self.qstats_dir}, analyze it with: "
                f"uv run python scripts/qstats_analyze.py {run_folder}/{self.qstats_dir} "
                f"--statistics-final {run_folder}/statistics.final.csv"
            )]
        return [
            f"cd {run_folder}",
            # Read by `result` to check that every requested checkpoint was simulated.
            f"echo {self.sample_size} > {FW_SAMPLE_SIZE_FILE}",
            *qstats_before,
            fw_cmd,
            *qstats_after,
            # TODO add the proper conditions to only create log and fp_gen_speed at the right time
            "rm -rf fp_gen_speed",
            "mkdir fp_gen_speed",
            "mv *.log ./fp_gen_speed",
        ]
