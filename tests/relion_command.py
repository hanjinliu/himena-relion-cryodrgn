"""The command line that RELION builds from the job.star of an External job.

`relion_external_command` follows `getCommandsExternalJob` and `prepareFinalCommand`
in RELION 5 (src/pipeline_jobs.cpp):

- `<fn_exe> --o <job dir>`, then `--in_movies`, `--in_mics`, `--in_parts`,
  `--in_coords`, `--in_3dref`, `--in_mask` for non-empty inputs,
- `--<paramN_label> <paramN_value>` for every non-empty label, values unquoted (an
  empty value leaves a bare `--<label>`),
- `--j <nr_threads>` and `other_args`,
- `--pipeline_control <job dir>` if the command contains "relion_" anywhere. It
  always does here (himena_relion_cryodrgn); himena-relion >= 0.0.17 accepts it.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from himena_relion.external import run_function

if TYPE_CHECKING:
    from himena_relion.schemas import JobStarModel

_INPUT_FLAGS = [
    ("in_mov", "--in_movies"),
    ("in_mic", "--in_mics"),
    ("in_part", "--in_parts"),
    ("in_coords", "--in_coords"),
    ("in_3dref", "--in_3dref"),
    ("in_mask", "--in_mask"),
]


def relion_external_command(job_star: JobStarModel, output_dir: str) -> list[str]:
    opts = {
        k: str(v)
        for k, v in zip(
            job_star.joboptions_values.variable, job_star.joboptions_values.value
        )
    }
    command = f"{opts['fn_exe']} --o {output_dir}"
    for key, flag in _INPUT_FLAGS:
        if opts.get(key, ""):
            command += f" {flag} {opts[key]}"
    for i in range(1, 11):
        if label := opts.get(f"param{i}_label", ""):
            command += f" --{label} {opts.get(f'param{i}_value', '')}"
    command += f" --j {opts['nr_threads']}"
    # other_args: the parameters beyond the 10 slots (himena-relion escapes them with
    # backslashes, which the shell of system() removes)
    other_args = shlex.split(opts.get("other_args", ""))
    if "relion_" in command + " ".join(other_args):
        other_args += ["--pipeline_control", output_dir]
    # no quoting in the slots: the shell splits at whitespace
    return command.split() + other_args


def run_as_relion(job_star: JobStarModel, output_dir: str) -> list[str]:
    """Run the job as RELION does, and return the command line."""
    argv = relion_external_command(job_star, output_dir)
    assert argv[0] == "himena-relion"
    run_function(argv[1:])
    return argv
