"""Run the jobs with the exact command line that RELION builds from job.star.

`relion_external_command` follows `getCommandsExternalJob` and `prepareFinalCommand`
in RELION 5 (src/pipeline_jobs.cpp):

- `<fn_exe> --o <job dir>`, then `--in_movies`, `--in_mics`, `--in_parts`,
  `--in_coords`, `--in_3dref`, `--in_mask` for non-empty inputs,
- `--<paramN_label> <paramN_value>` for every non-empty label, values unquoted,
- `--j <nr_threads>` and `other_args`,
- `--pipeline_control <job dir>` if the command contains "relion_" anywhere. It
  always does here (himena_relion_cryodrgn); himena-relion >= 0.0.17 accepts it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fake_cryodrgn import FakeCryoDRGN
from himena_relion.external import run_function
from starfile_rs import read_star

from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject
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
    command += f" {opts['other_args']}"
    if "relion_" in command:
        command += f" --pipeline_control {output_dir}"
    return command.split()  # no quoting: the shell splits at whitespace


def _run_as_relion(job_star: JobStarModel, output_dir: str) -> list[str]:
    argv = relion_external_command(job_star, output_dir)
    assert argv[0] == "himena-relion"
    run_function(argv[1:])
    return argv


def test_jobs_accept_pipeline_control(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    train_dir = f"{relion_project.job_dir.as_posix()}/"
    argv = _run_as_relion(
        CryoDrgnTrainJob.prep_job_star(
            in_mics=relion_project.tomograms_star,
            in_parts=relion_project.particles_star,
            box_size=64,
            dose_per_tilt=2.5,
            lazy=True,
        ),
        train_dir,
    )
    # "himena_relion_cryodrgn" contains "relion_", so RELION adds the argument.
    assert argv[-2:] == ["--pipeline_control", train_dir]
    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert fake_cryodrgn.parsed("downsample").D == 64
    assert fake_cryodrgn.parsed("backproject_voxel").lazy

    fake_cryodrgn.filter_selection = [0, 1]
    select_dir = f"{relion_project.select_job_dir.as_posix()}/"
    argv = _run_as_relion(
        CryoDrgnSelectParticlesJob.prep_job_star(
            train_weights=f"{train_dir}weights.pkl", epoch=-1, kmeans=-1
        ),
        select_dir,
    )
    assert argv[-2:] == ["--pipeline_control", select_dir]
    assert relion_project.select_job_file("RELION_JOB_EXIT_SUCCESS").exists()
    output = read_star(relion_project.select_job_file("particles.star"))
    names = output["particles"].trust_loop().to_polars()["rlnTomoParticleName"]
    assert names.to_list() == ["TS_01/1", "TS_01/3"]
