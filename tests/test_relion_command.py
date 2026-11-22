"""Run the jobs with the exact command line that RELION builds from job.star."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fake_cryodrgn import FakeCryoDRGN
from relion_command import run_as_relion as _run_as_relion
from starfile_rs import read_star

from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject


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
