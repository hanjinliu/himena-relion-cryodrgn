"""Dry-run tests of the CryoDRGN Select Particles job.

A CryoDRGN Train job is run first (with the fake cryoDRGN), then the selection job is
run through `himena_relion.external.run_function` like RELION does. The user's lasso
selection in the `cryodrgn filter` window is given by `FakeCryoDRGN.filter_selection`.

Particles in the project (see conftest.PARTICLES), with their numbering:

    particles.star row        0      1      2      3      4      5
    rlnTomoParticleName     TS_01/1 TS_01/2 TS_01/3 TS_01/4 TS_01/5 TS_01/6
    cryoDRGN particle         0      -      1      2      3      4   (TS_01/2: no tilt)
    point in the plot         0      -      1      -      2      3   (TS_01/4: < ntilts)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._pipeline import RelionPipeline
from himena_relion.external import pick_job_class, run_function
from starfile_rs import read_star

from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob

if TYPE_CHECKING:
    from conftest import RelionProject

IMPORT_PATH = "himena_relion_cryodrgn.select_particles:CryoDrgnSelectParticlesJob"


def _run_select(project: RelionProject, **params) -> None:
    argv = [
        IMPORT_PATH,
        "--o", f"{project.select_job_dir.as_posix()}/",
        "--train_weights", f"{project.job_dir.as_posix()}/weights.pkl",
    ]  # fmt: skip
    for key, value in params.items():
        argv += [f"--{key}", str(value)]
    run_function(argv)


def _particle_names(star_path) -> list[str]:
    df = read_star(star_path)["particles"].trust_loop().to_polars()
    return df["rlnTomoParticleName"].to_list()


def test_plugin_module_registers_job():
    import himena_relion_cryodrgn.main  # noqa: F401  (the himena.plugin entry point)

    assert CryoDrgnSelectParticlesJob.import_path() == IMPORT_PATH
    assert pick_job_class(IMPORT_PATH) is CryoDrgnSelectParticlesJob
    assert CryoDrgnSelectParticlesJob.job_title() == "CryoDRGN Select"


def test_prep_job_star():
    model = CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights="External/job011/weights.pkl", epoch=20, kmeans=-1
    )
    opts = dict(zip(model.joboptions_values.variable, model.joboptions_values.value))
    assert opts["fn_exe"] == f"himena-relion {IMPORT_PATH}"
    params = {
        opts[f"param{i}_label"]: opts[f"param{i}_value"]
        for i in range(1, 11)
        if opts[f"param{i}_label"]
    }
    assert params == {
        "train_weights": "External/job011/weights.pkl",
        "epoch": "20",
        "kmeans": "-1",
    }


def test_selected_particles_star(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    fake_cryodrgn.filter_selection = [1, 2]  # points of TS_01/3 and TS_01/5

    _run_select(trained_project)

    assert fake_cryodrgn.subcommands == [
        "filter",
        "filter_star",
        "backproject_voxel",
    ]
    assert trained_project.select_job_file("RELION_JOB_EXIT_SUCCESS").exists()
    output = trained_project.select_job_file("particles.star")
    assert _particle_names(output) == ["TS_01/3", "TS_01/5"]


def test_output_is_a_relion_tomo_particles_star(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    fake_cryodrgn.filter_selection = [0, 3]
    _run_select(trained_project)

    output = read_star(trained_project.select_job_file("particles.star"))
    original = read_star(trained_project.particles_star)
    assert list(output.keys()) == list(original.keys())  # data_general is kept
    assert (
        output["general"].trust_single().to_dict()
        == original["general"].trust_single().to_dict()
    )
    out_particles = output["particles"].trust_loop().to_polars()
    orig_particles = original["particles"].trust_loop().to_polars()
    assert out_particles.columns == orig_particles.columns
    assert out_particles.rows() == orig_particles[[0, 5]].rows()
    assert (
        output["optics"]
        .trust_loop()
        .to_polars()
        .equals(original["optics"].trust_loop().to_polars())
    )


def test_filter_command(trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    fake_cryodrgn.filter_selection = [0]
    _run_select(trained_project, epoch=20, kmeans=20)

    flt = fake_cryodrgn.parsed("filter")
    assert Path(flt.outdir) == trained_project.job_file("train")
    assert flt.epoch == 20
    assert flt.kmeans == 20
    assert flt.force  # never prompt on the terminal
    assert Path(flt.sel_dir) == trained_project.select_job_file()
    star = fake_cryodrgn.parsed("filter_star")
    # the copy of the RELION particles in the Train job
    assert Path(star.input) == trained_project.job_file("particles.star")
    assert not star.et  # particles.star has one row per particle


def test_train_job_without_particles_copy(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """A Train job run before particles.star was copied: its input is used."""
    trained_project.job_file("particles.star").unlink()
    fake_cryodrgn.filter_selection = [1, 2]

    _run_select(trained_project)

    star = fake_cryodrgn.parsed("filter_star")
    assert Path(star.input) == trained_project.root / trained_project.particles_star
    output = trained_project.select_job_file("particles.star")
    assert _particle_names(output) == ["TS_01/3", "TS_01/5"]


def test_commands_run_from_bin_dir(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN, cryodrgn_bin_dir: Path
):
    fake_cryodrgn.filter_selection = [0]
    _run_select(trained_project, cryodrgn_bin_dir=cryodrgn_bin_dir)

    assert fake_cryodrgn.subcommands == [
        "filter",
        "filter_star",
        "backproject_voxel",
    ]
    for cmd in fake_cryodrgn.calls:
        assert Path(cmd[0]).parent == cryodrgn_bin_dir


def test_output_node(trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    fake_cryodrgn.filter_selection = [0]
    _run_select(trained_project)

    pipeline = RelionPipeline.from_star(
        trained_project.select_job_file("job_pipeline.star")
    )
    registered = {node.path.as_posix(): node.type_label for node in pipeline.outputs}
    job = trained_project.select_job_dir.as_posix()
    assert registered == {
        f"{job}/particles.star": "ParticleGroupMetadata.star",
        # the back-projection of the selected particles (a 3D reference)
        f"{job}/backproject/backproject.mrc": "DensityMap.mrc",
    }
    assert trained_project.select_job_file("backproject", "backproject.mrc").exists()


def test_no_selection_fails(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    fake_cryodrgn.filter_selection = None  # window closed without selecting

    with pytest.raises(RuntimeError, match="No particles were selected"):
        _run_select(trained_project)

    assert fake_cryodrgn.subcommands == ["filter"]
    assert trained_project.select_job_file("RELION_JOB_EXIT_FAILURE").exists()
    assert not trained_project.select_job_file("particles.star").exists()


def test_missing_kmeans_fails(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    fake_cryodrgn.filter_selection = [0]

    with pytest.raises(subprocess.CalledProcessError):
        _run_select(trained_project, kmeans=5)  # only kmeans20 exists

    assert trained_project.select_job_file("RELION_JOB_EXIT_FAILURE").exists()


def test_weights_of_another_job_are_rejected(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    # A job.star of a job that is not CryoDRGN Train
    other = trained_project.root / "External/job013"
    other.mkdir()
    CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights="External/job011/weights.pkl"
    ).to_star_dict().write(other / "job.star")
    (other / "weights.pkl").touch()

    with pytest.raises(ValueError, match="CryoDRGN Train"):
        _run_select(trained_project, train_weights="External/job013/weights.pkl")

    assert fake_cryodrgn.calls == []


def test_selection_is_backprojected(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """The selected particles, not all of them, are back-projected to validate it."""
    fake_cryodrgn.filter_selection = [1, 2]  # cryoDRGN particles 1 and 3
    _run_select(trained_project, dose_per_tilt=2.5, lazy=True)

    bp = fake_cryodrgn.parsed("backproject_voxel")
    # the images, CTF and poses of the training (one row per tilt image)
    assert Path(bp.particles) == trained_project.job_file(
        "downsample", "particles_downsampled.star"
    )
    assert Path(bp.ctf) == trained_project.job_file("ctf.pkl")
    assert Path(bp.poses) == trained_project.job_file("poses.pkl")
    assert Path(bp.ind) == trained_project.select_job_file("indices.pkl")
    assert not bp.force_ntilts  # would override --ind
    assert bp.tilt
    assert bp.dose_per_tilt == pytest.approx(2.5)
    assert bp.lazy
    assert fake_cryodrgn.backprojected["backproject"] == [1, 3]
    assert trained_project.select_job_file("backproject", "backproject.mrc").exists()


def test_input_particles(trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    """in_parts: the RELION particles the model was trained on (an input node)."""
    fake_cryodrgn.filter_selection = [1, 2]
    _run_select(trained_project, in_parts=trained_project.particles_star)

    star = fake_cryodrgn.parsed("filter_star")
    # as given by RELION, relative to the project (the job runs there)
    assert Path(star.input) == Path(trained_project.particles_star)
    output = trained_project.select_job_file("particles.star")
    assert _particle_names(output) == ["TS_01/3", "TS_01/5"]
    job_star = CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights="External/job011/weights.pkl",
        in_parts=trained_project.particles_star,
    )
    opts = dict(
        zip(job_star.joboptions_values.variable, job_star.joboptions_values.value)
    )
    assert opts["in_part"] == trained_project.particles_star  # an input edge


def test_input_particles_copy(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """A copy of the trained particles (e.g. moved by the user) can be used."""
    import shutil

    copy = trained_project.root / "Select/job030/particles.star"
    copy.parent.mkdir(parents=True)
    shutil.copy(trained_project.root / trained_project.particles_star, copy)
    fake_cryodrgn.filter_selection = [0]

    _run_select(trained_project, in_parts="Select/job030/particles.star")

    assert Path(fake_cryodrgn.parsed("filter_star").input).resolve() == copy
    output = trained_project.select_job_file("particles.star")
    assert list(read_star(output).keys()) == ["general", "optics", "particles"]


def test_select_from_continued_train_job(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """A continued Train job (fn_exe of the continue class) is still a Train job."""
    from test_continue import _continue_job

    _continue_job(trained_project, n_epochs=30)
    fake_cryodrgn.calls.clear()
    fake_cryodrgn.filter_selection = [1, 2]

    _run_select(trained_project)

    assert trained_project.select_job_file("RELION_JOB_EXIT_SUCCESS").exists()
    output = trained_project.select_job_file("particles.star")
    assert _particle_names(output) == ["TS_01/3", "TS_01/5"]
