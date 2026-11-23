"""Dry-run tests of the CryoDRGN training job.

The job is run through `himena_relion.external.run_function`, the same entry point
RELION calls (`himena-relion <import path> --o ... --in_mics ...`). cryoDRGN itself is
replaced by `FakeCryoDRGN` (see fake_cryodrgn.py).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion._pipeline import RelionPipeline
from himena_relion.external import pick_job_class, run_function

from himena_relion_cryodrgn.train import CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject

# The import path is written in job.star as `fn_exe`. Changing it breaks existing jobs.
IMPORT_PATH = "himena_relion_cryodrgn.train:CryoDrgnTrainJob"
STEPS = ["parse_relion", "parse_star", "downsample", "backproject_voxel", "train_vae"]
NETWORK_OPTIONS = ["--zdim", "--enc-layers", "--enc-dim", "--dec-layers", "--dec-dim"]


def _run_job(project: RelionProject, **params) -> None:
    argv = [
        IMPORT_PATH,
        "--o", f"{project.job_dir.as_posix()}/",
        "--in_mics", project.tomograms_star,
        "--in_parts", project.particles_star,
    ]  # fmt: skip
    for key, value in params.items():
        argv += [f"--{key}", str(value)]
    run_function(argv)


def test_plugin_module_registers_job():
    import himena_relion_cryodrgn.main  # noqa: F401  (the himena.plugin entry point)

    assert CryoDrgnTrainJob.import_path() == IMPORT_PATH
    assert pick_job_class(IMPORT_PATH) is CryoDrgnTrainJob
    assert CryoDrgnTrainJob.job_title() == "CryoDRGN Train"


def test_prep_job_star():
    model = CryoDrgnTrainJob.prep_job_star(
        in_mics="Tomograms/job005/tomograms.star",
        in_parts="Extract/job010/particles.star",
        box_size=64,
        dose_per_tilt=2.5,
    )
    opts = dict(zip(model.joboptions_values.variable, model.joboptions_values.value))
    assert opts["fn_exe"] == f"himena-relion {IMPORT_PATH}"
    assert opts["in_mic"] == "Tomograms/job005/tomograms.star"
    assert opts["in_part"] == "Extract/job010/particles.star"
    params = {
        opts[f"param{i}_label"]: opts[f"param{i}_value"]
        for i in range(1, 11)
        if opts[f"param{i}_label"]
    }
    assert params == {"box_size": "64", "dose_per_tilt": "2.5"}


def test_all_parameters_fit_in_the_slots():
    """All the parameters are in the 10 slots of RELION (shown in the RELION GUI)."""
    model = CryoDrgnTrainJob.prep_job_star(
        in_mics="Tomograms/job005/tomograms.star",
        in_parts="Extract/job010/particles.star",
        box_size=64,
        lazy=True,
        n_epochs=50,
        dose_per_tilt=2.5,
        zdim=4,
        enc_layers=2,
        enc_dim=256,
        dec_layers=4,
        dec_dim=512,
        cryodrgn_bin_dir="/opt/cryodrgn/bin",
    )
    opts = dict(zip(model.joboptions_values.variable, model.joboptions_values.value))
    labels = {opts[f"param{i}_label"] for i in range(1, 11)}
    assert labels == set(CryoDrgnTrainJob._signature().parameters) - {
        "in_mics",
        "in_parts",
    }
    assert opts["other_args"] == ""


def test_default_network(relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    _run_job(relion_project)

    cmd = [c for c in fake_cryodrgn.calls if c[1] == "train_vae"][-1]
    # given explicitly: the defaults of cryoDRGN (zdim has none)
    network = {opt: cmd[cmd.index(opt) + 1] for opt in NETWORK_OPTIONS}
    assert network == {
        "--zdim": "8",
        "--enc-layers": "3",
        "--enc-dim": "1024",
        "--dec-layers": "3",
        "--dec-dim": "1024",
    }


def test_network_parameters(relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    network = {
        "zdim": 4,
        "enc_layers": 2,
        "enc_dim": 256,
        "dec_layers": 4,
        "dec_dim": 512,
    }
    _run_job(relion_project, **network)

    vae = fake_cryodrgn.parsed("train_vae")
    assert {name: getattr(vae, name) for name in network} == network
    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()


def test_dry_run_succeeds(relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    _run_job(relion_project)

    assert fake_cryodrgn.subcommands == STEPS
    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert not relion_project.job_file("RELION_JOB_EXIT_FAILURE").exists()


def test_steps_are_chained(relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    """Each step must read the files written by the previous steps."""
    _run_job(relion_project)

    relion = fake_cryodrgn.parsed("parse_relion")
    star = fake_cryodrgn.parsed("parse_star")
    down = fake_cryodrgn.parsed("downsample")
    bp = fake_cryodrgn.parsed("backproject_voxel")
    vae = fake_cryodrgn.parsed("train_vae")

    assert relion.tomograms == relion_project.tomograms_star
    assert relion.particles == relion_project.particles_star
    assert relion.tilt_dim == list(relion_project.tilt_dim)  # (nx, ny) of tilt images
    assert star.input == relion.output
    assert down.input == relion.output
    assert bp.particles == vae.particles == down.outfile
    assert bp.ctf == vae.ctf == os.path.abspath(star.ctf)
    assert bp.poses == vae.poses == os.path.abspath(star.poses)
    assert vae.ind is not None
    assert Path(vae.ind).parent == Path(bp.outdir)
    assert bp.force_ntilts
    assert vae.ntilts == bp.ntilts


@pytest.mark.parametrize("box_size, dose_per_tilt", [(128, 3.0), (64, 2.5)])
def test_parameters_are_forwarded(
    relion_project: RelionProject,
    fake_cryodrgn: FakeCryoDRGN,
    box_size: int,
    dose_per_tilt: float,
):
    _run_job(relion_project, box_size=box_size, dose_per_tilt=dose_per_tilt)

    # ctf.pkl / poses.pkl describe the extracted images, independent of box_size
    star = fake_cryodrgn.parsed("parse_star")
    assert star.D == relion_project.image_size
    assert star.Apix == pytest.approx(relion_project.image_pixel_size)
    assert fake_cryodrgn.parsed("downsample").D == box_size
    bp = fake_cryodrgn.parsed("backproject_voxel")
    assert bp.tilt
    assert bp.dose_per_tilt == pytest.approx(dose_per_tilt)
    vae = fake_cryodrgn.parsed("train_vae")
    assert vae.encode_mode == "tilt"
    assert vae.dose_per_tilt == pytest.approx(dose_per_tilt)


def test_box_size_larger_than_particles_fails(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    with pytest.raises(ValueError, match="box_size"):
        _run_job(relion_project, box_size=relion_project.image_size * 2)

    assert fake_cryodrgn.calls == []
    assert relion_project.job_file("RELION_JOB_EXIT_FAILURE").exists()


def test_output_nodes_exist(relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    _run_job(relion_project)

    job = CryoDrgnTrainJob(ExternalJobDirectory(relion_project.job_dir))
    for path, _label in job.output_nodes():
        assert relion_project.job_file(path).exists(), f"output node {path} is missing"

    pipeline = RelionPipeline.from_star(relion_project.job_file("job_pipeline.star"))
    registered = {node.path.as_posix() for node in pipeline.outputs}
    for path, _label in job.output_nodes():
        assert f"{relion_project.job_dir.as_posix()}/{path}" in registered


def test_weights_link_is_relative(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_job(relion_project)

    link = relion_project.job_file("weights.pkl")
    if not link.is_symlink():
        pytest.skip("symlinks are emulated on this platform")
    assert Path(os.readlink(link)) == Path("train", "weights.pkl")


def test_rerun_in_same_job_directory(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_job(relion_project)
    relion_project.job_file("RELION_JOB_EXIT_SUCCESS").unlink()

    _run_job(relion_project)

    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert relion_project.job_file("weights.pkl").exists()


@pytest.mark.parametrize("failing_step", STEPS)
def test_failure_marks_exit_failure(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN, failing_step: str
):
    fake_cryodrgn.fail_on.add(failing_step)

    with pytest.raises(subprocess.CalledProcessError):
        _run_job(relion_project)

    assert fake_cryodrgn.subcommands == STEPS[: STEPS.index(failing_step) + 1]
    assert relion_project.job_file("RELION_JOB_EXIT_FAILURE").exists()
    assert not relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()


def test_abort_between_steps(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """RELION_JOB_ABORT_NOW written during a step stops the job before the next one."""
    fake_cryodrgn.after["downsample"] = lambda: relion_project.job_file(
        "RELION_JOB_ABORT_NOW"
    ).touch()

    with pytest.raises(RuntimeError, match="aborted"):
        _run_job(relion_project)

    assert fake_cryodrgn.subcommands == STEPS[:3]
    assert relion_project.job_file("RELION_JOB_EXIT_ABORTED").exists()
    assert not relion_project.job_file("RELION_JOB_ABORT_NOW").exists()


def test_parse_relion_output(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_job(relion_project)

    parsed = relion_project.job_file("particles_parsed.star")
    assert Path(fake_cryodrgn.parsed("parse_relion").output) == parsed
    assert Path(fake_cryodrgn.parsed("parse_star").input) == parsed
    assert Path(fake_cryodrgn.parsed("downsample").input) == parsed


def test_training_star_is_the_downsampled_star(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """box_size (128) < extracted box (192): training uses the downsampled images."""
    _run_job(relion_project, box_size=128)

    downsampled = relion_project.job_file("downsample", "particles_downsampled.star")
    assert Path(fake_cryodrgn.parsed("backproject_voxel").particles) == downsampled
    assert Path(fake_cryodrgn.parsed("train_vae").particles) == downsampled


def test_training_star_without_downsampling(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """box_size = extracted box (192): training uses the parsed star."""
    _run_job(relion_project, box_size=relion_project.image_size)

    assert "downsample" not in fake_cryodrgn.subcommands
    parsed = relion_project.job_file("particles_parsed.star")
    assert Path(fake_cryodrgn.parsed("backproject_voxel").particles) == parsed
    assert Path(fake_cryodrgn.parsed("train_vae").particles) == parsed
    assert not relion_project.job_file("downsample").exists()


def test_output_nodes_are_registered(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_job(relion_project)

    pipeline = RelionPipeline.from_star(relion_project.job_file("job_pipeline.star"))
    registered = {node.path.as_posix(): node.type_label for node in pipeline.outputs}
    assert registered == {
        "External/job011/weights.pkl": "CryodrgnWeights.pkl",
        # the RELION particles trained on, passed to CryoDRGN Select
        "External/job011/particles.star": "ParticleGroupMetadata.star",
        # the back-projection of all the training particles (a 3D reference)
        "External/job011/backproject/backproject.mrc": "DensityMap.mrc",
        # the k-means clusters as classes, for Subset selection
        "External/job011/kmeans_optimiser.star": "OptimiserData.star.cryodrgn",
    }
    assert relion_project.job_file("backproject", "backproject.mrc").exists()


def test_input_particles_are_copied(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """particles.star is a copy (not a link) of the input, so it does not change
    when the upstream job is run again."""
    _run_job(relion_project)

    copy = relion_project.job_file("particles.star")
    original = relion_project.root / relion_project.particles_star
    assert not copy.is_symlink()
    assert copy.read_bytes() == original.read_bytes()
