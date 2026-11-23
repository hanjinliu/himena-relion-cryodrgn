"""Continue - CryoDRGN Train: train longer from the last checkpoint (train_vae --load).

The continued job.star is made by `RelionExternalContinue.make_job_star`, as the
scheduler does, and the job is run with the command line RELION builds from it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import ExternalJobDirectory
from relion_command import run_as_relion

from himena_relion_cryodrgn import _analysis
from himena_relion_cryodrgn.train import CryoDrgnTrainContinue, CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject


def _continue_job(project: RelionProject, **params) -> list[str]:
    """Continue the Train job with updated parameters, as RELION runs it."""
    job_dir = ExternalJobDirectory(project.root / project.job_dir)
    job_star = CryoDrgnTrainContinue(job_dir).make_job_star(**params)
    job_star.to_star_dict().write(job_dir.job_star())
    for name in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE"):
        job_dir.path.joinpath(name).unlink(missing_ok=True)
    return run_as_relion(job_star, f"{project.job_dir.as_posix()}/")


def test_continue_training(trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    argv = _continue_job(trained_project, n_epochs=30)

    assert CryoDrgnTrainContinue.import_path() in argv
    assert fake_cryodrgn.subcommands == ["train_vae"]  # no preprocessing again
    vae = fake_cryodrgn.parsed("train_vae")
    assert Path(vae.load) == trained_project.job_file("train", "weights.20.pkl")
    assert vae.num_epochs == 30  # total, not additional
    # the same data as the first run
    assert Path(vae.particles) == trained_project.job_file(
        "downsample", "particles_downsampled.star"
    )
    assert Path(vae.ind) == trained_project.job_file(
        "backproject", "indices_force_ntilts_10.pkl"
    )
    assert vae.encode_mode == "tilt"
    assert vae.dose_per_tilt == pytest.approx(3.0)
    assert Path(vae.outdir) == trained_project.job_file("train")

    assert trained_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    log = _analysis.parse_train_log(trained_project.job_file("train", "run.log"))
    assert log.epochs == list(range(1, 31))
    assert log.finished
    assert trained_project.job_file("train", "analyze.30").is_dir()
    assert trained_project.job_file("weights.pkl").exists()
    # the copy of the input particles (output node) is kept
    assert trained_project.job_file("particles.star").read_bytes() == (
        (trained_project.root / trained_project.particles_star).read_bytes()
    )


def test_continue_job_star(trained_project: RelionProject):
    job_dir = ExternalJobDirectory(trained_project.root / trained_project.job_dir)
    job_star = CryoDrgnTrainContinue(job_dir).make_job_star(n_epochs=50)
    opts = dict(
        zip(job_star.joboptions_values.variable, job_star.joboptions_values.value)
    )
    params = {
        opts[f"param{i}_label"]: opts[f"param{i}_value"]
        for i in range(1, 11)
        if opts[f"param{i}_label"]
    }

    assert job_star.job.job_is_continue == 1
    assert opts["fn_exe"] == f"himena-relion {CryoDrgnTrainContinue.import_path()}"
    assert opts["in_mic"] == trained_project.tomograms_star
    assert params["n_epochs"] == "50"
    assert params["box_size"] == "128"


def test_continue_needs_more_epochs(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    with pytest.raises(ValueError, match="larger than the epochs already trained"):
        _continue_job(trained_project, n_epochs=20)

    assert fake_cryodrgn.calls == []
    assert trained_project.job_file("RELION_JOB_EXIT_FAILURE").exists()


def test_continue_without_checkpoint(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    with pytest.raises(RuntimeError, match="No checkpoint"):
        _continue_job(relion_project, n_epochs=30)
    assert fake_cryodrgn.calls == []


def test_continue_with_ctf_premultiplied_particles(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    # e.g. a job trained before the check was added to CryoDRGN Train
    trained_project.set_ctf_premultiplied()

    with pytest.raises(ValueError, match="CTF-premultiplied"):
        _continue_job(trained_project, n_epochs=30)

    assert fake_cryodrgn.calls == []
    assert trained_project.job_file("RELION_JOB_EXIT_FAILURE").exists()


def test_continue_with_the_same_network(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """The checkpoint is loaded into the network of the first run (from job.star).

    The fake train_vae fails with --load and another network, as cryoDRGN does.
    """
    network = {
        "zdim": 4,
        "enc_layers": 2,
        "enc_dim": 256,
        "dec_layers": 4,
        "dec_dim": 512,
    }
    job_star = CryoDrgnTrainJob.prep_job_star(
        in_mics=relion_project.tomograms_star,
        in_parts=relion_project.particles_star,
        **network,
    )
    job_star.to_star_dict().write(relion_project.job_file("job.star"))
    run_as_relion(job_star, f"{relion_project.job_dir.as_posix()}/")
    fake_cryodrgn.calls.clear()

    _continue_job(relion_project, n_epochs=30)

    vae = fake_cryodrgn.parsed("train_vae")
    assert Path(vae.load) == relion_project.job_file("train", "weights.20.pkl")
    assert {name: getattr(vae, name) for name in network} == network
    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()


def test_data_cannot_be_changed_when_continuing():
    names = ["in_mics", "in_parts", "box_size", "lazy", "n_epochs", "dose_per_tilt"]
    names += ["zdim", "enc_layers", "enc_dim", "dec_layers", "dec_dim"]
    widgets = {name: SimpleNamespace(enabled=True) for name in names}

    CryoDrgnTrainContinue.setup_widgets(widgets)

    enabled = {name for name, w in widgets.items() if w.enabled}
    assert enabled == {"n_epochs"}


def test_continue_is_suggested_from_train_job():
    from himena_relion._job_class import CONNECTED_JOBS

    assert (CryoDrgnTrainJob, CryoDrgnTrainContinue) in CONNECTED_JOBS
    assert CryoDrgnTrainContinue.job_title() == CryoDrgnTrainJob.job_title()
