"""cryoDRGN installed in another environment than himena-relion.

The directory of the cryoDRGN commands is set in the himena settings (CryoDRGN). It is
read when a job is created (in the GUI process) and recorded in job.star, because the
process that runs the job has no himena window to read the settings from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion.external import run_function

from himena_relion_cryodrgn import _configs
from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject

JOBS = [CryoDrgnTrainJob, CryoDrgnSelectParticlesJob]
JOB_KWARGS = {
    CryoDrgnTrainJob: {
        "in_mics": "Tomograms/job005/tomograms.star",
        "in_parts": "Extract/job010/particles.star",
    },
    CryoDrgnSelectParticlesJob: {"train_weights": "External/job011/weights.pkl"},
}


def _job_star_params(job_cls, **kwargs) -> dict[str, str]:
    model = job_cls.prep_job_star(**JOB_KWARGS[job_cls], **kwargs)
    opts = dict(zip(model.joboptions_values.variable, model.joboptions_values.value))
    return {
        opts[f"param{i}_label"]: opts[f"param{i}_value"]
        for i in range(1, 11)
        if opts[f"param{i}_label"]
    }


@pytest.fixture
def spaceless_bin_dir(monkeypatch: pytest.MonkeyPatch, cryodrgn_bin_dir: Path) -> str:
    """`cryodrgn_bin_dir` as a relative path, because tmp_path may contain spaces
    (e.g. a Windows user name) and spaces are rejected by `prerun_check`."""
    monkeypatch.chdir(cryodrgn_bin_dir.parents[2])
    return "envs/cryodrgn/bin"


@pytest.fixture
def configured_bin_dir(monkeypatch: pytest.MonkeyPatch, spaceless_bin_dir: str) -> str:
    """The settings point to the cryoDRGN bin directory."""
    monkeypatch.setattr(_configs, "get_bin_dir", lambda: spaceless_bin_dir)
    return spaceless_bin_dir


def test_get_bin_dir_without_himena_window():
    assert _configs.get_bin_dir() == ""


def test_get_bin_dir_from_himena_settings(himena_ui):
    """`himena_ui` (from himena's pytest plugin) uses a temporary user data dir."""
    from himena.plugins import update_config_context

    assert _configs.get_bin_dir() == ""
    with update_config_context(_configs.CryoDrgnConfig, _configs.PLUGIN_ID) as cfg:
        cfg.bin_dir = " ~/miniforge3/envs/cryodrgn/bin "

    assert _configs.get_bin_dir() == os.path.expanduser(
        "~/miniforge3/envs/cryodrgn/bin"
    )
    params = _job_star_params(CryoDrgnTrainJob)
    assert params["cryodrgn_bin_dir"] == _configs.get_bin_dir()


@pytest.mark.parametrize("job_cls", JOBS)
def test_configured_bin_dir_is_recorded(job_cls, configured_bin_dir: str):
    params = _job_star_params(job_cls)
    assert params["cryodrgn_bin_dir"] == configured_bin_dir


@pytest.mark.parametrize("job_cls", JOBS)
def test_explicit_bin_dir_overrides_settings(job_cls, configured_bin_dir: str):
    params = _job_star_params(job_cls, cryodrgn_bin_dir="/other/env/bin")
    assert params["cryodrgn_bin_dir"] == "/other/env/bin"


@pytest.mark.parametrize("job_cls", JOBS)
def test_empty_bin_dir_is_not_recorded(job_cls):
    params = _job_star_params(job_cls, cryodrgn_bin_dir="")
    assert "cryodrgn_bin_dir" not in params


@pytest.mark.parametrize("job_cls", JOBS)
def test_prerun_check_finds_commands(job_cls, spaceless_bin_dir: str):
    job_cls.prerun_check(cryodrgn_bin_dir=spaceless_bin_dir)


@pytest.mark.parametrize("job_cls", JOBS)
def test_prerun_check_uses_settings(job_cls, configured_bin_dir: str):
    job_cls.prerun_check(cryodrgn_bin_dir="")


@pytest.mark.parametrize("job_cls", JOBS)
def test_prerun_check_missing_commands(job_cls, spaceless_bin_dir: str):
    Path("empty_bin").mkdir()  # in the current directory, see spaceless_bin_dir
    with pytest.raises(ValueError, match="`cryodrgn` was not found"):
        job_cls.prerun_check(cryodrgn_bin_dir="empty_bin")


def test_prerun_check_searches_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cryodrgn_bin_dir: Path
):
    empty = tmp_path / "empty_bin"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ValueError, match="not found in PATH"):
        CryoDrgnTrainJob.prerun_check(cryodrgn_bin_dir="")

    monkeypatch.setenv("PATH", str(cryodrgn_bin_dir))
    CryoDrgnTrainJob.prerun_check(cryodrgn_bin_dir="")


def test_prerun_check_rejects_spaces():
    with pytest.raises(ValueError, match="spaces"):
        CryoDrgnTrainJob.prerun_check(cryodrgn_bin_dir="/my envs/cryodrgn/bin")


def _run_train(project: RelionProject, *extra: str) -> None:
    run_function(
        [
            CryoDrgnTrainJob.import_path(),
            "--o", f"{project.job_dir.as_posix()}/",
            "--in_mics", project.tomograms_star,
            "--in_parts", project.particles_star,
            *extra,
        ]
    )  # fmt: skip


def test_commands_run_from_bin_dir(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN, cryodrgn_bin_dir: Path
):
    _run_train(relion_project, "--cryodrgn_bin_dir", str(cryodrgn_bin_dir))

    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert len(fake_cryodrgn.calls) == 5
    for cmd, env in zip(fake_cryodrgn.calls, fake_cryodrgn.envs):
        assert Path(cmd[0]).parent == cryodrgn_bin_dir  # full path in the other env
        assert env is not None
        # so that cryoDRGN also finds its own tools without activating the env
        assert env["PATH"].split(os.pathsep)[0] == str(cryodrgn_bin_dir)


def test_commands_run_from_path_by_default(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_train(relion_project)

    assert [cmd[0] for cmd in fake_cryodrgn.calls] == [
        "cryodrgn_utils",
        "cryodrgn",
        "cryodrgn",
        "cryodrgn",
        "cryodrgn",
    ]
    assert fake_cryodrgn.envs == [None] * 5


def test_missing_bin_dir_fails_the_job(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN, tmp_path: Path
):
    with pytest.raises(FileNotFoundError, match="cryodrgn_utils"):
        _run_train(relion_project, "--cryodrgn_bin_dir", str(tmp_path / "no_such_dir"))

    assert fake_cryodrgn.calls == []
    assert relion_project.job_file("RELION_JOB_EXIT_FAILURE").exists()
