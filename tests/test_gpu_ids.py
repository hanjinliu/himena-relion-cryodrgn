"""GPUs to run cryoDRGN on ("GPU IDs to use", as in RELION).

The jobs set CUDA_VISIBLE_DEVICES of the cryoDRGN commands, and train_vae is given
`--multigpu` with more than one GPU. The IDs number the GPUs visible to the job, so
that GPUs assigned by a queueing system (CUDA_VISIBLE_DEVICES) are respected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion.external import run_function
from relion_command import run_as_relion

from himena_relion_cryodrgn import _impl
from himena_relion_cryodrgn.graph_traversal import CryoDrgnTrainGraphTraversal
from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainContinue, CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject

CUDA = "CUDA_VISIBLE_DEVICES"
JOBS = [
    CryoDrgnTrainJob,
    CryoDrgnTrainContinue,
    CryoDrgnSelectParticlesJob,
    CryoDrgnTrainGraphTraversal,
]


@pytest.fixture(autouse=True)
def _no_gpus_assigned(monkeypatch: pytest.MonkeyPatch):
    """CUDA_VISIBLE_DEVICES is not set, unless a test sets it like a queueing system."""
    monkeypatch.delenv(CUDA, raising=False)


def _cuda_devices(fake: FakeCryoDRGN) -> dict[str, str | None]:
    """CUDA_VISIBLE_DEVICES set for each subcommand (None: inherited)."""
    return {
        cmd[1]: None if env is None else env.get(CUDA)
        for cmd, env in zip(fake.calls, fake.envs)
    }


def _run_train(project: RelionProject, **params) -> None:
    argv = [
        CryoDrgnTrainJob.import_path(),
        "--o", f"{project.job_dir.as_posix()}/",
        "--in_mics", project.tomograms_star,
        "--in_parts", project.particles_star,
    ]  # fmt: skip
    for key, value in params.items():
        argv += [f"--{key}", str(value)]
    run_function(argv)


def _train_as_relion(project: RelionProject, **params) -> list[str]:
    """Create the Train job with `params` and run it as RELION does."""
    job_star = CryoDrgnTrainJob.prep_job_star(
        in_mics=project.tomograms_star, in_parts=project.particles_star, **params
    )
    job_star.to_star_dict().write(project.job_file("job.star"))
    return run_as_relion(job_star, f"{project.job_dir.as_posix()}/")


def _continue(project: RelionProject, job_cls, **params) -> list[str]:
    """Continue the Train job with `job_cls`, as RELION runs it."""
    job_dir = ExternalJobDirectory(project.root / project.job_dir)
    job_star = job_cls(job_dir).make_job_star(**params)
    job_star.to_star_dict().write(job_dir.job_star())
    for name in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE"):
        job_dir.path.joinpath(name).unlink(missing_ok=True)
    return run_as_relion(job_star, f"{project.job_dir.as_posix()}/")


# --- parsing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("", []),
        ("0", [0]),
        ("10", [10]),
        ("0,1", [0, 1]),
        ("3,1", [3, 1]),
        ("0:1", [0, 1]),  # RELION's separator of MPI ranks
        (" 0, 2 ", [0, 2]),
        ("1,", [1]),  # allowed by the input field while typing
    ],
)
def test_parse_gpu_ids(text: str, ids: list[int]):
    assert _impl.parse_gpu_ids(text) == ids


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("a", "numbers"),
        ("-1", "numbers"),
        ("0.5", "numbers"),
        ("0 1", "numbers"),
        ("True", "numbers"),
        ("0,0", "twice"),
    ],
)
def test_parse_gpu_ids_invalid(text: str, match: str):
    with pytest.raises(ValueError, match=match):
        _impl.parse_gpu_ids(text)


@pytest.mark.parametrize(
    ("ids", "visible", "expected"),
    [
        ([1, 3], None, "1,3"),
        ([0], "2,3", "2"),
        ([1, 0], "2,3", "3,2"),
        ([1], "GPU-aaaa, GPU-bbbb", "GPU-bbbb"),  # devices given by UUID
    ],
)
def test_cuda_visible_devices(ids: list[int], visible: str | None, expected: str):
    assert _impl.cuda_visible_devices(ids, visible) == expected


@pytest.mark.parametrize("visible", ["2", ""])
def test_cuda_visible_devices_unavailable(visible: str):
    with pytest.raises(ValueError, match="GPU ID 1 is not available"):
        _impl.cuda_visible_devices([1], visible)


@pytest.mark.parametrize("job_cls", JOBS)
def test_prerun_check_rejects_invalid_gpu_ids(job_cls):
    with pytest.raises(ValueError, match="twice"):
        job_cls.prerun_check(gpu_ids="0,0")


@pytest.mark.parametrize("job_cls", JOBS)
def test_gpu_ids_parameter(job_cls):
    params = job_cls._signature().parameters
    assert params["gpu_ids"].default == ""


# --- jobs --------------------------------------------------------------------------


def test_train_on_default_gpu(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_train(relion_project)

    assert fake_cryodrgn.envs == [None] * 5
    assert not fake_cryodrgn.parsed("train_vae").multigpu


def test_train_on_one_gpu(relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    _run_train(relion_project, gpu_ids="1")

    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert _cuda_devices(fake_cryodrgn) == {
        "parse_relion": None,
        "parse_star": None,
        "downsample": None,
        "backproject_voxel": "1",
        "train_vae": "1",
    }
    assert not fake_cryodrgn.parsed("train_vae").multigpu


def test_train_on_multiple_gpus(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _run_train(relion_project, gpu_ids="0,2")

    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    devices = _cuda_devices(fake_cryodrgn)
    assert devices["backproject_voxel"] == devices["train_vae"] == "0,2"
    assert fake_cryodrgn.parsed("train_vae").multigpu


def test_gpu_ids_from_relion_command_line(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    argv = _train_as_relion(relion_project, gpu_ids="0,1")

    assert argv[argv.index("--gpu_ids") + 1] == "0,1"
    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert _cuda_devices(fake_cryodrgn)["train_vae"] == "0,1"
    assert fake_cryodrgn.parsed("train_vae").multigpu


def test_gpus_assigned_by_queueing_system(
    relion_project: RelionProject,
    fake_cryodrgn: FakeCryoDRGN,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The IDs number the GPUs visible to the job, as in RELION."""
    monkeypatch.setenv(CUDA, "4,5,6")

    _run_train(relion_project, gpu_ids="2")

    devices = _cuda_devices(fake_cryodrgn)
    assert devices["backproject_voxel"] == devices["train_vae"] == "6"
    assert "(CUDA_VISIBLE_DEVICES=6)" in capsys.readouterr().out


def test_unavailable_gpu_fails_before_running(
    relion_project: RelionProject,
    fake_cryodrgn: FakeCryoDRGN,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(CUDA, "0")  # one GPU assigned to the job

    with pytest.raises(ValueError, match="GPU ID 1 is not available"):
        _run_train(relion_project, gpu_ids="1")

    assert fake_cryodrgn.calls == []
    assert relion_project.job_file("RELION_JOB_EXIT_FAILURE").exists()


def test_continue_on_other_gpus(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """The GPUs can be changed when continuing the training."""
    _continue(trained_project, CryoDrgnTrainContinue, n_epochs=30, gpu_ids="0,1")

    assert trained_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert _cuda_devices(fake_cryodrgn) == {"train_vae": "0,1"}
    vae = fake_cryodrgn.parsed("train_vae")
    assert vae.multigpu
    assert vae.load is not None


def test_continue_keeps_gpus(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """The GPUs of the first run are used again if not changed."""
    _train_as_relion(relion_project, gpu_ids="1")
    fake_cryodrgn.calls.clear()
    fake_cryodrgn.envs.clear()

    _continue(relion_project, CryoDrgnTrainContinue, n_epochs=30)

    assert _cuda_devices(fake_cryodrgn) == {"train_vae": "1"}
    assert not fake_cryodrgn.parsed("train_vae").multigpu


def test_graph_traversal_on_gpu(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _continue(trained_project, CryoDrgnTrainGraphTraversal, gpu_ids="1")

    assert trained_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert _cuda_devices(fake_cryodrgn) == {"graph_traversal": "1", "eval_vol": "1"}


def test_select_backprojects_on_gpu(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    fake_cryodrgn.filter_selection = [1, 2]

    run_function(
        [
            CryoDrgnSelectParticlesJob.import_path(),
            "--o", f"{trained_project.select_job_dir.as_posix()}/",
            "--train_weights", f"{trained_project.job_dir.as_posix()}/weights.pkl",
            "--gpu_ids", "1",
        ]
    )  # fmt: skip

    assert trained_project.select_job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert _cuda_devices(fake_cryodrgn) == {
        "filter": None,
        "filter_star": None,
        "backproject_voxel": "1",
    }
