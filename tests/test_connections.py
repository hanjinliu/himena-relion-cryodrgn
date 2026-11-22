"""Jobs suggested from a job window, with inputs filled in (`connect_jobs`).

The context function that himena-relion builds from each mapping is called like
the suggestion does (with the job.star of the job window), and the filled-in
parameters are checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena.workflow import LocalReaderMethod
from himena_relion._job_class import CONNECTED_JOBS, _node_mapping_to_context
from himena_relion.external import run_function
from himena_relion.relion5_tomo import _builtins as _tomo
from himena_relion.relion5_tomo._connections import _subtomo_diameter_a

from himena_relion_cryodrgn import _connections
from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject

TOMOGRAMS = "Tomograms/job005/tomograms.star"
TRAJECTORIES = "Polish/job020/motion.star"


def _write_job_star(job_path: Path, job_type: str, params: dict[str, str]) -> None:
    lines = [
        "# version 50001\n\ndata_job\n",
        f"_rlnJobTypeLabel {job_type}",
        "_rlnJobIsContinue 0",
        "_rlnJobIsTomo 1\n\n",
        "# version 50001\n\ndata_joboptions_values\n",
        "loop_",
        "_rlnJobOptionVariable #1",
        "_rlnJobOptionValue #2",
    ]
    lines += [f"{key} {value or chr(34) * 2}" for key, value in params.items()]
    job_path.mkdir(parents=True, exist_ok=True)
    (job_path / "job.star").write_text("\n".join(lines) + "\n")


def _write_optimisation_set(path: Path, particles: str, trajectories: str = ""):
    text = (
        f"# version 50001\n\ndata_\n\n_rlnTomoParticlesFile {particles}\n"
        f"_rlnTomoTomogramsFile {TOMOGRAMS}\n"
    )
    if trajectories:
        text += f"_rlnTomoTrajectoriesFile {trajectories}\n"
    path.write_text(text)


@pytest.fixture
def extract_job(relion_project: RelionProject) -> Path:
    """Extract/job010, the Extract subtomos job that wrote particles.star."""
    path = relion_project.root / "Extract/job010"
    _write_job_star(
        path,
        "relion.pseudosubtomo",
        {
            "binning": "2",
            "box_size": "96",
            "crop_size": "64",
            "do_stack2d": "Yes",
            "in_optimisation": "",
            "in_particles": "Picks/job008/particles.star",
            "in_tomograms": TOMOGRAMS,
            "in_trajectories": "",
        },
    )
    _write_optimisation_set(
        path / "optimisation_set.star", "Extract/job010/particles.star", TRAJECTORIES
    )
    return path


def _suggested(pre_job: Path, pre: type, post: type) -> dict:
    """Parameters filled in when `post` is suggested from the window of `pre_job`."""
    (kwargs,) = [kw for p, q, kw in _connections.CONNECTIONS if (p, q) == (pre, post)]
    context = _node_mapping_to_context(**kwargs)
    step = LocalReaderMethod(
        path=pre_job / "job.star", plugin="himena_relion.io.read_relion_job"
    )
    return context(None, step)  # type: ignore[arg-type]


def test_all_connections_are_registered():
    import himena_relion_cryodrgn.main  # noqa: F401  (the himena.plugin entry point)

    for pre, post, _ in _connections.CONNECTIONS:
        assert (pre, post) in CONNECTED_JOBS


def test_extract_to_train(extract_job: Path):
    params = _suggested(extract_job, _tomo.ExtractParticlesTomoJob, CryoDrgnTrainJob)
    assert params == {
        "in_parts": "Extract/job010/particles.star",
        "in_mics": TOMOGRAMS,
    }


def test_refine3d_to_train(relion_project: RelionProject):
    path = relion_project.root / "Refine3D/job015"
    _write_job_star(
        path,
        "relion.refine3d.tomo",
        {"in_optimisation": "Extract/job010/optimisation_set.star"},
    )
    _write_optimisation_set(
        path / "run_optimisation_set.star", "Refine3D/job015/run_data.star"
    )

    params = _suggested(path, _tomo.Refine3DTomoJob, CryoDrgnTrainJob)

    assert params == {"in_parts": "Refine3D/job015/run_data.star", "in_mics": TOMOGRAMS}


def test_train_to_select(relion_project: RelionProject):
    params = _suggested(
        relion_project.root / relion_project.job_dir,
        CryoDrgnTrainJob,
        CryoDrgnSelectParticlesJob,
    )
    assert params == {"train_weights": "External/job011/weights.pkl"}


@pytest.fixture
def selected_project(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN, extract_job: Path
) -> RelionProject:
    """The project after CryoDRGN Train and CryoDRGN Select Particles."""
    CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights="External/job011/weights.pkl"
    ).to_star_dict().write(trained_project.select_job_file("job.star"))
    fake_cryodrgn.filter_selection = [0, 1]
    run_function(
        [
            CryoDrgnSelectParticlesJob.import_path(),
            "--o", f"{trained_project.select_job_dir.as_posix()}/",
            "--train_weights", "External/job011/weights.pkl",
        ]
    )  # fmt: skip
    return trained_project


def test_select_to_train(selected_project: RelionProject):
    params = _suggested(
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        CryoDrgnTrainJob,
    )
    assert params == {
        "in_parts": "External/job012/particles.star",
        "in_mics": TOMOGRAMS,
        # same data: inherited from the previous training
        "dose_per_tilt": 3.0,
        "box_size": 128,
        "n_epochs": 20,
    }


@pytest.mark.parametrize(
    "post",
    [
        _tomo.ReconstructParticlesJob,
        _tomo.Refine3DTomoJob,
        _tomo.Class3DTomoJob,
        _tomo.Class3DNoAlignmentTomoJob,
        _tomo.ExtractParticlesTomoJob,
    ],
)
def test_select_to_relion(selected_project: RelionProject, extract_job: Path, post):
    params = _suggested(
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        post,
    )
    assert params["in_optim"] == {
        "in_particles": "External/job012/particles.star",
        "in_tomograms": TOMOGRAMS,
        "in_trajectories": TRAJECTORIES,
    }
    if post is _tomo.ReconstructParticlesJob:
        assert (params["binning"], params["box_size"], params["crop_size"]) == (
            2,
            96,
            64,
        )
    elif post is not _tomo.ExtractParticlesTomoJob:
        assert params["particle_diameter"] == _subtomo_diameter_a(extract_job)


def test_select_to_relion_without_trajectories(
    selected_project: RelionProject, extract_job: Path
):
    _write_optimisation_set(
        extract_job / "optimisation_set.star", "Extract/job010/particles.star"
    )
    params = _suggested(
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        _tomo.Refine3DTomoJob,
    )
    assert "in_trajectories" not in params["in_optim"]


def test_select_to_train_inherits_recorded_values(selected_project: RelionProject):
    CryoDrgnTrainJob.prep_job_star(
        in_mics=TOMOGRAMS,
        in_parts="Extract/job010/particles.star",
        box_size=64,
        dose_per_tilt=2.5,
        n_epochs=50,
    ).to_star_dict().write(selected_project.job_file("job.star"))

    params = _suggested(
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        CryoDrgnTrainJob,
    )

    assert (params["dose_per_tilt"], params["box_size"], params["n_epochs"]) == (
        2.5,
        64,
        50,
    )
