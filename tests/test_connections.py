"""Jobs suggested from a job window, with inputs filled in (`connect_jobs`).

The mappings given to `connect_jobs` are recorded, and the context function that
himena-relion builds from them is called like the suggestion does (with the job.star
of the job window). The filled-in parameters are checked.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena.workflow import LocalReaderMethod
from himena_relion import _job_class
from himena_relion._job_class import CONNECTED_JOBS, _node_mapping_to_context
from himena_relion.external import run_function
from himena_relion.relion5 import _builtins as _spa
from himena_relion.relion5_tomo import _builtins as _tomo

from himena_relion_cryodrgn import _connections
from himena_relion_cryodrgn.graph_traversal import CryoDrgnTrainGraphTraversal
from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainContinue, CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject

TOMOGRAMS = "Tomograms/job005/tomograms.star"
TRAJECTORIES = "Polish/job020/motion.star"


@pytest.fixture(scope="module")
def connections() -> dict[tuple[type, type], dict]:
    """The keyword arguments of each `connect_jobs(pre, post, ...)` call."""
    recorded: dict[tuple[type, type], dict] = {}
    connect_jobs = _job_class.connect_jobs
    _job_class.connect_jobs = lambda pre, post, **kwargs: recorded.update(
        {(pre, post): kwargs}
    )
    try:
        importlib.reload(_connections)
    finally:
        _job_class.connect_jobs = connect_jobs
    return recorded


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


def _suggested(connections, pre_job: Path, pre: type, post: type) -> dict:
    """Parameters filled in when `post` is suggested from the window of `pre_job`."""
    context = _node_mapping_to_context(**connections[(pre, post)])
    step = LocalReaderMethod(
        path=pre_job / "job.star", plugin="himena_relion.io.read_relion_job"
    )
    return context(None, step)  # type: ignore[arg-type]


def test_connections_are_registered(connections):
    import himena_relion_cryodrgn.main  # noqa: F401  (the himena.plugin entry point)

    expected = {
        (_tomo.ExtractParticlesTomoJob, CryoDrgnTrainJob),
        (CryoDrgnTrainJob, CryoDrgnSelectParticlesJob),
        (CryoDrgnTrainJob, _spa.SelectClassesInteractiveJob),
        (CryoDrgnSelectParticlesJob, CryoDrgnTrainJob),
        (CryoDrgnSelectParticlesJob, _tomo.ExtractParticlesTomoJob),
    }
    assert set(connections) == expected
    continues = {
        (CryoDrgnTrainJob, CryoDrgnTrainContinue),
        (CryoDrgnTrainJob, CryoDrgnTrainGraphTraversal),
    }
    for pair in expected | continues:
        assert pair in CONNECTED_JOBS


def test_extract_to_train(connections, extract_job: Path):
    params = _suggested(
        connections, extract_job, _tomo.ExtractParticlesTomoJob, CryoDrgnTrainJob
    )
    assert params == {
        "in_parts": "Extract/job010/particles.star",
        "in_mics": TOMOGRAMS,
    }


def test_train_to_select(connections, relion_project: RelionProject):
    train_dir = relion_project.root / relion_project.job_dir
    CryoDrgnTrainJob.prep_job_star(
        in_mics=TOMOGRAMS,
        in_parts="Extract/job010/particles.star",
        dose_per_tilt=2.5,
        lazy=False,
    ).to_star_dict().write(train_dir / "job.star")

    params = _suggested(
        connections, train_dir, CryoDrgnTrainJob, CryoDrgnSelectParticlesJob
    )

    assert params == {
        "train_weights": "External/job011/weights.pkl",
        # the copy of the RELION particles trained on (not particles_2d.star)
        "in_parts": "External/job011/particles.star",
        "dose_per_tilt": 2.5,
        "lazy": False,  # "No" in job.star (bool("No") is True)
    }


def test_train_to_subset_selection(connections, relion_project: RelionProject):
    """The k-means clusters are selected in the interactive class selection."""
    params = _suggested(
        connections,
        relion_project.root / relion_project.job_dir,
        CryoDrgnTrainJob,
        _spa.SelectClassesInteractiveJob,
    )
    assert params == {"fn_model": "External/job011/kmeans_optimiser.star"}


def test_train_to_select_defaults(connections, relion_project: RelionProject):
    params = _suggested(
        connections,
        relion_project.root / relion_project.job_dir,
        CryoDrgnTrainJob,
        CryoDrgnSelectParticlesJob,
    )
    assert (params["dose_per_tilt"], params["lazy"]) == (3.0, False)


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


def test_select_to_train(connections, selected_project: RelionProject):
    params = _suggested(
        connections,
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        CryoDrgnTrainJob,
    )
    assert params == {
        "in_parts": "External/job012/particles.star",
        "in_mics": TOMOGRAMS,
        # same data: inherited from the previous training (defaults here)
        "dose_per_tilt": 3.0,
        "box_size": 128,
        "n_epochs": 20,
        # the same network
        "zdim": 8,
        "enc_layers": 3,
        "enc_dim": 1024,
        "dec_layers": 3,
        "dec_dim": 1024,
    }


def test_select_to_train_inherits_recorded_values(
    connections, selected_project: RelionProject
):
    CryoDrgnTrainJob.prep_job_star(
        in_mics=TOMOGRAMS,
        in_parts="Extract/job010/particles.star",
        box_size=64,
        dose_per_tilt=2.5,
        n_epochs=50,
        zdim=4,
        dec_dim=512,
    ).to_star_dict().write(selected_project.job_file("job.star"))

    params = _suggested(
        connections,
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        CryoDrgnTrainJob,
    )

    assert (params["dose_per_tilt"], params["box_size"], params["n_epochs"]) == (
        2.5,
        64,
        50,
    )
    assert (params["zdim"], params["enc_dim"], params["dec_dim"]) == (4, 1024, 512)


def test_select_to_extract(connections, selected_project: RelionProject):
    params = _suggested(
        connections,
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        _tomo.ExtractParticlesTomoJob,
    )
    assert params == {
        "in_optim": {
            "in_particles": "External/job012/particles.star",
            "in_tomograms": TOMOGRAMS,
            "in_trajectories": TRAJECTORIES,
        }
    }


def test_select_to_extract_without_trajectories(
    connections, selected_project: RelionProject, extract_job: Path
):
    _write_optimisation_set(
        extract_job / "optimisation_set.star", "Extract/job010/particles.star"
    )
    params = _suggested(
        connections,
        selected_project.root / selected_project.select_job_dir,
        CryoDrgnSelectParticlesJob,
        _tomo.ExtractParticlesTomoJob,
    )
    assert "in_trajectories" not in params["in_optim"]
