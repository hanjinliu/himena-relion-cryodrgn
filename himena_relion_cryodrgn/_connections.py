"""Suggest the next job from a job window, with its inputs filled in (`connect_jobs`).

    Extract subtomos ──> CryoDRGN Train ──> CryoDRGN Select Particles ──> Extract subtomos
                             ^   |                 |
                             |   └──> Subset selection (k-means clusters as classes)
                             └────── retrain ──────┘

"Continue - CryoDRGN Train" and "Graph Traversal - CryoDRGN Train" are suggested from
the Train job by RelionExternalContinue (train.py, graph_traversal.py).
Imported by the plugin entry point (main.py) only, i.e. in the himena GUI process.
"""

from __future__ import annotations

import logging
from pathlib import Path

from himena_relion._job_class import connect_jobs, parse_string
from himena_relion._job_dir import ExternalJobDirectory, JobDirectory
from himena_relion._utils import unwrap_annotated
from himena_relion.external import RelionExternalJob
from himena_relion.relion5 import _builtins as _spa
from himena_relion.relion5_tomo import _builtins as _tomo
from himena_relion.schemas import OptimisationSetModel

from himena_relion_cryodrgn import _kmeans_classes
from himena_relion_cryodrgn.select_particles import (
    CryoDrgnSelectParticlesJob,
    TrainJobFiles,
)
from himena_relion_cryodrgn.train import CryoDrgnTrainJob

_LOGGER = logging.getLogger(__name__)

# optimisation set written by Extract subtomos (optimisation_set.star) and by
# Refine3D / 3D classification (run_optimisation_set.star)
_OPTIMISATION_SETS = ("optimisation_set.star", "run_optimisation_set.star")


def _in_project(job: JobDirectory, path: str | Path | None) -> Path | None:
    """Absolute path of a path written relative to the RELION project."""
    if path is None or str(path) in ("", "."):
        return None
    path = Path(path)
    return path if path.is_absolute() else job.relion_project_dir / path


def _optimisation_set(job: JobDirectory) -> OptimisationSetModel | None:
    """Optimisation set produced by or given to the job."""
    for name in _OPTIMISATION_SETS:
        if (path := job.path / name).is_file():
            return OptimisationSetModel.validate_file(path)
    if path := _in_project(job, job.get_job_param("in_optimisation")):
        return OptimisationSetModel.validate_file(path)
    return None


def tomograms_of_job(path: Path) -> Path | None:
    """tomograms.star used by a RELION tomography job (Extract, Refine3D, ...)."""
    job = JobDirectory(path)
    if opt := _optimisation_set(job):
        return _in_project(job, opt.tomogram_star)
    return _in_project(job, job.get_job_param("in_tomograms"))


def _train_job(select_job_path: Path) -> TrainJobFiles:
    """The CryoDRGN Train job used by a CryoDRGN Select Particles job."""
    job = ExternalJobDirectory(select_job_path)
    return TrainJobFiles.from_weights(
        _in_project(job, job.get_job_param("train_weights"))
    )


def tomograms_of_train_job(path: Path) -> Path | None:
    job = ExternalJobDirectory(path)
    return _in_project(job, job.get_job_param("in_mic"))


def tomograms_of_select_job(path: Path) -> Path | None:
    return tomograms_of_train_job(_train_job(path).job_path)


def _upstream_extract_job(select_job_path: Path) -> Path:
    """Extract subtomos job that wrote the particles the cryoDRGN model was trained on."""
    extract_path = _train_job(select_job_path).particles_star.parent
    if not extract_path.joinpath("job.star").is_file():
        raise ValueError(f"{extract_path} is not a job directory.")
    return extract_path


def trajectories_of_select_job(path: Path) -> Path | None:
    extract = JobDirectory(_upstream_extract_job(path))
    if (opt := _optimisation_set(extract)) and opt.trajectories_star is not None:
        return _in_project(extract, opt.trajectories_star)
    return None


def _param_of(job_cls: type[RelionExternalJob], name: str):
    """A parameter of the job of the window (`job_cls`), as a value of its type.

    If the parameter is not in job.star, the job ran with the default value.
    """

    def _inner(path: Path):
        param = job_cls._signature().parameters[name]
        value = ExternalJobDirectory(path).get_job_param(name)
        if value == "":
            return None if param.default is param.empty else param.default
        # e.g. "No" -> False (bool("No") would be True)
        return parse_string(value, unwrap_annotated(param.annotation))

    _inner.__name__ = f"{job_cls.__name__}_{name}"
    return _inner


def _train_param_of_select_job(name: str):
    """A parameter of the Train job used by the CryoDRGN Select Particles job."""

    def _inner(path: Path):
        return _train_job(path).job_param(name)

    _inner.__name__ = f"train_{name}"
    return _inner


connect_jobs(
    _tomo.ExtractParticlesTomoJob,
    CryoDrgnTrainJob,
    node_mapping={"particles.star": "in_parts", tomograms_of_job: "in_mics"},
)
connect_jobs(
    CryoDrgnTrainJob,
    CryoDrgnSelectParticlesJob,
    node_mapping={
        "weights.pkl": "train_weights",
        # the RELION particles trained on (a copy in the Train job)
        "particles.star": "in_parts",
    },
    value_mapping={
        # the back-projection of the selected particles uses the same data
        _param_of(CryoDrgnTrainJob, "dose_per_tilt"): "dose_per_tilt",
        _param_of(CryoDrgnTrainJob, "lazy"): "lazy",
    },
)

# select k-means clusters in the window of the interactive class selection. Its
# particles.star goes to Extract subtomos etc. by the connections of himena-relion.
connect_jobs(
    CryoDrgnTrainJob,
    _spa.SelectClassesInteractiveJob,
    node_mapping={_kmeans_classes.OPTIMISER_STAR: "fn_model"},
)

# retrain on the selected particles
connect_jobs(
    CryoDrgnSelectParticlesJob,
    CryoDrgnTrainJob,
    node_mapping={
        "particles.star": "in_parts",
        tomograms_of_select_job: "in_mics",
    },
    value_mapping={
        _train_param_of_select_job("dose_per_tilt"): "dose_per_tilt",
        _train_param_of_select_job("box_size"): "box_size",
        _train_param_of_select_job("n_epochs"): "n_epochs",
        # the same network
        **{
            _train_param_of_select_job(name): name
            for name in ("zdim", "enc_layers", "enc_dim", "dec_layers", "dec_dim")
        },
    },
)

# back to RELION
connect_jobs(
    CryoDrgnSelectParticlesJob,
    _tomo.ExtractParticlesTomoJob,
    node_mapping={
        "particles.star": "in_optim.in_particles",
        tomograms_of_select_job: "in_optim.in_tomograms",
        trajectories_of_select_job: "in_optim.in_trajectories",
    },
)
