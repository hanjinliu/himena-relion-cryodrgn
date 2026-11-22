"""Suggest the next job from a job window, with its inputs filled in (`connect_jobs`).

    Extract subtomos / Refine3D (tomo) ──> CryoDRGN Train ──> CryoDRGN Select Particles
                                              ^                   |
                                              └─── retrain ───────┤
    Reconstruct particle / Refine3D / 3D classification / Extract <┘

Imported by the plugin entry point (main.py) only, i.e. in the himena GUI process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from himena_relion._job_class import connect_jobs
from himena_relion._job_dir import ExternalJobDirectory, JobDirectory
from himena_relion.relion5_tomo import _builtins as _tomo
from himena_relion.relion5_tomo._connections import (
    _subtomo_binning,
    _subtomo_box_size,
    _subtomo_crop_size,
    _subtomo_diameter_a,
)
from himena_relion.schemas import OptimisationSetModel

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


def _from_upstream_extract(func):
    def _inner(path: Path):
        return func(_upstream_extract_job(path))

    _inner.__name__ = f"{func.__name__}_of_upstream_extract"
    return _inner


def _train_param(name: str, typ: type):
    """A parameter of the CryoDRGN Train job used by a CryoDRGN Select Particles job."""

    def _inner(path: Path):
        value = ExternalJobDirectory(_train_job(path).job_path).get_job_param(name)
        if value == "":  # not in job.star: the job ran with the default value
            param = CryoDrgnTrainJob._signature().parameters.get(name)
            if param is None or param.default is param.empty:
                return None
            return param.default
        return typ(value)

    _inner.__name__ = f"train_{name}"
    return _inner


# (pre, post, keyword arguments of `connect_jobs`)
CONNECTIONS: list[tuple[type, type, dict[str, Any]]] = [
    (
        _tomo.ExtractParticlesTomoJob,
        CryoDrgnTrainJob,
        {"node_mapping": {"particles.star": "in_parts", tomograms_of_job: "in_mics"}},
    ),
    (
        _tomo.Refine3DTomoJob,
        CryoDrgnTrainJob,
        {"node_mapping": {"run_data.star": "in_parts", tomograms_of_job: "in_mics"}},
    ),
    (
        CryoDrgnTrainJob,
        CryoDrgnSelectParticlesJob,
        {"node_mapping": {"weights.pkl": "train_weights"}},
    ),
    (
        # train again on the selected particles (train -> filter -> retrain)
        CryoDrgnSelectParticlesJob,
        CryoDrgnTrainJob,
        {
            "node_mapping": {
                "particles.star": "in_parts",
                tomograms_of_select_job: "in_mics",
            },
            "value_mapping": {
                _train_param("dose_per_tilt", float): "dose_per_tilt",
                _train_param("box_size", int): "box_size",
                _train_param("n_epochs", int): "n_epochs",
            },
        },
    ),
]

# back to RELION with the selected particles
_SELECTED_PARTICLES = {
    "particles.star": "in_optim.in_particles",
    tomograms_of_select_job: "in_optim.in_tomograms",
    trajectories_of_select_job: "in_optim.in_trajectories",
}
_DIAMETER = {_from_upstream_extract(_subtomo_diameter_a): "particle_diameter"}
CONNECTIONS += [
    (
        CryoDrgnSelectParticlesJob,
        _tomo.ReconstructParticlesJob,
        {
            "node_mapping": _SELECTED_PARTICLES,
            "value_mapping": {
                _from_upstream_extract(_subtomo_binning): "binning",
                _from_upstream_extract(_subtomo_box_size): "box_size",
                _from_upstream_extract(_subtomo_crop_size): "crop_size",
            },
        },
    ),
    (
        CryoDrgnSelectParticlesJob,
        _tomo.Refine3DTomoJob,
        {"node_mapping": _SELECTED_PARTICLES, "value_mapping": _DIAMETER},
    ),
    (
        CryoDrgnSelectParticlesJob,
        _tomo.Class3DTomoJob,
        {"node_mapping": _SELECTED_PARTICLES, "value_mapping": _DIAMETER},
    ),
    (
        CryoDrgnSelectParticlesJob,
        _tomo.Class3DNoAlignmentTomoJob,
        {"node_mapping": _SELECTED_PARTICLES, "value_mapping": _DIAMETER},
    ),
    (
        CryoDrgnSelectParticlesJob,
        _tomo.ExtractParticlesTomoJob,
        {"node_mapping": _SELECTED_PARTICLES},
    ),
]

for _pre, _post, _kwargs in CONNECTIONS:
    connect_jobs(_pre, _post, **_kwargs)
