"""k-means clusters of a CryoDRGN Train job as the classes of a classification job.

The "Subset selection" job of RELION (relion.select.interactive, run by himena-relion
in its own window) selects the classes of a classification job from its
optimiser.star. The CryoDRGN Train job writes the k-means clusters of its latest
analysis (train/analyze.{epoch}/kmeans{K}, by `cryodrgn analyze`) in the same format,
so that the clusters can be selected there:

- kmeans_optimiser.star: the model and data STAR files (optimiser_general).
- kmeans_model.star: one class per cluster (model_classes). The reference is the
  volume at the cluster center, vol_{label + 1:03d}.mrc: the decoder output at the
  particle nearest to the k-means centroid, not an average of the cluster.
- kmeans_data.star: the RELION particles trained on, with rlnClassNumber =
  label + 1. Particles not used for training (too few tilts) are in no class.

The class numbers are the numbers of the volumes. Paths in the STAR files are relative
to the RELION project directory, as RELION writes them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from starfile_rs import LoopDataBlock, as_star, read_star
from starfile_rs.core import StarDict

from himena_relion_cryodrgn import _analysis, _impl

if TYPE_CHECKING:
    from himena_relion_cryodrgn.select_particles import TrainJobFiles

OPTIMISER_STAR = "kmeans_optimiser.star"
# relion_display reads <root>_data.star of <root>_optimiser.star
MODEL_STAR = "kmeans_model.star"
DATA_STAR = "kmeans_data.star"
_CLASS_NUMBER = "rlnClassNumber"
_STAR_HEADER = "\n# version 50001\n"


@dataclass
class KmeansResult:
    """k-means clustering of the latent space by `cryodrgn analyze`."""

    epoch: int
    k: int
    directory: Path  # train/analyze.{epoch}/kmeans{k}
    labels: np.ndarray  # cluster of each training particle (row of z.{epoch}.pkl)

    def volumes(self) -> list[Path]:
        # label 0 is vol_001.mrc (eval_vol --vol-start-index 1, the default)
        return [self.directory / f"vol_{i:03d}.mrc" for i in range(1, self.k + 1)]


def latest_kmeans(workdir) -> KmeansResult | None:
    """k-means result of the latest analysis (the smallest k if there are several)."""
    analyze_dirs = _analysis._numbered_files(workdir, _analysis._ANALYZE_DIR)
    for epoch in sorted(analyze_dirs, reverse=True):
        kmeans_dirs = _analysis._numbered_files(
            analyze_dirs[epoch], _analysis._KMEANS_DIR
        )
        for k in sorted(kmeans_dirs):
            labels = _analysis.load_kmeans_labels(workdir, epoch, k)
            if labels is not None:
                return KmeansResult(epoch, k, kmeans_dirs[k], labels)
    return None


def write_kmeans_classes(train: TrainJobFiles, project_dir) -> KmeansResult | None:
    """Write the k-means clusters of the Train job as classes (see the module doc).

    The files of a previous run are removed first. Nothing is written and None is
    returned if there is no k-means result with all its volumes.
    """
    job_path = Path(train.job_path)
    for name in (OPTIMISER_STAR, MODEL_STAR, DATA_STAR):
        job_path.joinpath(name).unlink(missing_ok=True)
    result = latest_kmeans(train.workdir)
    if result is None or not all(path.is_file() for path in result.volumes()):
        return None
    labels = result.labels
    if labels.size == 0 or labels.min() < 0 or labels.max() >= result.k:
        raise ValueError(
            f"Labels of {result.directory} must be in [0, {result.k}), got "
            f"[{labels.min()}, {labels.max()}]."
        )

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, project_dir)).as_posix()

    # training particle (row of z) -> cryoDRGN particle -> row of particles.star
    particle_ids = train.training_particle_ids(labels.size)
    rows = _impl.particle_star_rows(train.particles_parsed_star)[particle_ids]
    star = read_star(train.input_particles())
    block_name = "particles" if "particles" in star else next(iter(star.keys()))
    block = star[block_name].trust_loop()
    # read as strings to write the values back as they are
    df = block.to_polars(string_columns=list(block.columns))
    class_numbers = np.zeros(df.height, dtype=np.int64)
    class_numbers[rows] = labels + 1
    df = df.with_columns(pl.Series(_CLASS_NUMBER, class_numbers)).filter(
        pl.col(_CLASS_NUMBER) > 0
    )
    data = StarDict.from_blocks(
        [
            LoopDataBlock.from_polars(name, df) if name == block_name else blk
            for name, blk in star.items()
        ]
    )
    counts = np.bincount(labels, minlength=result.k)
    model = as_star(
        {
            "model_general": {
                "rlnReferenceDimensionality": 3,
                "rlnNrClasses": result.k,
            },
            "model_classes": pl.DataFrame(
                {
                    "rlnReferenceImage": [relative(p) for p in result.volumes()],
                    "rlnClassDistribution": counts / counts.sum(),
                }
            ),
        }
    )
    optimiser = as_star(
        {
            "optimiser_general": {
                "rlnModelStarFile": relative(job_path / MODEL_STAR),
                "rlnExperimentalDataStarFile": relative(job_path / DATA_STAR),
            }
        }
    )
    _write_star(job_path / DATA_STAR, data)
    _write_star(job_path / MODEL_STAR, model)
    _write_star(job_path / OPTIMISER_STAR, optimiser)  # the output node, last
    return result


def _write_star(path: Path, star: StarDict) -> None:
    """Write a STAR file atomically, with the version line of RELION 5."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_STAR_HEADER + star.to_string(), newline="\n")
    tmp.replace(path)
