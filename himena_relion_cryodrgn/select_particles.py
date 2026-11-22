from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion._widgets._magicgui import PathDrop

from himena_relion_cryodrgn import _impl
from himena_relion_cryodrgn._job_base import CRYODRGN_BIN_DIR, _CryoDrgnJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob


@dataclass
class TrainJobFiles:
    """Files of a CryoDRGN Train job needed to select particles."""

    job_path: Path  # the CryoDRGN Train job directory
    workdir: Path  # cryoDRGN output directory (train_vae -o)
    particles_2d_star: Path  # output of cryodrgn_utils parse_relion
    particles_star: Path  # RELION particles.star the job was trained on
    training_indices: Path  # particles used for training (train_vae --ind)

    def training_particle_ids(self, n: int) -> np.ndarray:
        """cryoDRGN particle indices of the `n` particles in the training dataset.

        The latent space (z, UMAP) has one row per training particle. Training used
        only the particles with enough tilts (`backproject_voxel --force-ntilts`).
        """
        if self.training_indices.is_file():
            ids = _impl.load_indices(self.training_indices)
            if len(ids) == n:
                return ids
        return np.arange(n)

    @classmethod
    def from_weights(cls, weights_path) -> "TrainJobFiles":
        # Do not resolve the symlink: weights.pkl links to train/weights.pkl.
        weights_path = Path(weights_path).absolute()
        for job_path in weights_path.parents:
            if job_path.joinpath("job.star").exists():
                break
        else:
            raise ValueError(f"{weights_path} is not an output of a RELION job.")
        job_dir = ExternalJobDirectory(job_path)
        if job_dir.job_import_path() != CryoDrgnTrainJob.import_path():
            raise ValueError(f"{job_path} is not a {CryoDrgnTrainJob.job_title()} job.")
        ntilts = _impl.DEFAULT_NTILTS
        return cls(
            job_path=job_dir.path,
            workdir=job_dir.path / "train",
            particles_2d_star=job_dir.path / "particles_2d.star",
            particles_star=job_dir.resolve_path(job_dir.get_job_param("in_part")),
            training_indices=(
                job_dir.path / "backproject" / f"indices_force_ntilts_{ntilts}.pkl"
            ),
        )


class CryoDrgnSelectParticlesJob(_CryoDrgnJob):
    """Select particles in the latent space of cryoDRGN and write a particles.star.

    `cryodrgn filter` opens an interactive window: select particles with the lasso
    and close the window. This job therefore needs a display and cannot run through
    a queueing system.
    """

    @classmethod
    def job_title(cls):
        return "CryoDRGN Select Particles"

    def input_edges(self, **kwargs) -> list[str]:
        if weight := kwargs.get("train_weights"):
            return [weight]
        return []

    def output_nodes(self):
        return [
            ("particles.star", "ParticleGroupMetadata.star"),
        ]

    def provide_widget(self, job_dir):
        from himena_relion_cryodrgn.widgets import QCryoDrgnSelectViewer

        return QCryoDrgnSelectViewer(job_dir)

    def run(
        self,
        train_weights: Annotated[
            str,
            {
                "label": "CryoDRGN weights",
                "tooltip": "weights.pkl of a finished CryoDRGN Train job.",
                "widget_type": PathDrop,
                "type_label": "CryodrgnWeights",
                "allowed_extensions": [".pkl"],
                "group": "I/O",
            },
        ],
        epoch: Annotated[
            int | None,
            {
                "label": "Epoch",
                "min": 1,
                "tooltip": "Training epoch to select particles from. Use the last epoch by default.",
            },
        ] = None,
        kmeans: Annotated[
            int | None,
            {
                "label": "K-means",
                "min": 1,
                "tooltip": (
                    "k of the k-means result used to color the particles. Use the first one found by default "
                    "(k=20 after the automatic analysis of training)."
                ),
            },
        ] = None,
        cryodrgn_bin_dir: CRYODRGN_BIN_DIR = "",
    ):
        bin_dir = self._resolve_bin_dir({"cryodrgn_bin_dir": cryodrgn_bin_dir})
        train = TrainJobFiles.from_weights(train_weights)
        job_dir = self.output_job_dir.path
        self.console.log(
            "Select particles with the lasso in the cryoDRGN window, then close it."
        )
        selection = _impl.cryodrgn_filter(
            train.workdir, sel_dir=job_dir, epoch=epoch, kmeans=kmeans, bin_dir=bin_dir
        )
        yield

        rows = _impl.particle_indices_to_star_rows(
            train.particles_2d_star, _impl.load_indices(selection.indices_file)
        )
        self.console.log(f"{rows.size} particles selected.")
        rows_file = job_dir / "indices_relion.pkl"  # row indices of particles.star
        _impl.save_indices(rows_file, rows)
        output = job_dir / "particles.star"
        _impl.cryodrgn_filter_star(
            train.particles_star, rows_file, output, bin_dir=bin_dir
        )
        _impl.restore_star_blocks(train.particles_star, output)
