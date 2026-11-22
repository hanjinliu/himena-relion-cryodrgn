from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from himena_relion._job_class import parse_string
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion._utils import unwrap_annotated
from himena_relion._widgets._magicgui import PathDrop

from himena_relion_cryodrgn import _impl
from himena_relion_cryodrgn._job_base import CRYODRGN_BIN_DIR, _CryoDrgnJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob


@dataclass
class TrainJobFiles:
    """Files of a CryoDRGN Train job needed to select particles."""

    job_path: Path  # the CryoDRGN Train job directory
    workdir: Path  # cryoDRGN output directory (train_vae -o)
    particles_parsed_star: Path  # output of cryodrgn_utils parse_relion
    particles_star: Path  # RELION particles.star the job was trained on
    training_indices: Path  # particles used for training (train_vae --ind)

    @property
    def ctf_file(self) -> Path:
        return self.job_path / "ctf.pkl"

    @property
    def poses_file(self) -> Path:
        return self.job_path / "poses.pkl"

    def input_particles(self) -> Path:
        """RELION particles.star to select from: the copy in the Train job (its
        output node particles.star), or its input for a job run before the copy."""
        copy = self.job_path / "particles.star"
        return copy if copy.is_file() else self.particles_star

    def job_param(self, name: str) -> Any:
        """A parameter of the Train job, or its default if not in job.star."""
        param = CryoDrgnTrainJob._signature().parameters[name]
        value = ExternalJobDirectory(self.job_path).get_job_param(name)
        if value == "":
            return param.default
        return parse_string(value, unwrap_annotated(param.annotation))

    def training_box_size(self) -> int:
        """Box size of the images the model was trained on."""
        image = _impl.read_particle_image_params(self.particles_star)
        return min(int(self.job_param("box_size")), image.box_size)

    def training_pixel_size(self) -> float:
        """Pixel size (Å) of the images the model was trained on."""
        image = _impl.read_particle_image_params(self.particles_star)
        return image.pixel_size * image.box_size / self.training_box_size()

    def training_particles_star(self) -> Path:
        """The 2D particle star file the model was trained on.

        The Train job downsamples the images only if `box_size` is smaller than the
        extracted box size.
        """
        image = _impl.read_particle_image_params(self.particles_star)
        if self.training_box_size() < image.box_size:
            return self.job_path / "downsample" / "particles_downsampled.star"
        return self.particles_parsed_star

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
        # fn_exe of job.star is the one of the continue class once the job was
        # continued (Continue / Graph Traversal), which is still a Train job.
        job_cls = job_dir._to_job_class()
        if job_cls is not CryoDrgnTrainJob and (
            getattr(job_cls, "original_class", None) is not CryoDrgnTrainJob
        ):
            raise ValueError(f"{job_path} is not a {CryoDrgnTrainJob.job_title()} job.")
        ntilts = _impl.DEFAULT_NTILTS
        return cls(
            job_path=job_dir.path,
            workdir=job_dir.path / "train",
            particles_parsed_star=job_dir.path / _impl.PARSED_PARTICLES,
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
        return "CryoDRGN Select"

    def output_nodes(self):
        return [
            ("particles.star", "ParticleGroupMetadata.star"),
            # back-projection of the selected particles, e.g. a 3D reference
            (_impl.BACKPROJECTION, "DensityMap.mrc"),
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
        in_parts: Annotated[
            str,
            {
                "label": "Input particles",
                "tooltip": (
                    "RELION particles.star the cryoDRGN model was trained on, to "
                    "select particles from. If empty, particles.star of the "
                    "CryoDRGN Train job (a copy of its input) is used."
                ),
                "widget_type": PathDrop,
                "type_label": ["ParticlesData", "ParticleGroupMetadata"],
                "allowed_extensions": [".star"],
                "group": "I/O",
            },
        ] = "",
        epoch: Annotated[
            int,
            {
                "label": "Epoch",
                "min": -1,
                "tooltip": "Training epoch to select particles from. Use the last epoch by default.",
            },
        ] = -1,
        kmeans: Annotated[
            int,
            {
                "label": "K-means",
                "min": -1,
                "tooltip": (
                    "k of the k-means result used to color the particles. Use the first one found by default "
                    "(k=20 after the automatic analysis of training)."
                ),
            },
        ] = -1,
        lazy: Annotated[
            bool, {"label": "Lazy-load backprojection", "group": "Backprojection"}
        ] = False,
        dose_per_tilt: Annotated[
            float, {"label": "Dose per tilt (e/A^2)", "group": "Backprojection"}
        ] = 3.0,
        cryodrgn_bin_dir: CRYODRGN_BIN_DIR = "",
    ):
        bin_dir = self._resolve_bin_dir({"cryodrgn_bin_dir": cryodrgn_bin_dir})
        train = TrainJobFiles.from_weights(train_weights)
        particles_star = Path(in_parts) if in_parts else train.input_particles()
        job_dir = self.output_job_dir.path
        self.console.log(
            "Select particles with the lasso in the cryoDRGN window, then close it."
        )
        selection = _impl.cryodrgn_filter(
            train.workdir, sel_dir=job_dir, epoch=epoch, kmeans=kmeans, bin_dir=bin_dir
        )
        yield

        # cryoDRGN particle indices -> rows of particles.star, through the
        # rlnOriginalParticle column written by parse_relion
        rows = _impl.particle_indices_to_star_rows(
            train.particles_parsed_star, _impl.load_indices(selection.indices_file)
        )
        self.console.log(f"{rows.size} particles selected.")
        rows_file = job_dir / "indices_relion.pkl"  # row indices of particles.star
        _impl.save_indices(rows_file, rows)
        output = job_dir / "particles.star"
        _impl.cryodrgn_filter_star(particles_star, rows_file, output, bin_dir=bin_dir)
        _impl.restore_star_blocks(particles_star, output)
        yield

        # Validate the selection by back-projecting the selected particles, as
        # recommended in the cryoDRGN-ET docs. The images, CTF and poses are those of
        # the training (one row per tilt image); the selection refers to particles.
        _impl.cryodrgn_backproject_voxel(
            particles_star=train.training_particles_star(),
            ctf_file=train.ctf_file,
            poses_file=train.poses_file,
            output_dir=job_dir / "backproject",
            dose_per_tilt=dose_per_tilt,
            lazy=lazy,
            indices_file=selection.indices_file,
            force_ntilts=False,  # --force-ntilts would ignore the selection
            bin_dir=bin_dir,
        )
