from pathlib import Path
from typing import Annotated

from himena_relion.external import RelionExternalContinue

from himena_relion_cryodrgn import _analysis, _impl
from himena_relion_cryodrgn._job_base import CRYODRGN_BIN_DIR, _CryoDrgnJob

BOX_SIZE = Annotated[
    int,
    {
        "label": "Box size (pix)",
        "min": 2,
        "tooltip": (
            "Box size to which the particles will be downsampled (even number). The "
            "particles are not downsampled if it is not smaller than the extracted "
            "box size."
        ),
        "group": "Downsampling",
    },
]
LAZY = Annotated[bool, {"label": "Lazy-load backprojection", "group": "Downsampling"}]
N_EPOCHS = Annotated[
    int,
    {
        "label": "Number of epochs",
        "min": 1,
        "tooltip": (
            "Total number of epochs. To train longer, continue the job with a larger "
            "number."
        ),
        "group": "Training",
    },
]
DOSE_PER_TILT = Annotated[
    float, {"label": "Dose per tilt (e/A^2)", "group": "Training"}
]
# The network (cryodrgn train_vae). It cannot be changed when continuing the training.
ZDIM = Annotated[
    int,
    {
        "label": "Latent dimension",
        "min": 1,
        "tooltip": "Dimension of the latent space (--zdim).",
        "group": "Network",
    },
]
ENC_LAYERS = Annotated[
    int,
    {
        "label": "Encoder layers",
        "min": 1,
        "tooltip": "Number of hidden layers of the encoder (--enc-layers).",
        "group": "Network",
    },
]
ENC_DIM = Annotated[
    int,
    {
        "label": "Encoder dimension",
        "min": 1,
        "max": 16384,
        "tooltip": "Number of nodes in the hidden layers of the encoder (--enc-dim).",
        "group": "Network",
    },
]
DEC_LAYERS = Annotated[
    int,
    {
        "label": "Decoder layers",
        "min": 1,
        "tooltip": "Number of hidden layers of the decoder (--dec-layers).",
        "group": "Network",
    },
]
DEC_DIM = Annotated[
    int,
    {
        "label": "Decoder dimension",
        "min": 1,
        "max": 16384,
        "tooltip": "Number of nodes in the hidden layers of the decoder (--dec-dim).",
        "group": "Network",
    },
]
_DEFAULT_ARCH = _impl.VaeArchitecture()

# Parameters of the Train job that define the data and the network: they cannot be
# changed when the job is continued.
FIXED_PARAMS = (
    "box_size", "lazy", "dose_per_tilt",
    "zdim", "enc_layers", "enc_dim", "dec_layers", "dec_dim",
)  # fmt: skip


def _training_particles_star(job_dir: Path, in_parts, box_size: int) -> Path:
    """The 2D particle star file to train on (downsampled only if smaller)."""
    image = _impl.read_particle_image_params(in_parts)
    if box_size < image.box_size:
        return job_dir / "downsample" / "particles_downsampled.star"
    return job_dir / _impl.PARSED_PARTICLES


def _link(job_dir: Path, name: str, target: Path) -> None:
    """Make the output node `job_dir/name` a link to `target` in the job directory.

    The link is relative so that the project directory can be moved, and a link
    left by a previous run is replaced.
    """
    link = job_dir / name
    link.unlink(missing_ok=True)
    link.symlink_to(target.relative_to(job_dir))


def _link_outputs(job_dir: Path, particles_star: Path, in_parts) -> None:
    """Link the output nodes to the files of the training.

    - weights.pkl -> train/weights.pkl
    - particles.star: a copy of the input RELION particles (in_parts), passed to
      CryoDRGN Select. A copy, so that the selection keeps referring to the rows
      trained on even if the upstream job is run again.
    """
    _link(job_dir, "weights.pkl", job_dir / "train" / "weights.pkl")
    (job_dir / "particles.star").write_bytes(Path(in_parts).read_bytes())


class CryoDrgnTrainJob(_CryoDrgnJob):
    @classmethod
    def job_title(cls):
        return "CryoDRGN Train"

    def output_nodes(self):
        return [
            ("weights.pkl", "CryodrgnWeights.pkl"),
            ("particles.star", "ParticleGroupMetadata.star"),
            # back-projection of all the training particles, e.g. a 3D reference
            (_impl.BACKPROJECTION, "DensityMap.mrc"),
        ]

    def provide_widget(self, job_dir):
        from himena_relion_cryodrgn.widgets import QCryoDrgnTrainViewer

        return QCryoDrgnTrainViewer(job_dir)

    def run(
        self,
        in_mics,
        in_parts,
        box_size: BOX_SIZE = 128,
        lazy: LAZY = False,
        n_epochs: N_EPOCHS = 20,
        dose_per_tilt: DOSE_PER_TILT = 3.0,
        zdim: ZDIM = _DEFAULT_ARCH.zdim,
        enc_layers: ENC_LAYERS = _DEFAULT_ARCH.enc_layers,
        enc_dim: ENC_DIM = _DEFAULT_ARCH.enc_dim,
        dec_layers: DEC_LAYERS = _DEFAULT_ARCH.dec_layers,
        dec_dim: DEC_DIM = _DEFAULT_ARCH.dec_dim,
        cryodrgn_bin_dir: CRYODRGN_BIN_DIR = "",
    ):
        bin_dir = self._resolve_bin_dir({"cryodrgn_bin_dir": cryodrgn_bin_dir})
        architecture = _impl.VaeArchitecture(
            zdim, enc_layers, enc_dim, dec_layers, dec_dim
        )
        image = _impl.read_particle_image_params(in_parts)
        if box_size > image.box_size:
            raise ValueError(
                f"box_size ({box_size}) must not be larger than the box size of the "
                f"extracted particles ({image.box_size})."
            )
        downsample = box_size < image.box_size
        if downsample and box_size % 2 != 0:
            # check before preprocessing: cryodrgn downsample would fail at the end
            raise ValueError(f"box_size ({box_size}) must be an even number.")
        job_dir = self.output_job_dir.path
        result = _impl.cryodrgn_parse_relion(
            in_mics, in_parts, job_dir, bin_dir=bin_dir
        )

        _impl.cryodrgn_parse_star(
            result.particles_file,
            ctffile=result.ctf_file,
            posesfile=result.poses_file,
            apix=image.pixel_size,
            box_size=image.box_size,
            bin_dir=bin_dir,
        )
        yield

        particles_star = _training_particles_star(job_dir, in_parts, box_size)
        if downsample:
            _impl.cryodrgn_downsample(
                input_file=result.particles_file,
                output_dir=particles_star.parent,
                px_binned=box_size,
                bin_dir=bin_dir,
            )
            yield

        result_backproject = _impl.cryodrgn_backproject_voxel(
            particles_star=particles_star,
            ctf_file=result.ctf_file,
            poses_file=result.poses_file,
            output_dir=job_dir / "backproject",
            dose_per_tilt=dose_per_tilt,
            lazy=lazy,
            bin_dir=bin_dir,
        )
        yield
        _impl.cryodrgn_train_vae(
            particles_star=particles_star,
            ctf_file=result.ctf_file,
            poses_file=result.poses_file,
            output_dir=job_dir / "train",
            dose_per_tilt=dose_per_tilt,
            n_epochs=n_epochs,
            architecture=architecture,
            indices_file=result_backproject.indices_file,
            bin_dir=bin_dir,
        )
        yield
        _link_outputs(job_dir, particles_star, in_parts)


class CryoDrgnTrainContinue(_CryoDrgnJob, RelionExternalContinue):
    """Train the cryoDRGN model for more epochs, starting from the last checkpoint.

    The cryoDRGN docs recommend training 25 epochs first and then doubling the number
    of epochs to check the convergence. Set the total number of epochs larger than
    the epochs already trained. The data (box size, dose) and the network cannot be
    changed: the checkpoint is loaded into the same network.
    """

    original_class = CryoDrgnTrainJob

    @classmethod
    def setup_widgets(cls, widgets):
        super().setup_widgets(widgets)  # the input nodes
        for name in FIXED_PARAMS:
            if (widget := widgets.get(name)) is not None:
                widget.enabled = False

    def run(
        self,
        in_mics,
        in_parts,
        box_size: BOX_SIZE = 128,
        lazy: LAZY = False,
        n_epochs: N_EPOCHS = 20,
        dose_per_tilt: DOSE_PER_TILT = 3.0,
        zdim: ZDIM = _DEFAULT_ARCH.zdim,
        enc_layers: ENC_LAYERS = _DEFAULT_ARCH.enc_layers,
        enc_dim: ENC_DIM = _DEFAULT_ARCH.enc_dim,
        dec_layers: DEC_LAYERS = _DEFAULT_ARCH.dec_layers,
        dec_dim: DEC_DIM = _DEFAULT_ARCH.dec_dim,
        cryodrgn_bin_dir: CRYODRGN_BIN_DIR = "",
    ):
        bin_dir = self._resolve_bin_dir({"cryodrgn_bin_dir": cryodrgn_bin_dir})
        # the network of the checkpoint (from job.star of the first run)
        architecture = _impl.VaeArchitecture(
            zdim, enc_layers, enc_dim, dec_layers, dec_dim
        )
        job_dir = self.output_job_dir.path
        workdir = job_dir / "train"
        last_epoch = _analysis.last_checkpoint_epoch(workdir)
        if last_epoch is None:
            raise RuntimeError(
                "No checkpoint (train/weights.{epoch}.pkl) to continue from. Run the "
                "job again from the beginning."
            )
        if n_epochs <= last_epoch:
            raise ValueError(
                f"The number of epochs ({n_epochs}) must be larger than the epochs "
                f"already trained ({last_epoch})."
            )
        self.console.log(f"Continue training from epoch {last_epoch} to {n_epochs}.")
        # the same particles as the first run (backproject_voxel --force-ntilts)
        indices = (
            job_dir
            / "backproject"
            / (f"indices_force_ntilts_{_impl.DEFAULT_NTILTS}.pkl")
        )
        particles_star = _training_particles_star(job_dir, in_parts, box_size)
        _impl.cryodrgn_train_vae(
            particles_star=particles_star,
            ctf_file=job_dir / "ctf.pkl",
            poses_file=job_dir / "poses.pkl",
            output_dir=workdir,
            dose_per_tilt=dose_per_tilt,
            n_epochs=n_epochs,
            architecture=architecture,
            indices_file=str(indices) if indices.is_file() else None,
            load=workdir / f"weights.{last_epoch}.pkl",
            bin_dir=bin_dir,
        )
        yield
        _link_outputs(job_dir, particles_star, in_parts)
