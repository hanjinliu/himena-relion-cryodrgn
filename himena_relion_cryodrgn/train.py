from pathlib import Path
from typing import Annotated

from himena_relion_cryodrgn import _impl
from himena_relion_cryodrgn._job_base import CRYODRGN_BIN_DIR, _CryoDrgnJob


class CryoDrgnTrainJob(_CryoDrgnJob):
    @classmethod
    def job_title(cls):
        return "CryoDRGN Train"
        
    def output_nodes(self):
        return [
            ("weights.pkl", "CryodrgnWeights.pkl"),
        ]

    def provide_widget(self, job_dir):
        from himena_relion_cryodrgn.widgets import QCryoDrgnTrainViewer

        return QCryoDrgnTrainViewer(job_dir)

    def run(
        self,
        in_mics,
        in_parts,
        box_size: Annotated[
            int,
            {
                "label": "Box size (pix)",
                "min": 2,
                "tooltip": "Box size to which the particles will be downsampled.",
                "group": "Downsampling",
            },
        ] = 128,
        lazy: Annotated[
            bool, {"label": "Lazy-load backprojection", "group": "Downsampling"}
        ] = False,
        n_epochs: Annotated[
            int,
            {
                "label": "Number of epochs",
                "group": "Training",
            },
        ] = 20,
        dose_per_tilt: Annotated[
            float, {"label": "Dose per tilt (e/A^2)", "group": "Training"}
        ] = 3.0,
        cryodrgn_bin_dir: CRYODRGN_BIN_DIR = "",
    ):
        bin_dir = self._resolve_bin_dir({"cryodrgn_bin_dir": cryodrgn_bin_dir})
        image = _impl.read_particle_image_params(in_parts)
        if box_size > image.box_size:
            raise ValueError(
                f"box_size ({box_size}) must not be larger than the box size of the "
                f"extracted particles ({image.box_size})."
            )
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

        if box_size < image.box_size:
            result_downsample = _impl.cryodrgn_downsample(
                input_file=result.particles_file,
                output_dir=job_dir / "downsample",
                px_binned=box_size,
                bin_dir=bin_dir,
            )
            particles_star = result_downsample.particles_file
            yield
        else:
            particles_star = result.particles_file

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
            indices_file=result_backproject.indices_file,
            bin_dir=bin_dir,
        )
        yield

        # Expose the final weights as the output node. Use a relative link so that the
        # project directory can be moved, and replace a link left by a previous run.
        link = job_dir / "weights.pkl"
        link.unlink(missing_ok=True)
        link.symlink_to(Path("train", "weights.pkl"))
