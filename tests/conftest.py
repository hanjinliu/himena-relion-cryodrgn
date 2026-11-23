from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from fake_cryodrgn import FakeCryoDRGN

TILT_NX, TILT_NY = 96, 64  # tilt image width / height in pixels
TILT_SERIES_PIXEL_SIZE = 1.35
# extracted with binning 2, so that it differs from the tilt series pixel size
IMAGE_SIZE = 192
IMAGE_PIXEL_SIZE = 2.7

# Particles of Extract/job010/particles.star: (rlnTomoParticleName, visible tilts).
# cryodrgn_utils parse_relion skips TS_01/2 (no visible tilt), and
# backproject_voxel --force-ntilts (ntilts=10) removes TS_01/4 from training.
PARTICLES = [
    ("TS_01/1", 12),
    ("TS_01/2", 0),
    ("TS_01/3", 12),
    ("TS_01/4", 5),
    ("TS_01/5", 12),
    ("TS_01/6", 12),
]
N_TILTS = 12


def _optics_row(ctf_premultiplied: int) -> str:
    """The optics group of Extract/job010/particles.star.

    `relion_tomo_subtomo --stack2d` sets rlnCtfDataAreCtfPremultiplied to 0 only with
    `--no_ctf`, the extraction cryoDRGN needs.
    """
    return (
        f"1 opticsGroup1 {IMAGE_PIXEL_SIZE} {IMAGE_SIZE} 300.0 2.7 0.1 "
        f"{ctf_premultiplied}\n"
    )


def _job_pipeline(process: str, inputs: list[tuple[str, str]]) -> str:
    """job_pipeline.star of a running job, as written by RELION."""
    nodes = "\n".join(f"{path} {label}            1" for path, label in inputs)
    edges = "\n".join(f"{path} {process}" for path, _ in inputs)
    return (
        "# version 50001\n\ndata_pipeline_general\n\n"
        "_rlnPipeLineJobCounter                       13\n\n\n"
        "# version 50001\n\ndata_pipeline_processes\n\nloop_\n"
        "_rlnPipeLineProcessName #1\n_rlnPipeLineProcessAlias #2\n"
        "_rlnPipeLineProcessTypeLabel #3\n_rlnPipeLineProcessStatusLabel #4\n"
        f"{process}       None relion.external    Running\n\n\n"
        "# version 50001\n\ndata_pipeline_nodes\n\nloop_\n"
        "_rlnPipeLineNodeName #1\n_rlnPipeLineNodeTypeLabel #2\n"
        f"_rlnPipeLineNodeTypeLabelDepth #3\n{nodes}\n\n\n"
        "# version 50001\n\ndata_pipeline_input_edges\n\nloop_\n"
        f"_rlnPipeLineEdgeFromNode #1\n_rlnPipeLineEdgeProcess #2\n{edges}\n"
    )


@dataclass
class RelionProject:
    root: Path
    tomograms_star: str  # relative to root, as RELION passes it
    particles_star: str
    job_dir: Path  # CryoDRGN Train job, relative to root
    select_job_dir: Path  # CryoDRGN Select Particles job, relative to root
    tilt_dim: tuple[int, int] = (TILT_NX, TILT_NY)
    image_size: int = IMAGE_SIZE
    image_pixel_size: float = IMAGE_PIXEL_SIZE

    def job_file(self, *parts: str) -> Path:
        return self.root / self.job_dir / Path(*parts)

    def select_job_file(self, *parts: str) -> Path:
        return self.root / self.select_job_dir / Path(*parts)

    def set_ctf_premultiplied(self) -> None:
        """Mark the particles as extracted without `--no_ctf` (CTF-premultiplied)."""
        path = self.root / self.particles_star
        text = path.read_text()
        assert _optics_row(0) in text
        path.write_text(text.replace(_optics_row(0), _optics_row(1)))


@pytest.fixture
def relion_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RelionProject:
    """A minimal RELION 5 tomography project with an External job directory.

    The current directory is changed to the project root, because RELION runs jobs
    from there.
    """
    root = tmp_path / "project"
    tilt_image = "MotionCorr/job002/frames/TS_01_001.mrc"
    (root / tilt_image).parent.mkdir(parents=True)
    with mrcfile.new(root / tilt_image) as mrc:
        mrc.set_data(np.zeros((TILT_NY, TILT_NX), dtype=np.float32))

    ts_star = root / "Tomograms/job005/tilt_series/TS_01.star"
    ts_star.parent.mkdir(parents=True)
    ts_star.write_text(
        "data_TS_01\n\nloop_\n"
        "_rlnMicrographName #1\n_rlnTomoNominalStageTiltAngle #2\n"
        f"{tilt_image} 0.0\n"
    )
    (root / "Tomograms/job005/tomograms.star").write_text(
        "data_global\n\nloop_\n"
        "_rlnTomoName #1\n_rlnTomoTiltSeriesStarFile #2\n"
        "_rlnTomoTiltSeriesPixelSize #3\n"
        f"TS_01 Tomograms/job005/tilt_series/TS_01.star {TILT_SERIES_PIXEL_SIZE}\n"
    )
    particles = root / "Extract/job010/particles.star"
    particles.parent.mkdir(parents=True)
    particle_rows = []
    for name, n_visible in PARTICLES:
        visible = [1] * n_visible + [0] * (N_TILTS - n_visible)
        stack = f"Extract/job010/Subtomograms/{name}_stack2d.mrcs"
        frames = "[" + ",".join(map(str, visible)) + "]"
        particle_rows.append(f"TS_01 {name} {stack} 1 {frames}")
    particles.write_text(
        "# version 50001\n\ndata_general\n\n_rlnTomoSubTomosAre2DStacks 1\n\n\n"
        "# version 50001\n\ndata_optics\n\nloop_\n"
        "_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n_rlnImagePixelSize #3\n"
        "_rlnImageSize #4\n_rlnVoltage #5\n_rlnSphericalAberration #6\n"
        "_rlnAmplitudeContrast #7\n_rlnCtfDataAreCtfPremultiplied #8\n"
        f"{_optics_row(0)}\n\n"
        "# version 50001\n\ndata_particles\n\nloop_\n"
        "_rlnTomoName #1\n_rlnTomoParticleName #2\n_rlnImageName #3\n"
        "_rlnOpticsGroup #4\n_rlnTomoVisibleFrames #5\n"
        + "\n".join(particle_rows)
        + "\n"
    )

    # RELION creates the job directory with job.star and job_pipeline.star before
    # running the job.
    from himena_relion_cryodrgn.train import CryoDrgnTrainJob

    job_dir = Path("External/job011")
    (root / job_dir).mkdir(parents=True)
    (root / job_dir / "job_pipeline.star").write_text(
        _job_pipeline(
            f"{job_dir.as_posix()}/",
            [
                ("Tomograms/job005/tomograms.star", "TomogramGroupMetadata.star"),
                ("Extract/job010/particles.star", "ParticleGroupMetadata.star"),
            ],
        )
    )
    CryoDrgnTrainJob.prep_job_star(
        in_mics="Tomograms/job005/tomograms.star",
        in_parts="Extract/job010/particles.star",
    ).to_star_dict().write(root / job_dir / "job.star")

    select_job_dir = Path("External/job012")
    (root / select_job_dir).mkdir(parents=True)
    (root / select_job_dir / "job_pipeline.star").write_text(
        _job_pipeline(f"{select_job_dir.as_posix()}/", [])
    )

    monkeypatch.chdir(root)
    return RelionProject(
        root=root,
        tomograms_star="Tomograms/job005/tomograms.star",
        particles_star="Extract/job010/particles.star",
        job_dir=job_dir,
        select_job_dir=select_job_dir,
    )


@pytest.fixture
def fake_cryodrgn(monkeypatch: pytest.MonkeyPatch) -> FakeCryoDRGN:
    """Replace `subprocess.run` so that no cryoDRGN command is actually executed."""
    fake = FakeCryoDRGN()
    monkeypatch.setattr(subprocess, "run", fake)
    # `cryodrgn filter` needs a display on Linux; the fake does not open a window.
    monkeypatch.setenv("DISPLAY", ":0")
    return fake


@pytest.fixture
def trained_project(
    relion_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
) -> RelionProject:
    """The project after a successful CryoDRGN Train job (External/job011)."""
    from himena_relion.external import run_function

    from himena_relion_cryodrgn.train import CryoDrgnTrainJob

    run_function(
        [
            CryoDrgnTrainJob.import_path(),
            "--o", f"{relion_project.job_dir.as_posix()}/",
            "--in_mics", relion_project.tomograms_star,
            "--in_parts", relion_project.particles_star,
        ]
    )  # fmt: skip
    assert relion_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    fake_cryodrgn.calls.clear()
    fake_cryodrgn.envs.clear()
    return relion_project


@pytest.fixture
def cryodrgn_bin_dir(tmp_path: Path) -> Path:
    """bin directory of another environment with (dummy) cryoDRGN executables.

    They are never executed: `fake_cryodrgn` intercepts the calls.
    """
    bin_dir = tmp_path / "envs" / "cryodrgn" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("cryodrgn", "cryodrgn_utils"):
        exe = bin_dir / (f"{name}.exe" if sys.platform == "win32" else name)
        exe.write_text("#!/bin/sh\nexit 1\n")
        exe.chmod(0o755)
    return bin_dir


@functools.cache
def _can_symlink() -> bool:
    with tempfile.TemporaryDirectory() as d:
        try:
            os.symlink(os.path.join(d, "target"), os.path.join(d, "link"))
        except OSError:
            return False
    return True


@pytest.fixture(autouse=True)
def _emulate_symlink_if_not_permitted(monkeypatch: pytest.MonkeyPatch):
    """Windows without Developer Mode cannot create symlinks; emulate them by copying.

    RELION runs on Linux, where symlinks always work. A link to a missing target is
    left absent, so `Path.exists()` is False just like for a real broken symlink.
    """
    if _can_symlink():
        return

    def _symlink(src, dst, target_is_directory=False, *, dir_fd=None):
        dst = Path(dst)
        if dst.exists():
            raise FileExistsError(f"File exists: {dst}")
        src = Path(src)
        if not src.is_absolute():
            src = dst.parent / src
        if src.exists():
            shutil.copy2(src, dst)

    monkeypatch.setattr(os, "symlink", _symlink)
