import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
from starfile_rs import read_star
from starfile_rs.components import SingleDataBlock
from starfile_rs.core import StarDict

# Number of tilts per particle used by `backproject_voxel --force-ntilts` and
# `train_vae`. Both commands default to 10 in cryoDRGN.
DEFAULT_NTILTS = 10

CRYODRGN = "cryodrgn"
CRYODRGN_UTILS = "cryodrgn_utils"


def find_command(name: str, bin_dir: str = "") -> str | None:
    """Full path of a cryoDRGN command in `bin_dir`, or in PATH if `bin_dir` is ""."""
    return shutil.which(name, path=bin_dir or None)


def _run(args: list[str], bin_dir: str = "", suppress_output: bool = False) -> None:
    """Run a cryoDRGN command, raising `CalledProcessError` if it fails.

    If `bin_dir` is given, the command is taken from there and `bin_dir` is put first
    in PATH, so that cryoDRGN can be installed in another environment than
    himena-relion without activating it.
    """
    if not bin_dir:
        subprocess.run(args, check=True)
        return
    exe = find_command(args[0], bin_dir)
    if exe is None:
        raise FileNotFoundError(f"`{args[0]}` was not found in {bin_dir}.")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH", "")])
    if suppress_output:
        subprocess.run(
            [exe, *args[1:]],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run([exe, *args[1:]], check=True, env=env)


def _first_value(star_path, column: str, block: str | None = None):
    """Value of the first row of `column` in a loop block (first block by default)."""
    star = read_star(star_path)
    data_block = star.first() if block is None else star[block]
    return data_block.trust_loop().to_polars()[column][0]


@dataclass
class ParticleImageParams:
    box_size: int  # box size of the extracted particle images in pixels
    pixel_size: float  # pixel size of the extracted particle images in Angstrom


def read_particle_image_params(particles_star_path) -> ParticleImageParams:
    """Read the extracted image size and pixel size from the optics block.

    `cryodrgn_utils parse_relion` copies the first optics group of particles.star, so
    the first row is used here too.
    """
    star = read_star(particles_star_path)
    optics = star["optics"].trust_loop().to_polars()
    return ParticleImageParams(
        box_size=int(optics["rlnImageSize"][0]),
        pixel_size=float(optics["rlnImagePixelSize"][0]),
    )


@dataclass
class ParseRelionOutput:
    particles_file: str
    ctf_file: str
    poses_file: str


def cryodrgn_parse_relion(
    tomo_star_path,
    particles_star_path,
    output_dir,
    bin_dir: str = "",
):
    ts_path = str(_first_value(tomo_star_path, "rlnTomoTiltSeriesStarFile"))
    img_path = str(_first_value(ts_path, "rlnMicrographName"))
    output_dir = Path(output_dir)
    output_particles_file = output_dir / "particles_2d.star"
    with mrcfile.open(img_path, header_only=True) as mrc:
        header = mrc.header
        assert header is not None
        tilt_x, tilt_y = int(header.nx), int(header.ny)
    _run(
        [
            "cryodrgn_utils", "parse_relion",
            "-t", str(tomo_star_path),
            "-p", str(particles_star_path),
            "--tilt-dim", str(tilt_x), str(tilt_y),
            "-o", str(output_particles_file),
        ],
        bin_dir=bin_dir,
    )  # fmt: skip

    # These pkl files are written by `cryodrgn parse_star`, not by parse_relion.
    return ParseRelionOutput(
        particles_file=str(output_particles_file),
        ctf_file=str(output_dir / "ctf.pkl"),
        poses_file=str(output_dir / "poses.pkl"),
    )


def cryodrgn_parse_star(
    input_star,
    ctffile,
    posesfile,
    apix: float,
    box_size: int,
    bin_dir: str = "",
):
    """Write ctf.pkl and poses.pkl.

    `apix` and `image_size` must be those of the images referred to by `input_star`
    (the extracted particles, before downsampling). cryoDRGN rescales the CTF pixel
    size from these values when training on downsampled images.
    """
    _run(
        [
            "cryodrgn", "parse_star", str(input_star),
            "--ctf", str(ctffile),
            "--poses", str(posesfile),
            "--Apix", str(round(apix, 3)),
            "-D", str(box_size),
            "--overwrite",
        ],
        bin_dir=bin_dir,
    )  # fmt: skip


@dataclass
class DownsampleOutput:
    particles_file: str


def cryodrgn_downsample(input_file, output_dir, px_binned: int, bin_dir: str = ""):
    output_file = Path(output_dir) / "particles_downsampled.star"
    _run(
        [
            "cryodrgn", "downsample", str(input_file),
            "-D", str(px_binned),
            "-o", str(output_file),
            "--outdir", str(output_dir),
        ],
        bin_dir=bin_dir,
        suppress_output=True,  # too much output from cryodrgn downsample
    )  # fmt: skip
    return DownsampleOutput(particles_file=str(output_file))


@dataclass
class BackprojectOutput:
    indices_file: str | None  # only written for tilt series


def cryodrgn_backproject_voxel(
    particles_star,
    ctf_file,
    poses_file,
    output_dir,
    dose_per_tilt: float | None = None,
    ntilts: int = DEFAULT_NTILTS,
    lazy: bool = False,
    bin_dir: str = "",
):
    """Back-project the particles. Tilt-series mode is used if `dose_per_tilt` is given.

    In tilt-series mode, particles with fewer than `ntilts` tilts are dropped and the
    indices of the remaining particles are returned.
    """
    args = [
        "cryodrgn", "backproject_voxel", str(particles_star),
        "--ctf", str(ctf_file),
        "--poses", str(poses_file),
        "-o", str(output_dir),
    ]  # fmt: skip
    indices_file = None
    if dose_per_tilt is not None:
        args += [
            "--tilt",
            "--dose-per-tilt", str(round(dose_per_tilt, 3)),
            "--ntilts", str(ntilts),
            "--force-ntilts",
        ]  # fmt: skip
        # file name defined in cryodrgn/commands/backproject_voxel.py
        indices_file = str(Path(output_dir) / f"indices_force_ntilts_{ntilts}.pkl")
    if lazy:
        args.append("--lazy")
    _run(args, bin_dir=bin_dir)
    return BackprojectOutput(indices_file=indices_file)


def cryodrgn_train_vae(
    particles_star,
    ctf_file,
    poses_file,
    output_dir,
    dose_per_tilt: float = 3.0,
    n_epochs: int = 20,
    zdim: int = 8,
    beta: float = 0.025,
    ntilts: int = DEFAULT_NTILTS,
    indices_file: str | None = None,
    bin_dir: str = "",
):
    cmd = [
        "cryodrgn", "train_vae", str(particles_star),
        "--ctf", str(ctf_file),
        "--poses", str(poses_file),
        "--encode-mode", "tilt",
        "--dose-per-tilt", str(round(dose_per_tilt, 3)),
        "--ntilts", str(ntilts),
        "--zdim", str(zdim),
        "--beta", str(round(beta, 4)),
        "--num-epochs", str(n_epochs),
        "-o", str(output_dir),
    ]  # fmt: skip
    if indices_file:
        cmd.extend(["--ind", str(indices_file)])
    _run(cmd, bin_dir=bin_dir)


# --- particle selection ----------------------------------------------------------


def load_indices(path) -> np.ndarray:
    """Load an index array saved by cryoDRGN (a pickled numpy array)."""
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f), dtype=int)


def save_indices(path, indices) -> None:
    """Save indices in a format cryoDRGN reads (`--ind`).

    A list of Python ints is pickled instead of a numpy array: cryoDRGN may run in
    another environment, and a numpy 2 array pickle may not load with numpy 1.
    """
    with open(path, "wb") as f:
        pickle.dump([int(i) for i in np.asarray(indices).ravel()], f)


def check_display() -> None:
    """`cryodrgn filter` opens a Tk window, which needs an X display on Linux."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "`cryodrgn filter` opens an interactive window, but no display is available "
            "(DISPLAY is not set). Run this job locally on a machine with a display, "
            "not through a queueing system."
        )


@dataclass
class FilterOutput:
    indices_file: str  # selected particles, numbered as in the training dataset
    inverse_indices_file: str  # all the other particles


def cryodrgn_filter(
    workdir,
    sel_dir,
    epoch: int | None = None,
    kmeans: int | None = None,
    bin_dir: str = "",
) -> FilterOutput:
    """Open the interactive particle selection of cryoDRGN and wait until it is closed.

    `--force` saves the selection as `indices.pkl` in `sel_dir` without prompting for
    a file name, since a RELION job has no interactive terminal. The saved indices
    refer to the whole dataset parsed for training, even if training used `--ind`.
    `epoch=-1` uses the last epoch and `kmeans=-1` the first k-means result found.
    """
    check_display()
    sel_dir = Path(sel_dir)
    out = FilterOutput(
        indices_file=str(sel_dir / "indices.pkl"),
        inverse_indices_file=str(sel_dir / "indices_inverse.pkl"),
    )
    # Never mistake the selection of a previous run for the current one.
    for path in (out.indices_file, out.inverse_indices_file):
        Path(path).unlink(missing_ok=True)
    _run(
        [
            "cryodrgn", "filter", str(workdir),
            "--epoch", str(-1 if epoch is None else epoch),
            "--kmeans", str(-1 if kmeans is None else kmeans),
            "--force",
            "--sel-dir", str(sel_dir),
        ],
        bin_dir=bin_dir,
    )  # fmt: skip
    if not Path(out.indices_file).exists():
        raise RuntimeError("No particles were selected in the cryoDRGN filter window.")
    return out


def particle_indices_to_star_rows(particles_2d_star, particle_indices) -> np.ndarray:
    """Convert cryoDRGN particle indices to row indices of the RELION particles.star.

    For tilt series, cryoDRGN numbers the particles in the order of first appearance
    of `rlnGroupName` in the star file written by `cryodrgn_utils parse_relion`.
    parse_relion skips particles that are not visible in any tilt, so this numbering
    can differ from the rows of the original particles.star. parse_relion also writes
    `rlnOriginalParticle` (1-based row in particles.star), which links the two.

    Returns sorted unique row indices, so that the particle order of particles.star
    is kept.
    """
    df = read_star(particles_2d_star).first().trust_loop().to_polars()
    for column in ("rlnGroupName", "rlnOriginalParticle"):
        if column not in df.columns:
            raise ValueError(
                f"{particles_2d_star} has no column {column!r}. It must be written by "
                "`cryodrgn_utils parse_relion`."
            )
    particles = df.unique(subset="rlnGroupName", keep="first", maintain_order=True)
    rows = particles["rlnOriginalParticle"].to_numpy().astype(int) - 1
    particle_indices = np.asarray(particle_indices, dtype=int)
    if particle_indices.size > 0 and (
        particle_indices.min() < 0 or particle_indices.max() >= rows.size
    ):
        raise IndexError(
            f"Particle indices must be in [0, {rows.size}), got "
            f"[{particle_indices.min()}, {particle_indices.max()}]."
        )
    return np.unique(rows[particle_indices])


def cryodrgn_filter_star(
    input_star, indices_file, output_star, bin_dir: str = ""
) -> None:
    """Select rows of the particles block of `input_star` by `indices_file`.

    The output has only `data_optics` and `data_particles`; use `restore_star_blocks`
    to put back the other blocks.
    """
    _run(
        [
            "cryodrgn_utils", "filter_star", str(input_star),
            "--ind", str(indices_file),
            "-o", str(output_star),
        ],
        bin_dir=bin_dir,
    )  # fmt: skip


def restore_star_blocks(original_star, filtered_star) -> None:
    """Put back the single-value blocks that cryoDRGN drops when it rewrites a star.

    cryoDRGN's star parser ignores blocks without a loop, and it writes only
    `data_optics` and `data_particles`. RELION 5 tomography particles also have
    `data_general` (`rlnTomoSubTomosAre2DStacks`), which RELION needs to read the
    particles correctly. The restored blocks are placed first, as RELION writes them.
    """
    original = read_star(original_star)
    filtered = read_star(filtered_star)
    restored = [
        block
        for name, block in original.items()
        if name not in filtered and isinstance(block, SingleDataBlock)
    ]
    if not restored:
        return
    StarDict.from_blocks([*restored, *filtered.values()]).write(filtered_star)
