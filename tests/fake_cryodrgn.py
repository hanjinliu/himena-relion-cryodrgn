"""A fake cryoDRGN command line used for dry-run tests.

`FakeCryoDRGN` replaces `subprocess.run`. It never executes cryoDRGN. Instead, it

1. parses the command with an argparse parser that mirrors the `add_args` of the real
   command (cryoDRGN main branch), so unknown or missing options fail the test just
   like they would make the real command exit with an error, and
2. creates the files the real command would write, so that the next step of the job
   can find its inputs. Star files and index arrays are real (with the same particle
   numbering as cryoDRGN); images, volumes and models are placeholders.

`CLIError` means that the job built an invalid command (a test failure).
`CommandFailed` means that the real command would exit with an error; it is reported
as a non-zero return code, like the real command.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mrcfile
import numpy as np
from starfile_rs import read_star
from starfile_rs.components import SingleDataBlock


class CLIError(AssertionError):
    """The real cryoDRGN would reject this command."""


class CommandFailed(Exception):
    """The real cryoDRGN command would exit with an error."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CLIError(f"{self.prog}: {message}")


def _require_file(path: str | None, what: str):
    if path is None or not Path(path).is_file():
        raise CLIError(f"{what} does not exist: {path}")


def _touch(path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake cryoDRGN output\n")


def _save_pkl(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _particles_of_tilt_star(path) -> list[list[int]]:
    """Tilt rows of each particle, like `TiltSeriesData.parse_particle_tilt`.

    Particles are grouped by `rlnGroupName` in the order of first appearance.
    """
    df = read_star(path).first().trust_loop().to_polars()
    particles: dict[str, list[int]] = {}
    for i, name in enumerate(df["rlnGroupName"].to_list()):
        particles.setdefault(name, []).append(i)
    return list(particles.values())


def _write_cryodrgn_star(path, data, data_optics=None) -> None:
    """Write polars tables the way `cryodrgn.starfile.write_star` does."""

    def block(header, df):
        lines = [f"{header}\n", "loop_"]
        lines += [f"_{col} #{i + 1}" for i, col in enumerate(df.columns)]
        lines += [" ".join(str(v) for v in row) for row in df.iter_rows()]
        return "\n".join(lines) + "\n"

    text = "# Created by fake cryoDRGN\n\n"
    if data_optics is not None:
        text += block("data_optics", data_optics) + "\n\n"
        text += block("data_particles", data)
    else:
        text += block("data_", data)
    Path(path).write_text(text)


# --- cryodrgn_utils parse_relion -------------------------------------------------


def _parse_relion_parser() -> _Parser:
    p = _Parser(prog="cryodrgn_utils parse_relion", add_help=False)
    p.add_argument("-t", "--tomograms", required=True)
    p.add_argument("-p", "--particles", required=True)
    p.add_argument("--tilt-dim", nargs=2, type=int, required=True)
    p.add_argument("-o", "--output", default="particles_2d.star")
    return p


def _parse_relion_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.tomograms, "--tomograms")
    _require_file(a.particles, "--particles")
    # The only output: one row per visible tilt image of each particle.
    particles = read_star(a.particles)["particles"].trust_loop().to_polars()
    lines = [
        "data_\n",
        "loop_",
        "_rlnImageName #1",
        "_rlnGroupName #2",
        "_rlnOriginalParticle #3",
    ]
    for idx, row in enumerate(particles.iter_rows(named=True)):
        visible = json.loads(row.get("rlnTomoVisibleFrames", "[1,1,1]"))
        n_visible = sum(1 for v in visible if v == 1)  # 0 -> particle is skipped
        for i in range(n_visible):
            image = f"{i + 1:06d}@{row['rlnImageName']}"
            lines.append(f"{image} {row['rlnTomoParticleName']} {idx + 1}")
    Path(a.output).write_text("\n".join(lines) + "\n")


# --- cryodrgn parse_star ---------------------------------------------------------


def _parse_star_parser() -> _Parser:
    p = _Parser(prog="cryodrgn parse_star", add_help=False)
    p.add_argument("input")
    p.add_argument("--ctf")
    p.add_argument("--poses")
    p.add_argument("--png")
    p.add_argument("-D", type=int)
    p.add_argument("--Apix", type=float)
    p.add_argument("--kv", type=float)
    p.add_argument("--cs", type=float)
    p.add_argument("-w", type=float)
    p.add_argument("--ps", type=float)
    p.add_argument("--overwrite", action="store_true")
    return p


def _parse_star_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.input, "input star file")
    # main() calls os.path.exists() on both without a None check.
    if a.ctf is None or a.poses is None:
        raise CLIError("cryodrgn parse_star: both --ctf and --poses must be given")
    _touch(a.ctf)
    _touch(a.poses)


# --- cryodrgn downsample ---------------------------------------------------------


def _downsample_parser() -> _Parser:
    p = _Parser(prog="cryodrgn downsample", add_help=False)
    p.add_argument("input")
    p.add_argument("-D", type=int, required=True)
    p.add_argument("-o", "--outfile", type=os.path.abspath, required=True)
    p.add_argument("--outdir", type=os.path.abspath)
    p.add_argument("-b", "--batch-size", type=int, default=5000)
    p.add_argument("--is-vol", action="store_true")
    p.add_argument("--chunk", type=int)
    p.add_argument("--datadir")
    p.add_argument("--max-threads", type=int, default=16)
    p.add_argument("--ind", type=os.path.abspath)
    return p


def _downsample_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.input, "input particles")
    if a.D % 2 != 0:
        raise CLIError(f"cryodrgn downsample: new box size must be even, got {a.D}")
    outdir = Path(a.outdir or os.path.dirname(a.outfile))
    _touch(outdir / "particles.mrcs")
    # Same rows in the same order. The image names are made relative to --outdir
    # (`os.path.relpath(newpath, outdir)`), so the star file only works from there.
    lines = []
    for line in Path(a.input).read_text().splitlines():
        if "@" in (tokens := line.split() or [""])[0]:
            index = tokens[0].split("@")[0]
            line = " ".join([f"{index}@particles.mrcs", *tokens[1:]])
        lines.append(line)
    Path(a.outfile).write_text("\n".join(lines) + "\n")


# --- cryodrgn backproject_voxel --------------------------------------------------


def _backproject_voxel_parser() -> _Parser:
    p = _Parser(prog="cryodrgn backproject_voxel", add_help=False)
    p.add_argument("particles", type=os.path.abspath)
    p.add_argument("--poses", type=os.path.abspath, required=True)
    p.add_argument("--ctf", type=os.path.abspath)
    p.add_argument("--outdir", "-o", type=os.path.abspath, required=True)
    p.add_argument("--no-half-maps", action="store_false", dest="half_maps")
    p.add_argument("--no-fsc-vals", action="store_false", dest="fsc_vals")
    p.add_argument("-b", "--batch-size", type=int, default=1000)
    p.add_argument("--ctf-alg", choices=("flip", "mul"), default="mul")
    p.add_argument("--reg-weight", type=float, default=1.0)
    p.add_argument("--output-sumcount", action="store_true")
    p.add_argument("--uninvert-data", action="store_false", dest="invert_data")
    p.add_argument("--datadir", type=os.path.abspath)
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--ind", type=os.path.abspath)
    p.add_argument("--first", type=int)
    p.add_argument("--log-interval", default="5000")
    p.add_argument("--tilt", action="store_true")
    p.add_argument("--ntilts", type=int, default=10)
    p.add_argument("--force-ntilts", action="store_true", dest="force_ntilts")
    p.add_argument("-d", "--dose-per-tilt", type=float)
    p.add_argument("-a", "--angle-per-tilt", type=float, default=3)
    return p


def _backproject_voxel_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.particles, "particles")
    _require_file(a.poses, "--poses")
    if a.ctf is not None:
        _require_file(a.ctf, "--ctf")
    if a.ind is not None:
        _require_file(a.ind, "--ind")
    if a.tilt and a.dose_per_tilt is None:
        raise CLIError("cryodrgn backproject_voxel: --tilt requires --dose-per-tilt")
    if a.force_ntilts and not a.tilt:
        raise CLIError("cryodrgn backproject_voxel: --force-ntilts requires --tilt")
    if a.tilt:
        # TiltSeriesData groups the tilt images of a particle by rlnGroupName
        columns = read_star(a.particles).first().trust_loop().columns
        if "rlnGroupName" not in columns:
            raise CommandFailed(f"{a.particles} is not a star file of tilt images")
        n_particles = len(_particles_of_tilt_star(a.particles))
        if a.ind is not None and not a.force_ntilts:
            ind = np.asarray(_load_pkl(a.ind), dtype=int)
            if ind.size and (ind.min() < 0 or ind.max() >= n_particles):
                raise CommandFailed("--ind has indices out of the particles")
            fake.backprojected[outdir_key(a.outdir)] = ind.tolist()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(outdir / "backproject.mrc", overwrite=True) as mrc:
        zz, yy, xx = np.mgrid[-8:8, -8:8, -8:8]
        mrc.set_data(np.exp(-(xx**2 + yy**2 + zz**2) / 20).astype(np.float32))
    if a.half_maps:
        _touch(outdir / "half_map_a.mrc")
        _touch(outdir / "half_map_b.mrc")
        if a.fsc_vals:
            _touch(outdir / "fsc-vals.txt")
            _touch(outdir / "fsc-plot.png")
    if a.force_ntilts:
        # particles (not tilts) with at least --ntilts tilt images
        particles = _particles_of_tilt_star(a.particles)
        keep = [i for i, tilts in enumerate(particles) if len(tilts) >= a.ntilts]
        _save_pkl(outdir / f"indices_force_ntilts_{a.ntilts}.pkl", np.array(keep))


# --- cryodrgn train_vae ----------------------------------------------------------


def _train_vae_parser() -> _Parser:
    p = _Parser(prog="cryodrgn train_vae", add_help=False)
    p.add_argument("particles", type=os.path.abspath)
    p.add_argument("-o", "--outdir", type=os.path.abspath, required=True)
    p.add_argument("--zdim", type=int, required=True)
    p.add_argument("--poses", type=os.path.abspath, required=True)
    p.add_argument("--ctf", type=os.path.abspath)
    p.add_argument("--load")
    p.add_argument("--no-analysis", action="store_false", dest="do_analysis")
    p.add_argument("--checkpoint", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--shuffle-seed", type=int)
    # dataset loading
    p.add_argument("--ind", type=os.path.abspath)
    p.add_argument("--uninvert-data", action="store_false", dest="invert_data")
    p.add_argument("--no-window", action="store_false", dest="window")
    p.add_argument("--window-r", type=float, default=0.85)
    p.add_argument("--datadir", type=os.path.abspath)
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--shuffler-size", type=int, default=0)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--max-threads", type=int, default=16)
    # tilt series
    p.add_argument("--ntilts", type=int, default=10)
    p.add_argument("--random-tilts", action="store_true")
    p.add_argument("--t-emb-dim", type=int, default=64)
    p.add_argument("--tlayers", type=int, default=3)
    p.add_argument("--tdim", type=int, default=1024)
    p.add_argument("-d", "--dose-per-tilt", type=float)
    p.add_argument("-a", "--angle-per-tilt", type=float, default=3)
    # training
    p.add_argument("-n", "--num-epochs", type=int, default=20)
    p.add_argument("-b", "--batch-size", type=int, default=16)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta")
    p.add_argument("--beta-control", type=float)
    p.add_argument("--norm", type=float, nargs=2)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--multigpu", action="store_true")
    # pose SGD
    p.add_argument("--do-pose-sgd", action="store_true")
    p.add_argument("--pretrain", type=int, default=1)
    p.add_argument("--emb-type", choices=("s2s2", "quat"), default="quat")
    p.add_argument("--pose-lr", type=float, default=3e-4)
    # encoder / decoder
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--enc-dim", type=int, default=1024)
    p.add_argument(
        "--encode-mode", choices=("conv", "resid", "mlp", "tilt"), default="resid"
    )
    p.add_argument("--enc-mask", type=int)
    p.add_argument("--use-real", action="store_true")
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--dec-dim", type=int, default=1024)
    p.add_argument("--pe-type", default="gaussian")
    p.add_argument("--feat-sigma", type=float, default=0.5)
    p.add_argument("--pe-dim", type=int)
    p.add_argument("--domain", choices=("hartley", "fourier"), default="fourier")
    p.add_argument("--activation", choices=("relu", "leaky_relu"), default="relu")
    return p


def _train_vae_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.particles, "particles")
    _require_file(a.poses, "--poses")
    if a.ctf is not None:
        _require_file(a.ctf, "--ctf")
    if a.ind is not None:
        _require_file(a.ind, "--ind")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # The real file is YAML; JSON is valid YAML and keeps the fake dependency-free.
    config = {
        "dataset_args": {
            "particles": a.particles,
            "ctf": a.ctf,
            "poses": a.poses,
            "ind": a.ind,
        },
        # the names in the config.yaml of cryoDRGN
        "model_args": {
            "encode_mode": a.encode_mode,
            "zdim": a.zdim,
            "qlayers": a.enc_layers,
            "qdim": a.enc_dim,
            "players": a.dec_layers,
            "pdim": a.dec_dim,
        },
    }
    start_epoch = 1
    if a.load is not None:
        # torch.load(...)["epoch"] + 1; the fake reads the epoch from the file name
        _require_file(a.load, "--load")
        start_epoch = int(Path(a.load).name.split(".")[1]) + 1
        # The checkpoint is loaded into the network made from the arguments.
        trained = json.loads((outdir / "config.yaml").read_text())["model_args"]
        if trained != config["model_args"]:
            raise CommandFailed(
                "Error(s) in loading state_dict for HetOnlyVAE: size mismatch "
                f"(trained: {trained}, given: {config['model_args']})"
            )
        if start_epoch > a.num_epochs:
            raise CommandFailed(
                f"The number of epochs to train must be greater than {a.num_epochs}!"
            )
    (outdir / "config.yaml").write_text(json.dumps(config, indent=2))

    n_particles = len(_particles_of_tilt_star(a.particles))
    if a.ind is not None:
        n_particles = len(_load_pkl(a.ind))
    rng = np.random.default_rng(0)
    # run.log has bare messages (no formatter on the FileHandler of train_vae), and the
    # FileHandler appends to it.
    with open(outdir / "run.log", "a") as log:
        log.write("cryodrgn train_vae ...\ncryoDRGN 3.5.0\n")
        for epoch in range(start_epoch, a.num_epochs + 1):  # counted from 1
            gen, kld = 1.0 / epoch, 0.1 + 0.01 * epoch
            log.write(
                f"# [Train Epoch: {epoch}/{a.num_epochs}] [{n_particles}/{n_particles} "
                f"particles] gen loss={gen:.6f}, kld={kld:.6f}, beta=0.025000, "
                f"loss={gen + 0.025 * kld:.6f}\n"
                f"# =====> Epoch: {epoch} Average gen loss = {gen:.6}, KLD = {kld:.6f}, "
                f"total loss = {gen + 0.025 * kld:.6f}; Finished in 0:00:01\n"
            )
            # the summary is logged before the checkpoint is saved
            if a.checkpoint and epoch % a.checkpoint == 0:
                _touch(outdir / f"weights.{epoch}.pkl")
                _save_z(
                    outdir / f"z.{epoch}.pkl", rng.normal(size=(n_particles, a.zdim))
                )
        log.write("Training complete\nFinished in 0:00:20 (0:00:01 per epoch)\n")
    _touch(outdir / "weights.pkl")
    _save_z(outdir / "z.pkl", rng.normal(size=(n_particles, a.zdim)))
    if a.do_analysis:  # `cryodrgn analyze <outdir> <num_epochs>` with --ksample 20
        anlzdir = outdir / f"analyze.{a.num_epochs}"
        if a.zdim > 2:
            _save_pkl(anlzdir / "umap.pkl", rng.normal(size=(n_particles, 2)))
        labels = np.arange(n_particles) % 20
        _save_pkl(anlzdir / "kmeans20" / "labels.pkl", labels)
        # rows of z closest to the k-means centers
        centers_ind = np.unique(labels, return_index=True)[1]
        np.savetxt(anlzdir / "kmeans20" / "centers_ind.txt", centers_ind, fmt="%d")
        # volumes (eval_vol: vol_001.mrc, ... 1-based), --pc 2 --n-per-pc 10
        for i in (1, 2):
            _write_volumes(anlzdir / f"pc{i}", 10)
        _write_volumes(anlzdir / "kmeans20", 20)


def _write_volumes(outdir: Path, n: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    zz, yy, xx = np.mgrid[-4:5, -4:5, -4:5]  # D + 1 = 9
    for i in range(1, n + 1):
        vol = np.exp(-(xx**2 + yy**2 + zz**2) / (2.0 + i)).astype(np.float32)
        with mrcfile.new(outdir / f"vol_{i:03d}.mrc", overwrite=True) as mrc:
            mrc.set_data(vol)
            mrc.voxel_size = 5.4
    np.savetxt(outdir / "z_values.txt", np.zeros((n, 8)))


def _save_z(path, z_mu: np.ndarray) -> None:
    """z.{epoch}.pkl: z_mu then z_logvar, two pickles in one file."""
    with open(path, "wb") as f:
        pickle.dump(z_mu.astype(np.float32), f)
        pickle.dump(np.zeros_like(z_mu, dtype=np.float32), f)


# --- cryodrgn filter -------------------------------------------------------------


def _filter_parser() -> _Parser:
    p = _Parser(prog="cryodrgn filter", add_help=False)
    p.add_argument("outdir", type=os.path.abspath)
    p.add_argument("--epoch", "-e", type=int, default=-1)
    p.add_argument("--kmeans", "-k", type=int, default=-1)
    p.add_argument("--force", "-f", action="store_true")
    p.add_argument("--plot-inds")
    p.add_argument("--sel-dir")
    return p


def _filter_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    workdir = Path(a.outdir)
    if not (workdir / "config.yaml").is_file():
        raise CommandFailed("Missing config.yaml file in given output folder!")
    epochs = [
        int(m.group(1))
        for f in workdir.iterdir()
        if (m := re.fullmatch(r"z\.([0-9]+)\.pkl", f.name))
    ]
    if not epochs:
        raise CommandFailed(
            "Filtering is not available for homogeneous reconstruction!"
        )
    epoch = max(epochs) if a.epoch == -1 else a.epoch
    anlzdir = workdir / f"analyze.{epoch}"
    if not anlzdir.is_dir():
        raise CommandFailed(f"No analysis available for epoch {epoch}")
    if a.kmeans == -1:
        kmeans_dirs = [
            d
            for d in anlzdir.iterdir()
            if d.is_dir() and re.match(r"^kmeans\d+$", d.name)
        ]
        if not kmeans_dirs:
            raise CommandFailed("Did not find any k-means clustering result outputs!")
        kmeans_dir = kmeans_dirs[0]
    else:
        kmeans_dir = anlzdir / f"kmeans{a.kmeans}"
        if not kmeans_dir.is_dir():
            raise CommandFailed(f"No k-means results for k={a.kmeans}")
    _require_file(str(kmeans_dir / "labels.pkl"), "k-means labels")

    # The interactive window. `fake.filter_selection` is what the user lassoes: the
    # positions of the points in the plot, i.e. in the training dataset.
    if fake.filter_selection is None:
        return  # "Exiting without having made a selection."
    if not a.force:
        raise CLIError(
            "cryodrgn filter: without --force it asks for a file name with input(), "
            "but a RELION job has no interactive terminal"
        )
    dataset = json.loads((workdir / "config.yaml").read_text())["dataset_args"]
    n_particles = len(_particles_of_tilt_star(dataset["particles"]))
    all_indices = np.arange(n_particles)
    if dataset["ind"] is not None:
        all_indices = all_indices[_load_pkl(dataset["ind"])]
    selected = np.array([all_indices[i] for i in fake.filter_selection], dtype=int)
    filename = os.path.join(a.sel_dir, "indices") if a.sel_dir else "indices"
    _save_pkl(filename + ".pkl", selected)
    _save_pkl(filename + "_inverse.pkl", np.setdiff1d(np.arange(n_particles), selected))


# --- cryodrgn_utils filter_star --------------------------------------------------


def _filter_star_parser() -> _Parser:
    p = _Parser(prog="cryodrgn_utils filter_star", add_help=False)
    p.add_argument("input")
    p.add_argument("--ind", required=True)
    p.add_argument("--et", action="store_true")
    p.add_argument("--micrograph-files", "-m", action="store_true")
    p.add_argument("-o", type=os.path.abspath)
    return p


def _filter_star_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.input, "input star file")
    _require_file(a.ind, "--ind")
    if a.o is None:
        raise CLIError("cryodrgn_utils filter_star: -o is needed to write the output")
    if a.et or a.micrograph_files:
        raise CLIError("--et / --micrograph-files are not emulated")
    # cryodrgn.starfile ignores single-value blocks such as data_general, and uses the
    # largest non-optics loop block as the particle table.
    star = read_star(a.input)
    loops = {
        name: block.trust_loop().to_polars()
        for name, block in star.items()
        if not isinstance(block, SingleDataBlock)
    }
    optics = loops.pop("optics", None)
    data = max(loops.values(), key=len)
    # `stardf.loc[ind]` accepts both a numpy array and a list of ints.
    filtered = data[np.asarray(_load_pkl(a.ind), dtype=int).tolist()]
    if optics is not None:
        used = set(filtered["rlnOpticsGroup"].to_list())
        optics = optics.filter(optics["rlnOpticsGroup"].is_in(list(used)))
    _write_cryodrgn_star(a.o, filtered, optics)


# --- cryodrgn graph_traversal ----------------------------------------------------


def _graph_traversal_parser() -> _Parser:
    p = _Parser(prog="cryodrgn graph_traversal", add_help=False)
    p.add_argument("zfile")
    p.add_argument("--anchors", required=True, nargs="+")
    p.add_argument("--max-neighbors", type=int, default=10)
    p.add_argument("--avg-neighbors", type=float, default=5)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--outtxt", "-o", type=os.path.abspath, default="z-path.txt")
    p.add_argument("--outind", type=os.path.abspath, default="z-path-indices.txt")
    p.add_argument("--verbose", action="store_true")
    return p


def _graph_traversal_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.zfile, "z file")
    z = np.asarray(_load_pkl(a.zfile))  # the first pickle: z_mu
    anchors = [int(x) for x in a.anchors]
    if any(i < 0 or i >= len(z) for i in anchors):
        raise CommandFailed("anchor index out of the z file")
    # a path through each anchor, with one particle between consecutive anchors
    path = [anchors[0]]
    for prev, nxt in itertools.pairwise(anchors):
        path += [(prev + nxt) // 2, nxt]
    np.savetxt(a.outtxt, z[path])
    np.savetxt(a.outind, path, fmt="%d")


# --- cryodrgn eval_vol ------------------------------------------------------------


def _eval_vol_parser() -> _Parser:
    p = _Parser(prog="cryodrgn eval_vol", add_help=False)
    p.add_argument("weights")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-o", type=os.path.abspath, required=True)
    p.add_argument("--device", type=int)
    p.add_argument("--prefix", default="vol_")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-z", type=float, nargs="*")
    p.add_argument("--z-start", type=float, nargs="*")
    p.add_argument("--z-end", type=float, nargs="*")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--zfile")
    p.add_argument("--Apix", type=float, default=1)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--invert", action="store_true")
    p.add_argument("-d", "--downsample", type=int)
    p.add_argument("--low-pass", type=float)
    p.add_argument("--crop", type=int)
    p.add_argument("--vol-start-index", type=int, default=1)
    return p


def _eval_vol_run(a: argparse.Namespace, fake: FakeCryoDRGN) -> None:
    _require_file(a.weights, "weights")
    _require_file(a.config, "--config")
    if a.zfile is None:
        raise CLIError("cryodrgn eval_vol: only --zfile is emulated")
    _require_file(a.zfile, "--zfile")
    if a.downsample is not None and a.downsample % 2 != 0:
        raise CommandFailed("Boxsize must be even")
    z = np.loadtxt(a.zfile, ndmin=2)
    box = (a.downsample or 8) + 1  # lattice D = image size + 1
    zz, yy, xx = np.mgrid[:box, :box, :box] - box // 2
    Path(a.o).mkdir(parents=True, exist_ok=True)
    for i, _ in enumerate(z, start=a.vol_start_index):
        vol = np.exp(-(xx**2 + yy**2 + zz**2) / (2.0 + i)).astype(np.float32)
        if a.invert:
            vol *= -1
        with mrcfile.new(Path(a.o) / f"{a.prefix}{i:03d}.mrc", overwrite=True) as mrc:
            mrc.set_data(vol)
            mrc.voxel_size = a.Apix


_COMMANDS: dict[tuple[str, str], tuple[Callable[[], _Parser], Callable]] = {
    ("cryodrgn_utils", "parse_relion"): (_parse_relion_parser, _parse_relion_run),
    ("cryodrgn", "parse_star"): (_parse_star_parser, _parse_star_run),
    ("cryodrgn", "downsample"): (_downsample_parser, _downsample_run),
    ("cryodrgn", "backproject_voxel"): (
        _backproject_voxel_parser,
        _backproject_voxel_run,
    ),
    ("cryodrgn", "train_vae"): (_train_vae_parser, _train_vae_run),
    ("cryodrgn", "filter"): (_filter_parser, _filter_run),
    ("cryodrgn_utils", "filter_star"): (_filter_star_parser, _filter_star_run),
    ("cryodrgn", "graph_traversal"): (_graph_traversal_parser, _graph_traversal_run),
    ("cryodrgn", "eval_vol"): (_eval_vol_parser, _eval_vol_run),
}


def outdir_key(path) -> str:
    return Path(path).name


@dataclass
class FakeCryoDRGN:
    """Drop-in replacement of `subprocess.run` for cryoDRGN commands."""

    calls: list[list[str]] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)
    """Subcommands that exit with a non-zero code (after recording the call)."""
    after: dict[str, Callable[[], None]] = field(default_factory=dict)
    """Callbacks run after a subcommand succeeded, e.g. to request an abort."""
    filter_selection: list[int] | None = None
    """Points lassoed in the `cryodrgn filter` window (None: closed without selection)."""
    errors: list[str] = field(default_factory=list)
    """Messages of the commands that failed (`CommandFailed`)."""
    envs: list[dict[str, str] | None] = field(default_factory=list)
    backprojected: dict[str, list[int]] = field(default_factory=dict)
    """Particles back-projected with --ind, by output directory name."""
    """`env` passed to each call (None: inherited from the current process)."""

    def __call__(self, cmd, *args, check: bool = False, env=None, **kwargs):
        cmd = [os.fspath(c) for c in cmd]
        self.calls.append(cmd)
        self.envs.append(env)
        key = _command_key(cmd)
        if key not in _COMMANDS:
            raise CLIError(f"Unexpected command: {cmd}")
        make_parser, run = _COMMANDS[key]
        namespace = make_parser().parse_args(cmd[2:])
        subcommand = cmd[1]
        returncode = 0
        if subcommand in self.fail_on:
            returncode = 1
        else:
            try:
                run(namespace, self)
            except CommandFailed as e:
                self.errors.append(f"{' '.join(cmd[:2])}: {e}")
                returncode = 1
            else:
                if callback := self.after.get(subcommand):
                    callback()
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return subprocess.CompletedProcess(cmd, returncode)

    @property
    def subcommands(self) -> list[str]:
        return [c[1] for c in self.calls]

    def parsed(self, subcommand: str) -> argparse.Namespace:
        """Parsed arguments of the (only) call of the given subcommand."""
        matches = [c for c in self.calls if c[1] == subcommand]
        assert len(matches) == 1, f"{subcommand} called {len(matches)} times"
        cmd = matches[0]
        make_parser, _ = _COMMANDS[_command_key(cmd)]
        return make_parser().parse_args(cmd[2:])


def _command_key(cmd: list[str]) -> tuple[str, str]:
    """("cryodrgn", "train_vae") also for "/env/bin/cryodrgn" or "cryodrgn.exe"."""
    return Path(cmd[0]).stem, cmd[1]
