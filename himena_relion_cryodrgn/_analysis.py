"""Read the outputs of cryoDRGN training for the job widgets (no Qt here)."""

import functools
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import mrcfile
import numpy as np

from himena_relion_cryodrgn import _impl

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf"
# `# =====> Epoch: {} Average gen loss = {:.6}, KLD = {:.6f}, total loss = {:.6f}; ...`
_EPOCH_SUMMARY = re.compile(
    rf"=====> Epoch: (\d+) Average gen loss = ({_FLOAT}), KLD = ({_FLOAT}), "
    rf"total loss = ({_FLOAT})"
)
# `# [Train Epoch: {}/{}] [{}/{} particles] gen loss=..., kld=..., beta=..., loss=...`
_TRAIN_BATCH = re.compile(r"\[Train Epoch: (\d+)/(\d+)\]")
_WEIGHTS = re.compile(r"weights\.(\d+)\.pkl")
_Z = re.compile(r"z\.(\d+)\.pkl")


@dataclass
class TrainLog:
    """Learning curve parsed from the run.log of `cryodrgn train_vae`."""

    epochs: list[int] = field(default_factory=list)
    gen_loss: list[float] = field(default_factory=list)  # reconstruction error
    kld: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    num_epochs: int | None = None  # target number of epochs
    current_epoch: int | None = None  # epoch being trained (from the batch lines)
    finished: bool = False

    @property
    def last_epoch(self) -> int | None:
        return self.epochs[-1] if self.epochs else None


def parse_train_log(path) -> TrainLog:
    """Parse the run.log written by `cryodrgn train_vae` (missing file: empty log).

    Lines are matched anywhere, so that a logging prefix such as
    "(INFO) (train_vae.py) (22-Nov-2026 10:00:00)" does not matter. If an epoch appears
    more than once (training resumed with `--load`), the last summary is kept.
    """
    log = TrainLog()
    path = Path(path)
    if not path.is_file():
        return log
    summaries: dict[int, tuple[float, float, float]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            if m := _EPOCH_SUMMARY.search(line):
                epoch = int(m.group(1))
                summaries[epoch] = (
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                )
                log.finished = False  # training continued (--load) after a finished run
            elif m := _TRAIN_BATCH.search(line):
                log.current_epoch = int(m.group(1))
                log.num_epochs = int(m.group(2))
                log.finished = False
            elif "Training complete" in line:
                log.finished = True
    for epoch in sorted(summaries):
        gen, kld, total = summaries[epoch]
        log.epochs.append(epoch)
        log.gen_loss.append(gen)
        log.kld.append(kld)
        log.total_loss.append(total)
    return log


def _numbered_files(workdir, pattern: re.Pattern) -> dict[int, Path]:
    workdir = Path(workdir)
    if not workdir.is_dir():
        return {}
    out = {}
    for path in workdir.iterdir():
        if m := pattern.fullmatch(path.name):
            out[int(m.group(1))] = path
    return out


def is_checkpoint_file(path) -> bool:
    """True for weights.pkl and weights.{epoch}.pkl, written after each epoch."""
    name = Path(path).name
    return name == "weights.pkl" or _WEIGHTS.fullmatch(name) is not None


def last_checkpoint_epoch(workdir) -> int | None:
    """The last epoch with a weights.{epoch}.pkl file."""
    epochs = _numbered_files(workdir, _WEIGHTS)
    return max(epochs) if epochs else None


def latent_epochs(workdir) -> list[int]:
    """Epochs with z.{epoch}.pkl (saved with the checkpoints), in ascending order."""
    return sorted(_numbered_files(workdir, _Z))


def resolve_epoch(workdir, epoch: int = -1) -> int | None:
    """Resolve epoch -1 to the last epoch with z.{epoch}.pkl, as `cryodrgn filter`."""
    if epoch != -1:
        return epoch
    epochs = latent_epochs(workdir)
    return epochs[-1] if epochs else None


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_z(workdir, epoch: int) -> np.ndarray | None:
    """z_mu of an epoch, shape (N, zdim). z.{epoch}.pkl has z_mu then z_logvar."""
    path = Path(workdir) / f"z.{epoch}.pkl"
    if not path.is_file():
        return None
    z = np.asarray(_load_pkl(path), dtype=np.float32)
    return z.reshape(len(z), -1)


@dataclass
class LatentEmbedding:
    """2D coordinates of the particles in the latent space, for plotting."""

    xy: np.ndarray  # (N, 2), one row per particle of the training dataset
    xlabel: str
    ylabel: str
    epoch: int


def load_latent_2d(workdir, epoch: int = -1) -> LatentEmbedding | None:
    """UMAP of the analysis if available, otherwise the first PCs of z."""
    epoch_ = resolve_epoch(workdir, epoch)
    if epoch_ is None:
        return None
    umap_path = Path(workdir) / f"analyze.{epoch_}" / "umap.pkl"
    if umap_path.is_file():
        xy = np.asarray(_load_pkl(umap_path), dtype=np.float32)
        return LatentEmbedding(xy[:, :2], "UMAP1", "UMAP2", epoch_)
    z = load_z(workdir, epoch_)
    if z is None or len(z) == 0:
        return None
    if z.shape[1] == 1:
        xy = np.stack([z[:, 0], np.zeros(len(z), dtype=z.dtype)], axis=1)
        return LatentEmbedding(xy, "z", "", epoch_)
    if z.shape[1] == 2:
        return LatentEmbedding(z, "z1", "z2", epoch_)
    centered = z - z.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return LatentEmbedding(centered @ vt[:2].T, "PC1", "PC2", epoch_)


def load_kmeans_labels(workdir, epoch: int, kmeans: int = -1) -> np.ndarray | None:
    """k-means labels of the analysis (-1: the first kmeans{K} folder, as filter)."""
    anlzdir = Path(workdir) / f"analyze.{epoch}"
    if not anlzdir.is_dir():
        return None
    if kmeans == -1:
        dirs = sorted(
            (d for d in anlzdir.iterdir() if re.fullmatch(r"kmeans\d+", d.name)),
            key=lambda d: int(d.name[6:]),
        )
        if not dirs:
            return None
        labels_path = dirs[0] / "labels.pkl"
    else:
        labels_path = anlzdir / f"kmeans{kmeans}" / "labels.pkl"
    if not labels_path.is_file():
        return None
    return np.asarray(_load_pkl(labels_path), dtype=int).ravel()


# --- volumes -----------------------------------------------------------------------

_VOLUME = re.compile(r"vol_(\d+)\.mrc")
_ANALYZE_DIR = re.compile(r"analyze\.(\d+)")
_PC_DIR = re.compile(r"pc(\d+)")
_KMEANS_DIR = re.compile(r"kmeans(\d+)")
_TRAVERSAL_DIR = re.compile(r"traversal_(\d+)")


@dataclass
class VolumeSeries:
    """Volumes to browse one by one, e.g. the volumes along a principal component."""

    name: str
    paths: list[Path]  # in order (vol_001.mrc, vol_002.mrc, ...)


def volume_files(directory) -> list[Path]:
    files = _numbered_files(directory, _VOLUME)
    return [files[k] for k in sorted(files)]


def selection_volume_series(select_job_path, train_job_path=None) -> list[VolumeSeries]:
    """Back-projections of a CryoDRGN Select job: of the selected particles, and of
    all the particles (the CryoDRGN Train job) to compare with."""
    series = []
    selected = Path(select_job_path) / _impl.BACKPROJECTION
    if selected.is_file():
        series.append(VolumeSeries("Back-projection (selected particles)", [selected]))
    if train_job_path is not None:
        all_particles = Path(train_job_path) / _impl.BACKPROJECTION
        if all_particles.is_file():
            series.append(
                VolumeSeries("Back-projection (all particles)", [all_particles])
            )
    return series


def find_volume_series(job_path) -> list[VolumeSeries]:
    """Volume series of a CryoDRGN Train job directory.

    - the back-projection of all particles (backproject/backproject.mrc)
    - the volumes along each graph traversal (traversal_NNN/vol_001.mrc ..., latest
      first), written by "Graph Traversal - CryoDRGN Train"
    - for each analysis (train/analyze.{epoch}, latest first), the volumes along
      each principal component (pc{i}/vol_001.mrc ...) and at the k-means centers
      (kmeans{K}/vol_001.mrc ...), as written by `cryodrgn analyze`.

    Only file names are collected here; volumes are read one by one when shown.
    """
    job_path = Path(job_path)
    series = []
    backprojection = job_path / _impl.BACKPROJECTION
    if backprojection.is_file():
        series.append(VolumeSeries("Back-projection (all particles)", [backprojection]))
    traversals = traversal_dirs(job_path)
    for number in sorted(traversals, reverse=True):
        if paths := volume_files(traversals[number]):
            name = f"Graph traversal {number}"
            if (epoch := _traversal_epoch(traversals[number])) is not None:
                name += f" (epoch {epoch})"
            series.append(VolumeSeries(name, paths))
    analyze_dirs = _numbered_files(job_path / "train", _ANALYZE_DIR)
    for epoch in sorted(analyze_dirs, reverse=True):
        anlzdir = analyze_dirs[epoch]
        pc_dirs = _numbered_files(anlzdir, _PC_DIR)
        for i in sorted(pc_dirs):
            if paths := volume_files(pc_dirs[i]):
                series.append(VolumeSeries(f"PC{i} (epoch {epoch})", paths))
        kmeans_dirs = _numbered_files(anlzdir, _KMEANS_DIR)
        for k in sorted(kmeans_dirs):
            if paths := volume_files(kmeans_dirs[k]):
                series.append(VolumeSeries(f"k-means {k} (epoch {epoch})", paths))
    return series


@functools.lru_cache(maxsize=8)
def _read_volume(path: str, mtime_ns: int) -> tuple[np.ndarray, float]:
    with mrcfile.open(path, permissive=True) as mrc:
        if mrc.data is None:
            raise ValueError(f"{path} has no data.")
        return np.array(mrc.data, dtype=np.float32), float(mrc.voxel_size.x)


def read_volume(path) -> tuple[np.ndarray, float]:
    """Read a volume and its voxel size (Å). The last few volumes are cached."""
    path = Path(path)
    return _read_volume(str(path), path.stat().st_mtime_ns)


def load_kmeans_centers(workdir, epoch: int, kmeans: int = -1) -> np.ndarray | None:
    """Rows of z.{epoch}.pkl closest to the k-means centers (centers_ind.txt)."""
    anlzdir = Path(workdir) / f"analyze.{epoch}"
    if kmeans == -1:
        dirs = _numbered_files(anlzdir, _KMEANS_DIR)
        if not dirs:
            return None
        path = dirs[min(dirs)] / "centers_ind.txt"
    else:
        path = anlzdir / f"kmeans{kmeans}" / "centers_ind.txt"
    if not path.is_file():
        return None
    return np.loadtxt(path, dtype=int, ndmin=1)


def traversal_dirs(job_path) -> dict[int, Path]:
    """Subdirectories of the graph traversals of a Train job: {1: traversal_001, ...}"""
    return {
        n: path
        for n, path in _numbered_files(job_path, _TRAVERSAL_DIR).items()
        if path.is_dir()
    }


def _traversal_epoch(traversal_dir: Path) -> int | None:
    """Epoch of the model used for a graph traversal (traversal.json)."""
    import json

    try:
        return int(json.loads((traversal_dir / "traversal.json").read_text())["epoch"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
