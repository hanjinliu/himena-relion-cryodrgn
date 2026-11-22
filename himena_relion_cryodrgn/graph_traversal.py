"""Volumes along a path through the latent space (cryodrgn graph_traversal).

"Graph Traversal - CryoDRGN Train" continues a CryoDRGN Train job: each run writes
the volumes showing the structural change along the path to a new subdirectory of
the job (traversal_001, traversal_002, ...), browsed in the volume viewer of the job
window.

No `from __future__ import annotations`: himena-relion reads the annotations of
`run` at runtime.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from himena_relion.external import RelionExternalContinue

from himena_relion_cryodrgn import _analysis, _impl
from himena_relion_cryodrgn._job_base import CRYODRGN_BIN_DIR, _CryoDrgnJob
from himena_relion_cryodrgn.select_particles import TrainJobFiles
from himena_relion_cryodrgn.train import (
    _DEFAULT_ARCH,
    BOX_SIZE,
    DEC_DIM,
    DEC_LAYERS,
    DOSE_PER_TILT,
    ENC_DIM,
    ENC_LAYERS,
    FIXED_PARAMS,
    LAZY,
    N_EPOCHS,
    ZDIM,
    CryoDrgnTrainJob,
)

TRAVERSAL_INFO = "traversal.json"

EPOCH = Annotated[
    int,
    {
        "label": "Epoch",
        "min": -1,
        "tooltip": "Training epoch of the model. -1 uses the last epoch.",
        "group": "Graph traversal",
    },
]
ANCHORS = Annotated[
    str,
    {
        "label": "Anchors",
        "tooltip": (
            "Particles to visit, as rows of z.{epoch}.pkl (the training particles), "
            "separated by commas, e.g. '12,345,678'. If empty, the particles closest "
            "to the k-means centers are used."
        ),
        "group": "Graph traversal",
    },
]
KMEANS = Annotated[
    int,
    {
        "label": "K-means",
        "min": -1,
        "tooltip": (
            "k of the k-means result whose centers are the anchors when no anchors "
            "are given. -1 uses the smallest k found (20 after training)."
        ),
        "group": "Graph traversal",
    },
]
DOWNSAMPLE = Annotated[
    int,
    {
        "label": "Downsample volumes to (pix)",
        "min": 0,
        "tooltip": (
            "Box size of the volumes (even, smaller than the training box size), to "
            "save disk space. 0 keeps the training box size."
        ),
        "group": "Graph traversal",
    },
]
FLIP = Annotated[bool, {"label": "Flip handedness", "group": "Graph traversal"}]


def parse_anchors(text: str) -> list[int]:
    """'12 345, 678' -> [12, 345, 678]."""
    return [int(s) for s in re.split(r"[,\s]+", text.strip()) if s]


@dataclass
class TraversalPlan:
    """Validated inputs of the graph traversal of a CryoDRGN Train job."""

    train: TrainJobFiles
    epoch: int
    anchors: list[int]  # rows of z.{epoch}.pkl
    anchors_from: str  # "k-means centers" or "given"
    max_neighbors: int
    avg_neighbors: float
    downsample: int | None  # box size of the volumes, if smaller than training
    apix: float  # pixel size of the volumes
    flip: bool
    invert: bool

    @property
    def z_file(self) -> Path:
        return self.train.workdir / f"z.{self.epoch}.pkl"

    @property
    def weights(self) -> Path:
        return self.train.workdir / f"weights.{self.epoch}.pkl"

    @property
    def config(self) -> Path:
        return self.train.workdir / "config.yaml"

    def info(self) -> dict:
        """Parameters saved with the volumes."""
        out = asdict(self)
        out.pop("train")
        return out


def prepare_traversal(
    train_job_path,
    epoch: int = -1,
    anchors: str = "",
    kmeans: int = -1,
    max_neighbors: int = 10,
    avg_neighbors: float = 5.0,
    downsample: int = 0,
    flip: bool = False,
    invert: bool = False,
) -> TraversalPlan:
    """Check the inputs before running anything."""
    train = TrainJobFiles.from_weights(Path(train_job_path) / "weights.pkl")
    epoch_ = _analysis.resolve_epoch(train.workdir, epoch)
    if epoch_ is None:
        raise RuntimeError(f"No trained epoch (z.{{epoch}}.pkl) in {train.workdir}.")
    for path in (
        train.workdir / f"z.{epoch_}.pkl",
        train.workdir / f"weights.{epoch_}.pkl",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist.")
    train_box = train.training_box_size()
    if downsample and (downsample % 2 != 0 or downsample >= train_box):
        raise ValueError(
            f"downsample ({downsample}) must be an even number smaller than the "
            f"training box size ({train_box})."
        )
    try:
        anchor_ids = parse_anchors(anchors)
    except ValueError:
        raise ValueError(
            f"Anchors must be integers separated by commas: {anchors!r}"
        ) from None
    if anchor_ids:
        anchors_from = "given"
    else:
        centers = _analysis.load_kmeans_centers(train.workdir, epoch_, kmeans)
        if centers is None:
            raise RuntimeError(
                f"No k-means result for epoch {epoch_} (analyze.{epoch_}/kmeans"
                "*/centers_ind.txt). Give the anchors, or analyze this epoch."
            )
        anchor_ids = [int(i) for i in centers]
        anchors_from = "k-means centers"
    if len(anchor_ids) < 2:
        raise ValueError("At least two anchors are needed for a path.")
    vol_box = downsample or train_box
    return TraversalPlan(
        train=train,
        epoch=epoch_,
        anchors=anchor_ids,
        anchors_from=anchors_from,
        max_neighbors=max_neighbors,
        avg_neighbors=avg_neighbors,
        downsample=downsample or None,
        apix=train.training_pixel_size() * train_box / vol_box,
        flip=flip,
        invert=invert,
    )


def next_traversal_dir(job_path) -> Path:
    """A new subdirectory for the volumes: traversal_001, traversal_002, ..."""
    numbers = _analysis.traversal_dirs(job_path)
    return Path(job_path) / f"traversal_{max(numbers, default=0) + 1:03d}"


class CryoDrgnTrainGraphTraversal(_CryoDrgnJob, RelionExternalContinue):
    """Volumes along the shortest path through the latent space.

    `cryodrgn graph_traversal` finds the shortest path through the nearest-neighbor
    graph of the particles in the latent space that visits the anchors, and
    `cryodrgn eval_vol` generates a volume for each particle on the path. By default,
    the anchors are the particles closest to the k-means centers. The volumes are
    written to a new subdirectory traversal_NNN of the CryoDRGN Train job.
    """

    original_class = CryoDrgnTrainJob

    @classmethod
    def job_title(cls) -> str:
        return "Graph Traversal"

    @classmethod
    def setup_widgets(cls, widgets):
        super().setup_widgets(widgets)  # the input nodes
        for name in (*FIXED_PARAMS, "n_epochs"):  # the training
            if (widget := widgets.get(name)) is not None:
                widget.enabled = False

    @classmethod
    def prerun_check(cls, **kwargs) -> None:
        try:
            parse_anchors(str(kwargs.get("anchors") or ""))
        except ValueError:
            raise ValueError(
                f"Anchors must be integers separated by commas: {kwargs['anchors']!r}"
            ) from None
        super().prerun_check(**kwargs)  # the cryoDRGN commands

    def make_job_star(self, **kwargs):
        # RELION passes parameter values unquoted, so "12, 345" would be split into
        # two arguments: write the anchors without spaces.
        if anchors := str(kwargs.get("anchors") or ""):
            kwargs["anchors"] = ",".join(str(a) for a in parse_anchors(anchors))
        return super().make_job_star(**kwargs)

    # The parameters of the Train job (unused here) are also passed, as the job.star
    # of the continued job has all of them. External jobs have 10 parameter slots:
    # himena-relion passes the others by other_args ("Additional arguments").
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
        epoch: EPOCH = -1,
        anchors: ANCHORS = "",
        kmeans: KMEANS = -1,
        downsample: DOWNSAMPLE = 0,
        flip: FLIP = False,
        cryodrgn_bin_dir: CRYODRGN_BIN_DIR = "",
    ):
        bin_dir = self._resolve_bin_dir({"cryodrgn_bin_dir": cryodrgn_bin_dir})
        job_dir = self.output_job_dir.path
        plan = prepare_traversal(
            job_dir,
            epoch=epoch,
            anchors=anchors,
            kmeans=kmeans,
            downsample=downsample,
            flip=flip,
        )
        out = next_traversal_dir(job_dir)
        out.mkdir()
        (out / TRAVERSAL_INFO).write_text(json.dumps(plan.info(), indent=2))
        self.console.log(
            f"Graph traversal through {len(plan.anchors)} anchors ({plan.anchors_from},"
            f" epoch {plan.epoch}): {plan.anchors}. Volumes are written to {out.name}."
        )
        path = _impl.cryodrgn_graph_traversal(
            plan.z_file,
            plan.anchors,
            out,
            max_neighbors=plan.max_neighbors,
            avg_neighbors=plan.avg_neighbors,
            bin_dir=bin_dir,
        )
        yield
        _impl.cryodrgn_eval_vol(
            plan.weights,
            plan.config,
            path.z_path_file,
            out,
            apix=plan.apix,
            downsample=plan.downsample,
            flip=plan.flip,
            invert=plan.invert,
            bin_dir=bin_dir,
        )
