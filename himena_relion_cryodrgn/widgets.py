"""Job widgets of the cryoDRGN jobs, shown in the himena-relion job window."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from himena import StandardType, WidgetDataModel
from himena.standards import plotting as hplt
from himena_relion import _job_dir
from himena_relion._widgets import Q3DViewer, QPlotCanvas, spacer_widget
from himena_relion._widgets._shared.resizer import QResizer
from matplotlib import colormaps
from matplotlib.colors import to_hex
from qtpy import QtCore
from qtpy import QtWidgets as QtW
from starfile_rs import read_star

from himena_relion_cryodrgn import _analysis, _impl

_LOGGER = logging.getLogger(__name__)
_SELECTABLE = QtCore.Qt.TextInteractionFlag.TextSelectableByMouse


def _title(text: str) -> QtW.QLabel:
    return QtW.QLabel(f"<b>&#9679; {text}</b>")


def _is_state_file(path: Path) -> bool:
    return path.name.startswith("RELION_JOB_")


def _train_job_files(job_dir: _job_dir.JobDirectory):
    """Files of the CryoDRGN Train job given as `train_weights` of a job."""
    from himena_relion_cryodrgn.select_particles import TrainJobFiles

    weights = Path(job_dir.get_job_param("train_weights"))
    if not weights.is_absolute():
        # himena does not run in the project directory
        weights = job_dir.relion_project_dir / weights
    return TrainJobFiles.from_weights(weights)


def _latent_title(emb: _analysis.LatentEmbedding, extra: str = "") -> str:
    how = "UMAP" if emb.xlabel == "UMAP1" else "PCA of z"
    return f"<b>&#9679; Latent space</b> (epoch {emb.epoch}, {how}{extra})"


class QCryoDrgnPlotCanvas(QPlotCanvas):
    """Plots of cryoDRGN results, in the style of the himena-relion plots."""

    @classmethod
    def for_scatter(cls, parent: QtW.QWidget) -> QCryoDrgnPlotCanvas:
        """A canvas closer to square, for the latent space."""
        canvas = cls(parent)
        canvas.setMinimumSize(300, 300)
        canvas.setMaximumSize(400, 360)
        return canvas

    def _colors(self) -> tuple[str, str]:
        light = self._main_theme().is_light_background()
        return ("#ff7f0e" if light else "#ff69b4", "#1f77b4" if light else "#00ced1")

    def _show(self, fig: hplt.SingleAxes) -> None:
        self.update_model(WidgetDataModel(value=fig, type=StandardType.PLOT))
        self.tight_layout()

    @staticmethod
    def _scatter(fig: hplt.SingleAxes, x: np.ndarray, y: np.ndarray, **kwargs):
        """Scatter plot without marker edges."""
        fig.scatter(x, y, width=0, **kwargs)

    @staticmethod
    def _point_size(n: int) -> float:
        return 6.0 if n < 1000 else 3.0 if n < 5000 else 1.5

    def plot_loss(self, log: _analysis.TrainLog) -> None:
        fig = hplt.figure(self._main_theme().background)
        color_total, color_gen = self._colors()
        fig.plot(log.epochs, log.total_loss, name="Total", color=color_total)
        fig.plot(log.epochs, log.gen_loss, name="Reconstruction", color=color_gen)
        fig.x.label = "Epoch"
        fig.y.label = "Loss"
        # inside the axes, where the decreasing curves leave space
        fig.set_legend(location="top_right", font_size=9.0)
        self._show(fig)

    def plot_kld(self, log: _analysis.TrainLog) -> None:
        fig = hplt.figure(self._main_theme().background)
        fig.plot(log.epochs, log.kld, color=self._colors()[1])
        fig.x.label = "Epoch"
        fig.y.label = "KLD"
        self._show(fig)

    def plot_latent(
        self,
        emb: _analysis.LatentEmbedding,
        labels: np.ndarray | None = None,
        selected: np.ndarray | None = None,
    ) -> None:
        """Scatter plot of the particles, colored by k-means labels or selection."""
        fig = hplt.figure(self._main_theme().background)
        x, y = emb.xy[:, 0], emb.xy[:, 1]
        size = self._point_size(len(x))
        color_sel, color_all = self._colors()
        if selected is not None:
            self._scatter(
                fig, x[~selected], y[~selected], name="Not selected", size=size,
                color="#80808060",
            )  # fmt: skip
            self._scatter(
                fig, x[selected], y[selected], name="Selected", size=size,
                color=color_sel + "c0",
            )  # fmt: skip
        elif labels is not None and len(labels) == len(x):
            cmap = colormaps["tab20"]
            for i, label in enumerate(np.unique(labels)):
                mask = labels == label
                color = to_hex(cmap(i % cmap.N)) + "c0"
                self._scatter(fig, x[mask], y[mask], size=size, color=color)
        else:
            self._scatter(fig, x, y, size=size, color=color_all + "80")
        fig.x.label = emb.xlabel
        fig.y.label = emb.ylabel
        self._show(fig)


class QVolumeSeriesViewer(QtW.QWidget):
    """Choose a volume series and a volume of it. Only the shown volume is read."""

    def __init__(self):
        super().__init__()
        self._series: list[_analysis.VolumeSeries] = []
        self._source = QtW.QComboBox()
        self._source.setToolTip("Volume series to show")
        self._source.setFixedWidth(280)
        # the name of the volume shown is in self._info
        self._index = QtW.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._index.setRange(1, 1)
        self._index.setPageStep(1)
        self._index.setToolTip("Volume of the series (vol_001.mrc, ...)")
        self._viewer = Q3DViewer()
        self._viewer.setMinimumHeight(320)
        self._info = QtW.QLabel("")
        self._info.setTextInteractionFlags(_SELECTABLE)
        self._resizer = QResizer(self._viewer)
        row = QtW.QWidget()
        row_layout = QtW.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self._source, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        row_layout.addWidget(self._index, stretch=1)

        layout = QtW.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(row)
        layout.addWidget(self._viewer)
        layout.addWidget(self._info)
        layout.addWidget(self._resizer)
        self._source.currentIndexChanged.connect(lambda _: self._on_source_changed())
        self._index.valueChanged.connect(lambda _: self._show_volume())

    def set_series(self, series: list[_analysis.VolumeSeries]) -> None:
        """Set the volume series, keeping the current one (and volume) if it exists."""
        current = self._source.currentText()
        self._series = series
        names = [s.name for s in series]
        with QtCore.QSignalBlocker(self._source):
            self._source.clear()
            self._source.addItems(names)
            if current in names:
                self._source.setCurrentIndex(names.index(current))
        self._on_source_changed(reset_view=current not in names)

    def series_names(self) -> list[str]:
        return [s.name for s in self._series]

    def set_current(self, index: int) -> None:
        """Show the `index`-th volume series."""
        self._source.setCurrentIndex(index)

    def _current_series(self) -> _analysis.VolumeSeries | None:
        index = self._source.currentIndex()
        if 0 <= index < len(self._series):
            return self._series[index]
        return None

    def _on_source_changed(self, reset_view: bool = True):
        series = self._current_series()
        n_volumes = len(series.paths) if series else 0
        with QtCore.QSignalBlocker(self._index):
            self._index.setRange(1, max(n_volumes, 1))
        self._index.setVisible(n_volumes > 1)
        self._show_volume(reset_view=reset_view)

    def _show_volume(self, reset_view: bool = False):
        """Read and show the current volume (only this one is read)."""
        series = self._current_series()
        if series is None:
            self._viewer.set_image(None)
            self._info.setText("")
            return
        index = min(max(self._index.value(), 1), len(series.paths)) - 1
        path = series.paths[index]
        try:
            img, voxel_size = _analysis.read_volume(path)
        except Exception:  # e.g. still being written
            _LOGGER.debug("Failed to read %s", path, exc_info=True)
            self._viewer.set_image(None)
            self._info.setText(f"Cannot read {path.name}.")
            return
        if reset_view:
            self._viewer.set_image(img, update_now=False)
            self._viewer.auto_threshold(update_now=False)
            self._viewer.auto_fit()
        else:
            # keep the threshold, so that the volumes of a series can be compared
            self._viewer.set_image(img, update_clim=False)
        shape = "x".join(str(s) for s in img.shape)
        self._info.setText(f"{path.name}  ({shape}, {voxel_size:.3g} Å/pix)")


class QCryoDrgnTrainViewer(QtW.QWidget):
    """Progress and results of the CryoDRGN Train job.

    The learning curve is re-read from train/run.log every time a checkpoint
    (train/weights.{epoch}.pkl) is written: cryoDRGN logs the summary of an epoch
    before it saves the checkpoint.
    """

    def __init__(self, job_dir: _job_dir.JobDirectory):
        super().__init__()
        self._job_dir = job_dir
        self._status = QtW.QLabel("")
        self._status.setTextInteractionFlags(_SELECTABLE)
        self._loss_canvas = QCryoDrgnPlotCanvas(self)
        self._kld_canvas = QCryoDrgnPlotCanvas(self)
        self._latent_title = _title("Latent space")
        self._latent_canvas = QCryoDrgnPlotCanvas.for_scatter(self)
        # the back-projection and the volume series of the analysis (along each
        # principal component and at the k-means centers)
        self._volumes = QVolumeSeriesViewer()

        layout = QtW.QVBoxLayout(self)
        layout.setSpacing(2)
        layout.addWidget(self._status)
        layout.addWidget(_title("Loss"))
        layout.addWidget(self._loss_canvas)
        layout.addWidget(_title("KL divergence"))
        layout.addWidget(self._kld_canvas)
        layout.addWidget(self._latent_title)
        layout.addWidget(self._latent_canvas)
        layout.addWidget(_title("Volumes"))
        layout.addWidget(self._volumes)
        layout.addWidget(spacer_widget())

    @property
    def _workdir(self) -> Path:
        return self._job_dir.path / "train"

    def initialize(self, job_dir: _job_dir.JobDirectory):
        self._job_dir = job_dir
        self._update_learning_curve()
        self._update_latent()
        self._update_volume_sources()

    def on_job_updated(self, job_dir: _job_dir.JobDirectory, path):
        self.on_job_files_updated(job_dir, [path])

    def on_job_files_updated(self, job_dir: _job_dir.JobDirectory, fps: Iterable):
        self._job_dir = job_dir
        paths = [Path(fp) for fp in fps]
        state_changed = any(_is_state_file(p) for p in paths)
        if state_changed or any(self._is_checkpoint(p) for p in paths):
            self._update_learning_curve()
        if state_changed or any(self._changes_latent(p) for p in paths):
            self._update_latent()
        if state_changed or any(self._changes_volumes(p) for p in paths):
            self._update_volume_sources()

    def widget_added_callback(self):
        for canvas in (self._loss_canvas, self._kld_canvas, self._latent_canvas):
            canvas.widget_added_callback()

    def _is_checkpoint(self, path: Path) -> bool:
        return path.parent.name == "train" and _analysis.is_checkpoint_file(path)

    def _changes_latent(self, path: Path) -> bool:
        if path.parent.name == "train":
            return path.name.startswith("z.") and path.suffix == ".pkl"
        return path.name in ("umap.pkl", "labels.pkl") and any(
            part.startswith("analyze.") for part in path.parts
        )

    def _changes_volumes(self, path: Path) -> bool:
        if path.name == "backproject.mrc":
            return True
        if path.suffix != ".mrc":
            return False
        # graph traversal (traversal_NNN/vol_*.mrc) or analysis (analyze.N/.../*.mrc)
        return path.parent.name.startswith("traversal_") or any(
            part.startswith("analyze.") for part in path.parts
        )

    def _update_learning_curve(self):
        log = _analysis.parse_train_log(self._workdir / "run.log")
        if log.epochs:
            self._loss_canvas.plot_loss(log)
            self._kld_canvas.plot_kld(log)
        else:
            self._loss_canvas.clear()
            self._kld_canvas.clear()
        self._status.setText(self._status_text(log))

    def _status_text(self, log: _analysis.TrainLog) -> str:
        if log.finished:
            return f"Training complete ({log.last_epoch} epochs)."
        if log.num_epochs is None and not log.epochs:
            return "Preparing particles for training..."
        last = log.last_epoch or 0
        text = f"Epoch {last} / {log.num_epochs or '?'} finished."
        if log.epochs:
            text += f" Total loss: {log.total_loss[-1]:.5g}"
        return text

    def _update_latent(self):
        try:
            emb = _analysis.load_latent_2d(self._workdir)
        except Exception:
            _LOGGER.exception("Failed to load the latent space")
            emb = None
        if emb is None:
            self._latent_title.setText("<b>&#9679; Latent space</b>")
            self._latent_canvas.clear()
            return
        labels = _analysis.load_kmeans_labels(self._workdir, emb.epoch)
        extra = ""
        if labels is not None:
            extra = f", {len(np.unique(labels))} k-means clusters"
        self._latent_title.setText(_latent_title(emb, extra))
        self._latent_canvas.plot_latent(emb, labels=labels)

    def _update_volume_sources(self):
        self._volumes.set_series(_analysis.find_volume_series(self._job_dir.path))


class QCryoDrgnSelectViewer(QtW.QWidget):
    """Particles selected in the latent space by the CryoDRGN Select Particles job."""

    def __init__(self, job_dir: _job_dir.JobDirectory):
        super().__init__()
        self._job_dir = job_dir
        self._latent_title = _title("Latent space")
        self._summary = QtW.QLabel("")
        self._summary.setTextInteractionFlags(_SELECTABLE)
        self._latent_canvas = QCryoDrgnPlotCanvas.for_scatter(self)
        # the back-projection of the selected particles, and of all the particles
        # (CryoDRGN Train job) to compare with
        self._volumes = QVolumeSeriesViewer()

        layout = QtW.QVBoxLayout(self)
        layout.setSpacing(2)
        layout.addWidget(self._latent_title)
        layout.addWidget(self._summary)
        layout.addWidget(self._latent_canvas)
        layout.addWidget(_title("Volumes"))
        layout.addWidget(self._volumes)
        layout.addWidget(spacer_widget())

    def initialize(self, job_dir: _job_dir.JobDirectory):
        self._job_dir = job_dir
        self._update()

    def on_job_updated(self, job_dir: _job_dir.JobDirectory, path):
        self.on_job_files_updated(job_dir, [path])

    def on_job_files_updated(self, job_dir: _job_dir.JobDirectory, fps: Iterable):
        self._job_dir = job_dir
        names = {Path(fp).name for fp in fps}
        if names & {"indices.pkl", "particles.star", "backproject.mrc"} or any(
            n.startswith("RELION_JOB_") for n in names
        ):
            self._update()

    def widget_added_callback(self):
        self._latent_canvas.widget_added_callback()

    def _update_volumes(self, train_job_path: Path | None):
        series = _analysis.selection_volume_series(self._job_dir.path, train_job_path)
        names_before = self._volumes.series_names()
        self._volumes.set_series(series)
        # show the back-projection of the selection (the first one) once written
        if series and series[0].name not in names_before:
            self._volumes.set_current(0)

    def _update(self):
        job_dir = self._job_dir
        try:
            train = _train_job_files(job_dir)
        except (OSError, ValueError) as e:  # missing or another kind of job
            self._summary.setText(f"CryoDRGN Train job not found: {e}")
            self._latent_canvas.clear()
            self._update_volumes(None)
            return
        self._update_volumes(train.job_path)
        epoch = int(job_dir.get_job_param("epoch", "-1") or -1)
        emb = _analysis.load_latent_2d(train.workdir, epoch)
        selected_file = job_dir.path / "indices.pkl"
        selected_ids = (
            _impl.load_indices(selected_file) if selected_file.is_file() else None
        )

        if emb is None:
            self._latent_title.setText("<b>&#9679; Latent space</b>")
            self._latent_canvas.clear()
        else:
            self._latent_title.setText(_latent_title(emb))
            mask = None
            if selected_ids is not None:
                # plot points are the training particles; selected_ids refer to all
                # particles parsed for training (cryodrgn filter maps them back).
                ids = train.training_particle_ids(len(emb.xy))
                mask = np.isin(ids, selected_ids)
            self._latent_canvas.plot_latent(emb, selected=mask)

        if selected_ids is None:
            self._summary.setText(
                "Waiting for the selection in the cryoDRGN window. Select particles "
                "with the lasso and close the window."
            )
        else:
            # the particles selected from (in_parts), by default those of training
            particles_star = train.input_particles()
            if in_part := job_dir.get_job_param("in_part"):
                particles_star = job_dir.resolve_path(in_part)
            try:
                star = read_star(particles_star)
                n_total = str(star["particles"].trust_loop().shape[0])
            except Exception:  # moved or being written: the count is not essential
                _LOGGER.debug("Failed to read %s", particles_star, exc_info=True)
                n_total = "?"
            self._summary.setText(
                f"Selected: <b>{selected_ids.size}</b>/{n_total} particles"
            )
