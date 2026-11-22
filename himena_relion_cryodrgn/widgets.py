"""Job widgets of the cryoDRGN jobs, shown in the himena-relion job window."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from himena import StandardType, WidgetDataModel
from himena.standards import plotting as hplt
from himena_relion import _job_dir
from himena_relion._widgets import Q3DViewer, QIntWidget, QPlotCanvas, spacer_widget, QMicrographListWidget
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
        # pyright does not see the defaults of the other fields of this pydantic model
        model = WidgetDataModel(value=fig, type=StandardType.PLOT)  # pyright: ignore[reportCallIssue]
        self.update_model(model)
        self.tight_layout()

    @staticmethod
    def _scatter(fig: hplt.SingleAxes, x: np.ndarray, y: np.ndarray, **kwargs):
        # hplt annotates Sequence[float], but numpy arrays are accepted (no copy).
        fig.scatter(x, y, width=0, **kwargs)  # pyright: ignore[reportArgumentType]

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
        size = 6.0 if len(x) < 1000 else 3.0 if len(x) < 5000 else 1.5
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
            # above the axes, not to hide points nor to narrow the plot
            fig.set_legend(location="top_side_right", font_size=9.0)
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
        # Volumes: the back-projection and the volume series of the analysis (along
        # each principal component and at the k-means centers), read when shown.
        self._series: list[_analysis.VolumeSeries] = []
        self._volume_source = QtW.QComboBox()
        self._volume_source.setToolTip("Back-projection or volume series to show")
        self._volume_index = QIntWidget("Volume", label_width=50)
        self._volume_index.setToolTip("Volume of the series (vol_001.mrc, ...)")
        self._viewer = Q3DViewer()
        self._viewer.setMinimumHeight(240)
        self._volume_info = QtW.QLabel("")
        self._volume_info.setTextInteractionFlags(_SELECTABLE)
        self._resizer = QResizer(self._viewer)
        volume_row = QtW.QWidget()
        volume_row_layout = QtW.QHBoxLayout(volume_row)
        volume_row_layout.setContentsMargins(0, 0, 0, 0)
        volume_row_layout.addWidget(self._volume_source, stretch=1)
        volume_row_layout.addWidget(self._volume_index)

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
        layout.addWidget(volume_row)
        layout.addWidget(self._viewer)
        layout.addWidget(self._volume_info)
        layout.addWidget(self._resizer)
        layout.addWidget(spacer_widget())
        self._volume_source.currentIndexChanged.connect(
            lambda _: self._on_volume_source_changed()
        )
        self._volume_index.valueChanged.connect(lambda _: self._show_volume())

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
        return path.suffix == ".mrc" and any(
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
        how = "UMAP" if emb.xlabel == "UMAP1" else "PCA of z"
        if labels is not None:
            how += f", {len(np.unique(labels))} k-means clusters"
        self._latent_title.setText(
            f"<b>&#9679; Latent space</b> (epoch {emb.epoch}, {how})"
        )
        self._latent_canvas.plot_latent(emb, labels=labels)

    def _update_volume_sources(self):
        """Re-scan the volume series, keeping the current one if it still exists."""
        current = self._volume_source.currentText()
        self._series = _analysis.find_volume_series(self._job_dir.path)
        names = [s.name for s in self._series]
        with QtCore.QSignalBlocker(self._volume_source):
            self._volume_source.clear()
            self._volume_source.addItems(names)
            if current in names:
                self._volume_source.setCurrentIndex(names.index(current))
        self._on_volume_source_changed(reset_view=current not in names)

    def _current_series(self) -> _analysis.VolumeSeries | None:
        index = self._volume_source.currentIndex()
        if 0 <= index < len(self._series):
            return self._series[index]
        return None

    def _on_volume_source_changed(self, reset_view: bool = True):
        series = self._current_series()
        n_volumes = len(series.paths) if series else 0
        with QtCore.QSignalBlocker(self._volume_index):
            self._volume_index.setRange(1, max(n_volumes, 1))
        self._volume_index.setVisible(n_volumes > 1)
        self._show_volume(reset_view=reset_view)

    def _show_volume(self, reset_view: bool = False):
        """Read and show the current volume (only this one is read)."""
        series = self._current_series()
        if series is None:
            self._viewer.set_image(None)
            self._volume_info.setText("")
            return
        index = min(max(self._volume_index.value(), 1), len(series.paths)) - 1
        path = series.paths[index]
        try:
            img, voxel_size = _analysis.read_volume(path)
        except Exception:  # e.g. still being written
            _LOGGER.debug("Failed to read %s", path, exc_info=True)
            self._viewer.set_image(None)
            self._volume_info.setText(f"Cannot read {path.name}.")
            return
        if reset_view:
            self._viewer.set_image(img, update_now=False)
            self._viewer.auto_threshold(update_now=False)
            self._viewer.auto_fit()
        else:
            # keep the threshold, so that the volumes of a series can be compared
            self._viewer.set_image(img, update_clim=False)
        shape = "x".join(str(s) for s in img.shape)
        self._volume_info.setText(f"{path.name}  ({shape}, {voxel_size:.3g} Å/pix)")


class QCryoDrgnSelectViewer(QtW.QWidget):
    """Particles selected in the latent space by the CryoDRGN Select Particles job."""

    def __init__(self, job_dir: _job_dir.JobDirectory):
        super().__init__()
        self._job_dir = job_dir
        self._latent_title = _title("Latent space")
        self._summary = QtW.QLabel("")
        self._summary.setTextInteractionFlags(_SELECTABLE)
        self._latent_canvas = QCryoDrgnPlotCanvas.for_scatter(self)

        layout = QtW.QVBoxLayout(self)
        layout.setSpacing(2)
        layout.addWidget(self._latent_title)
        layout.addWidget(self._summary)
        layout.addWidget(self._latent_canvas)
        layout.addWidget(_title("Particles per tomogram"))
        layout.addWidget(spacer_widget())

    def initialize(self, job_dir: _job_dir.JobDirectory):
        self._job_dir = job_dir
        self._update()

    def on_job_updated(self, job_dir: _job_dir.JobDirectory, path):
        self.on_job_files_updated(job_dir, [path])

    def on_job_files_updated(self, job_dir: _job_dir.JobDirectory, fps: Iterable):
        self._job_dir = job_dir
        names = {Path(fp).name for fp in fps}
        if names & {"indices.pkl", "particles.star"} or any(
            n.startswith("RELION_JOB_") for n in names
        ):
            self._update()

    def widget_added_callback(self):
        self._latent_canvas.widget_added_callback()

    def _train_job_files(self):
        from himena_relion_cryodrgn.select_particles import TrainJobFiles

        weights = Path(self._job_dir.get_job_param("train_weights"))
        if not weights.is_absolute():
            weights = self._job_dir.relion_project_dir / weights
        return TrainJobFiles.from_weights(weights)

    def _update(self):
        job_dir = self._job_dir
        try:
            train = self._train_job_files()
        except (OSError, ValueError) as e:  # missing or another kind of job
            self._summary.setText(f"CryoDRGN Train job not found: {e}")
            self._latent_canvas.clear()
            return
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
            how = "UMAP" if emb.xlabel == "UMAP1" else "PCA of z"
            self._latent_title.setText(
                f"<b>&#9679; Latent space</b> (epoch {emb.epoch}, {how})"
            )
            mask = None
            if selected_ids is not None:
                # plot points are the training particles; selected_ids refer to all
                # particles parsed for training (cryodrgn filter maps them back).
                ids = train.training_particle_ids(len(emb.xy))
                mask = np.isin(ids, selected_ids)
            self._latent_canvas.plot_latent(emb, selected=mask)

        n_total = read_star(train.particles_star)["particles"].trust_loop().shape[0]
        n_selected = read_star(self._job_dir.path / "particles.star")["particles"].trust_loop().shape[0]
        if selected_ids is None:
            self._summary.setText(
                "Waiting for the selection in the cryoDRGN window. Select particles "
                "with the lasso and close the window."
            )
        else:
            self._summary.setText(
                f"Selected: <b>{n_selected}</b>/{n_total} particles"
            )
