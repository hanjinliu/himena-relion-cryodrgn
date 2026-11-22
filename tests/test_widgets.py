"""Job widgets of the CryoDRGN jobs (Qt, offscreen)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion.external import run_function

from himena_relion_cryodrgn import _analysis
from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob
from himena_relion_cryodrgn.widgets import (
    QCryoDrgnPlotCanvas,
    QCryoDrgnSelectViewer,
    QCryoDrgnTrainViewer,
)

if TYPE_CHECKING:
    from conftest import RelionProject


@pytest.fixture(autouse=True)
def _sync_viewers():
    """Render synchronously, as in the tests of himena-relion."""
    from himena_relion._impl_objects import set_is_testing
    from himena_relion._widgets._view_nd import QViewer

    set_is_testing(True)
    QViewer._always_force_sync = True


def _lines(canvas: QCryoDrgnPlotCanvas):
    return canvas.figure.axes[0].lines


def _scatters(canvas: QCryoDrgnPlotCanvas):
    return canvas.figure.axes[0].collections


def _train_widget(qtbot, project: RelionProject) -> QCryoDrgnTrainViewer:
    job_dir = ExternalJobDirectory(project.root / project.job_dir)
    widget = CryoDrgnTrainJob(job_dir).provide_widget(job_dir)
    assert isinstance(widget, QCryoDrgnTrainViewer)
    qtbot.addWidget(widget)
    widget.initialize(job_dir)
    return widget


def _epoch_lines(epoch: int, num_epochs: int = 20) -> str:
    return (
        f"# [Train Epoch: {epoch}/{num_epochs}] [100/100 particles] gen loss=0.5, "
        "kld=1.0, beta=0.025000, loss=0.525\n"
        f"# =====> Epoch: {epoch} Average gen loss = {1 / epoch:.6}, KLD = 1.000000, "
        f"total loss = {1 / epoch + 0.025:.6f}; Finished in 0:00:01\n"
    )


def test_train_widget_before_training(qtbot, relion_project: RelionProject):
    widget = _train_widget(qtbot, relion_project)

    assert widget._status.text() == "Preparing particles for training..."
    assert len(_lines(widget._loss_canvas)) == 0
    assert len(_scatters(widget._latent_canvas)) == 0
    assert not widget._viewer.has_image


def test_train_widget_after_training(qtbot, trained_project: RelionProject):
    widget = _train_widget(qtbot, trained_project)

    assert widget._status.text() == "Training complete (20 epochs)."
    total, gen = _lines(widget._loss_canvas)
    assert len(total.get_xdata()) == len(gen.get_xdata()) == 20
    (kld,) = _lines(widget._kld_canvas)
    assert list(kld.get_xdata()) == list(range(1, 21))
    # UMAP of the automatic analysis, one scatter per k-means cluster
    assert "epoch 20, UMAP, 4 k-means clusters" in widget._latent_title.text()
    assert len(_scatters(widget._latent_canvas)) == 4
    n_points = sum(len(c.get_offsets()) for c in _scatters(widget._latent_canvas))
    assert n_points == 4  # training particles (TS_01/2 and TS_01/4 excluded)
    assert widget._viewer.has_image  # back-projection
    widget.widget_added_callback()


def test_learning_curve_updates_with_checkpoints(
    qtbot, relion_project: RelionProject, monkeypatch: pytest.MonkeyPatch
):
    widget = _train_widget(qtbot, relion_project)
    n_parsed = []
    parse = _analysis.parse_train_log
    monkeypatch.setattr(
        _analysis, "parse_train_log", lambda p: n_parsed.append(p) or parse(p)
    )
    train = relion_project.job_file("train")
    train.mkdir()
    run_log = train / "run.log"
    run_log.write_text("".join(_epoch_lines(e) for e in (1, 2, 3)))
    (train / "weights.3.pkl").touch()

    widget.on_job_files_updated(widget._job_dir, [train / "weights.3.pkl"])

    assert len(_lines(widget._loss_canvas)[0].get_xdata()) == 3
    assert widget._status.text().startswith("Epoch 3 / 20 finished.")

    # run.log changes during an epoch: the plot is not updated
    with open(run_log, "a") as f:
        f.write(_epoch_lines(4))
    widget.on_job_files_updated(widget._job_dir, [run_log, train / "z.3.pkl.tmp"])
    assert len(n_parsed) == 1
    assert len(_lines(widget._loss_canvas)[0].get_xdata()) == 3

    # ... until the checkpoint of the epoch is written
    (train / "weights.4.pkl").touch()
    widget.on_job_files_updated(widget._job_dir, [str(train / "weights.4.pkl")])
    assert len(n_parsed) == 2
    assert len(_lines(widget._loss_canvas)[0].get_xdata()) == 4
    assert widget._status.text().startswith("Epoch 4 / 20 finished.")


def test_train_widget_shows_latent_during_training(
    qtbot, trained_project: RelionProject
):
    """Before the analysis exists, the PCA of the last z is shown."""
    import shutil

    shutil.rmtree(trained_project.job_file("train", "analyze.20"))
    widget = _train_widget(qtbot, trained_project)

    assert "epoch 20, PCA of z" in widget._latent_title.text()
    assert len(_scatters(widget._latent_canvas)) == 1


def _write_select_job_star(project: RelionProject) -> None:
    """RELION writes job.star when the job is created."""
    CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights=f"{project.job_dir.as_posix()}/weights.pkl", epoch=-1, kmeans=-1
    ).to_star_dict().write(project.select_job_file("job.star"))


def _select_widget(qtbot, project: RelionProject) -> QCryoDrgnSelectViewer:
    job_dir = ExternalJobDirectory(project.root / project.select_job_dir)
    widget = CryoDrgnSelectParticlesJob(job_dir).provide_widget(job_dir)
    assert isinstance(widget, QCryoDrgnSelectViewer)
    qtbot.addWidget(widget)
    widget.initialize(job_dir)
    return widget


def _table(widget: QCryoDrgnSelectViewer) -> list[tuple[str, ...]]:
    t = widget._table
    return [
        tuple(t.item(r, c).text() for c in range(t.columnCount()))
        for r in range(t.rowCount())
    ]


def test_select_widget_waiting(qtbot, trained_project: RelionProject):
    _write_select_job_star(trained_project)
    widget = _select_widget(qtbot, trained_project)

    assert widget._summary.text().startswith("Waiting for the selection")
    assert len(_scatters(widget._latent_canvas)) == 1  # all training particles
    assert _table(widget) == [("TS_01", "0", "6")]


def test_select_widget_after_selection(
    qtbot,
    trained_project: RelionProject,
    fake_cryodrgn: FakeCryoDRGN,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_select_job_star(trained_project)
    widget = _select_widget(qtbot, trained_project)
    fake_cryodrgn.filter_selection = [1, 2]  # points of TS_01/3 and TS_01/5
    run_function(
        [
            CryoDrgnSelectParticlesJob.import_path(),
            "--o", f"{trained_project.select_job_dir.as_posix()}/",
            "--train_weights", f"{trained_project.job_dir.as_posix()}/weights.pkl",
        ]
    )  # fmt: skip

    # himena does not run in the project directory: paths must not depend on cwd
    monkeypatch.chdir(Path(trained_project.root).parent)
    widget.on_job_files_updated(
        widget._job_dir,
        [
            trained_project.select_job_file("indices.pkl"),
            trained_project.select_job_file("particles.star"),
        ],
    )

    assert widget._summary.text() == "<b>2</b> of 6 particles selected."
    not_selected, selected = _scatters(widget._latent_canvas)
    # the lassoed points, not just any two points
    umap = _analysis.load_latent_2d(trained_project.job_file("train")).xy
    np.testing.assert_allclose(selected.get_offsets(), umap[[1, 2]], rtol=1e-5)
    np.testing.assert_allclose(not_selected.get_offsets(), umap[[0, 3]], rtol=1e-5)
    assert _table(widget) == [("TS_01", "2", "6")]
    widget.widget_added_callback()


def test_select_widget_without_train_job(qtbot, relion_project: RelionProject):
    CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights="External/job099/weights.pkl"
    ).to_star_dict().write(relion_project.select_job_file("job.star"))
    widget = _select_widget(qtbot, relion_project)

    assert widget._summary.text().startswith("CryoDRGN Train job not found")
    assert _table(widget) == []


def _volume_sources(widget: QCryoDrgnTrainViewer) -> list[str]:
    combo = widget._volume_source
    return [combo.itemText(i) for i in range(combo.count())]


def test_volume_viewer_reads_volumes_lazily(
    qtbot, trained_project: RelionProject, monkeypatch: pytest.MonkeyPatch
):
    reads = []
    read_volume = _analysis.read_volume
    monkeypatch.setattr(
        _analysis, "read_volume", lambda p: reads.append(Path(p).name) or read_volume(p)
    )
    widget = _train_widget(qtbot, trained_project)

    assert _volume_sources(widget) == [
        "Back-projection (all particles)",
        "PC1 (epoch 20)",
        "PC2 (epoch 20)",
        "k-means 20 (epoch 20)",
    ]
    assert reads == ["backproject.mrc"]  # not the 40 volumes of the analysis
    assert widget._volume_index.isHidden()  # a single volume

    widget._volume_source.setCurrentIndex(1)  # PC1
    assert not widget._volume_index.isHidden()
    assert widget._volume_index.maximum() == 10
    assert reads == ["backproject.mrc", "vol_001.mrc"]

    widget._volume_index.setValue(3)
    assert reads[-1] == "vol_003.mrc"
    assert len(reads) == 3
    assert widget._volume_info.text().startswith("vol_003.mrc  (9x9x9, 5.4 Å/pix)")
    assert widget._viewer.has_image

    widget._volume_source.setCurrentIndex(3)  # k-means
    assert widget._volume_index.maximum() == 20


def test_volume_viewer_keeps_selection_on_update(qtbot, trained_project: RelionProject):
    widget = _train_widget(qtbot, trained_project)
    widget._volume_source.setCurrentIndex(2)  # PC2
    widget._volume_index.setValue(7)

    # e.g. `cryodrgn analyze` wrote another volume
    vol = trained_project.job_file("train", "analyze.20", "pc2", "vol_007.mrc")
    widget.on_job_files_updated(widget._job_dir, [vol])

    assert widget._volume_source.currentText() == "PC2 (epoch 20)"
    assert widget._volume_index.value() == 7
    assert widget._volume_info.text().startswith("vol_007.mrc")


def test_volume_viewer_before_training(qtbot, relion_project: RelionProject):
    widget = _train_widget(qtbot, relion_project)
    assert _volume_sources(widget) == []
    assert widget._volume_info.text() == ""
    assert not widget._viewer.has_image
